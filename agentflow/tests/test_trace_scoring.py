"""Exercise GT teacher forcing preparation without loading Ray or VERL."""

import ast
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import Mock
import torch
import argparse
import contextlib
import io
import os
import subprocess
import types
from unittest import mock
import re
import warnings


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("trace_scoring_under_test", ROOT / "agentflow/agentflow/trace/credit.py")
scoring = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scoring
spec.loader.exec_module(scoring)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        assert tokenize and add_generation_prompt
        return [1] + [ord(c) for c in messages[0]["content"]] + [2]


class BatchTokenizer(Tokenizer):
    """Model the HF fast tokenizer's scalar and batched template contract."""

    is_fast = True

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        self.calls.append(messages)
        if isinstance(messages[0], list):
            return [super(BatchTokenizer, self).apply_chat_template(
                conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
            ) for conversation in messages]
        return super().apply_chat_template(messages, tokenize=tokenize,
                                           add_generation_prompt=add_generation_prompt)


class ScorePreparationTests(unittest.TestCase):
    def encode(self, contexts, gts, **kw):
        params = dict(max_prompt_length=1000, max_gt_tokens=256, answer_instruction="Answer.")
        params.update(kw)
        return scoring.encode_state_requests(contexts, gts, Tokenizer(), **params)

    def test_canonical_alias_and_deduplication_do_not_leak_gt_into_prompt(self):
        requests, mapping, counts = self.encode(["question", "question", "next state"], [["", "SECRET", "alias"]] * 3)
        self.assertEqual(mapping, [0, 0, 1])
        self.assertEqual(len(requests), 2)
        self.assertNotIn("deduplicated_states", counts)
        self.assertEqual(requests[0]["target_ids"], Tokenizer().encode("SECRET"))
        self.assertNotIn("SECRET", "".join(chr(t) for t in requests[0]["prompt_ids"]))

    def test_missing_and_long_states_or_targets_skip_instead_of_truncating(self):
        requests, mapping, counts = self.encode(["q", "q", "q", "long context", None],
                                                [None, "", "very long GT", "a", "a"],
                                                max_prompt_length=12, max_gt_tokens=2)
        self.assertEqual(requests, [])
        self.assertEqual(mapping, [None] * 5)
        self.assertEqual(counts["missing_gt"], 2)
        self.assertEqual(counts["invalid_gt_length"], 1)
        self.assertEqual(counts["overlong_state"], 1)
        self.assertEqual(counts["missing_state"], 1)

    def test_fast_batch_and_scalar_paths_preserve_aliases_instructions_and_row_counts(self):
        contexts = [" q ", "q", "q", "next", "q", "long" * 10, "long" * 10, None, "q", "q"]
        gts = [["", "answer", "alias"], "answer", "other", "answer", "answer", "a", "a", "a", None, "x" * 20]
        instructions = [None, None, None, None, "Other.", None, None, None, None, ""]
        params = dict(max_prompt_length=30, max_gt_tokens=10, answer_instruction="Answer.",
                      answer_instructions=instructions)
        scalar = scoring.encode_state_requests(contexts, gts, Tokenizer(), **params)
        tokenizer = BatchTokenizer()
        batched = scoring.encode_state_requests(contexts, gts, tokenizer, **params)
        self.assertEqual(batched, scalar)
        requests, mapping, counts = batched
        self.assertEqual(mapping[:5], [0, 0, 1, 2, 3])
        self.assertEqual(mapping[5:], [None] * 5)
        self.assertEqual(counts["overlong_state"], 2)
        self.assertEqual(counts["missing_state"], 1)
        self.assertEqual(counts["missing_gt"], 1)
        self.assertEqual(counts["invalid_gt_length"], 1)
        self.assertEqual(len(requests), 4)
        self.assertEqual(len(tokenizer.calls), 1)
        # Different GT strings share prompt encoding, while each instruction
        # remains part of the actual prompt. The two overlong rows share work.
        self.assertEqual(len(tokenizer.calls[0]), 4)

    def test_scalar_tokenizer_encodes_a_shared_prompt_once_even_for_different_targets(self):
        tokenizer = Tokenizer()
        tokenizer.apply_chat_template = Mock(wraps=tokenizer.apply_chat_template)
        requests, mapping, _ = scoring.encode_state_requests(
            ["q", " q ", "q"], ["a", "b", "a"], tokenizer,
            max_prompt_length=30, max_gt_tokens=10, answer_instruction="Answer.",
        )
        self.assertEqual(mapping, [0, 1, 0])
        self.assertEqual(len(requests), 2)
        self.assertEqual(tokenizer.apply_chat_template.call_count, 1)

    def test_different_texts_that_tokenize_identically_still_deduplicate(self):
        class NormalizingTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return super().apply_chat_template(
                    [{"role": "user", "content": messages[0]["content"].lower()}], **kwargs)
        requests, mapping, counts = scoring.encode_state_requests(
            ["STATE", "state"], ["a", "a"], NormalizingTokenizer(),
            max_prompt_length=30, max_gt_tokens=10, answer_instruction="Answer.",
        )
        self.assertEqual(mapping, [0, 0])
        self.assertEqual(len(requests), 1)
        self.assertEqual(counts["scoring_requests"], 1)

    def test_fast_tokenizer_empty_input_does_not_call_template(self):
        tokenizer = BatchTokenizer()
        requests, mapping, counts = scoring.encode_state_requests(
            [None, "q"], ["a", None], tokenizer,
            max_prompt_length=30, max_gt_tokens=10, answer_instruction="Answer.",
        )
        self.assertEqual(requests, [])
        self.assertEqual(mapping, [None, None])
        self.assertEqual(tokenizer.calls, [])
        self.assertEqual(counts["scoring_requests"], 0)

    def test_left_prompt_right_target_padding_and_position_ids(self):
        requests = [{"prompt_ids": [1, 2], "target_ids": [4]},
                    {"prompt_ids": [1], "target_ids": [4, 5]}]
        tensors = scoring.build_score_tensors(requests, 0)
        self.assertEqual(tensors["input_ids"].tolist(), [[1, 2, 4, 0], [0, 1, 4, 5]])
        self.assertEqual(tensors["attention_mask"].tolist(), [[1, 1, 1, 0], [0, 1, 1, 1]])
        self.assertEqual(tensors["position_ids"].tolist(), [[0, 1, 2, 2], [0, 0, 1, 2]])

    def test_bulk_tensor_construction_matches_ragged_teacher_forcing_with_real_pad_tokens(self):
        requests = [{"prompt_ids": [9, 0, 4], "target_ids": [0, 7]},
                    {"prompt_ids": [0], "target_ids": [8]},
                    {"prompt_ids": [1, 2], "target_ids": [3, 0, 5, 6]}]
        tensors = scoring.build_score_tensors(requests, 0)
        for key, tensor in tensors.items():
            self.assertEqual(tensor.dtype, torch.long, key)
            self.assertEqual(tensor.device.type, "cpu", key)
        self.assertEqual(tensors["prompts"].tolist(), [[9, 0, 4], [0, 0, 0], [0, 1, 2]])
        self.assertEqual(tensors["responses"].tolist(), [[0, 7, 0, 0], [8, 0, 0, 0], [3, 0, 5, 6]])
        self.assertEqual(tensors["response_mask"].tolist(), [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(tensors["attention_mask"].tolist(),
                         [[1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1]])
        self.assertEqual(tensors["position_ids"].tolist(),
                         [[0, 1, 2, 3, 4, 4, 4], [0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 2, 3, 4, 5]])
        # Tensor fields were separate before the optimization. Keep writes to
        # one field from modifying input_ids behind another consumer's back.
        original_inputs = tensors["input_ids"].clone()
        tensors["prompts"][0, 0] = 99
        tensors["responses"][0, 0] = 99
        self.assertTrue(torch.equal(tensors["input_ids"], original_inputs))

    def test_logp_mean_masks_padding_and_preserves_invalid_rows(self):
        means = scoring.mean_gt_logps(
            torch.tensor([[-2., float("nan")], [-2., -4.], [0.1, -2.], [float("nan"), -2.]]),
            torch.tensor([[1, 0], [1, 1], [1, 1], [1, 1]]),
        )
        self.assertEqual(means[:2], [-2., -3.])
        self.assertTrue(all(math.isnan(v) for v in means[2:]))

    def test_real_reference_worker_uses_frozen_ref_and_untempered_gt_only(self):
        path = ROOT / "agentflow/verl/diy/workers/fsdp_workers.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActorRolloutRefWorker")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_ref_log_prob")
        fn.decorator_list = []
        calls = []
        class Proto:
            def __init__(self, **meta):
                self.meta_info = meta
            def to(self, device):
                return self
            @classmethod
            def from_dict(cls, tensors):
                return cls()
        scope = {"DataProto": Proto}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
        def ref_compute(data, calculate_entropy):
            calls.append(data.meta_info["temperature"])
            return torch.zeros(1, 1), None
        worker = SimpleNamespace(
            _is_lora=False, _is_ref=True, world_size=1,
            config=SimpleNamespace(rollout=SimpleNamespace(temperature=0.7),
                ref=SimpleNamespace(log_prob_micro_batch_size_per_gpu=1,
                                    log_prob_max_token_len_per_gpu=1024, log_prob_use_dynamic_bsz=False)),
            ulysses_sharding_manager=nullcontext(), ref_policy=SimpleNamespace(compute_log_prob=ref_compute),
        )
        scope[fn.name](worker, Proto(trace_gt_scoring=True))
        scope[fn.name](worker, Proto())
        self.assertEqual(calls, [1.0, 0.7])

    def test_lora_gt_probe_disables_adapter_and_uses_temperature_one(self):
        path = ROOT / "agentflow/verl/diy/workers/fsdp_workers.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "ActorRolloutRefWorker")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("compute_log_prob", "compute_ref_log_prob")]
        for fn in methods:
            fn.decorator_list = []
        class Proto:
            def __init__(self, batch=None, meta_info=None):
                self.batch, self.meta_info = batch or {}, meta_info or {}
            def to(self, device):
                return self
            @classmethod
            def from_dict(cls, tensors, meta_info=None):
                return cls(tensors, meta_info)
        scope = {"DataProto": Proto}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), scope)
        disabled = []
        @contextmanager
        def disable_adapter():
            disabled.append(True)
            try:
                yield
            finally:
                disabled.pop()
        calls = []
        def compute(data, calculate_entropy):
            calls.append((bool(disabled), data.meta_info["temperature"]))
            return torch.zeros(1, 1), torch.zeros(1, 1)
        worker = SimpleNamespace(
            _is_lora=True, _is_actor=True, _is_offload_param=False, world_size=1,
            config=SimpleNamespace(rollout=SimpleNamespace(temperature=0.7,
                log_prob_micro_batch_size_per_gpu=1, log_prob_max_token_len_per_gpu=1024,
                log_prob_use_dynamic_bsz=False)),
            ulysses_sharding_manager=nullcontext(),
            actor=SimpleNamespace(actor_module=SimpleNamespace(disable_adapter=disable_adapter),
                                  compute_log_prob=compute),
        )
        worker.compute_log_prob = MethodType(scope["compute_log_prob"], worker)
        output = scope["compute_ref_log_prob"](worker, Proto(meta_info={"trace_gt_scoring": True}))
        self.assertEqual(calls, [(True, 1.0)])
        self.assertIn("ref_log_prob", output.batch)
        self.assertFalse(disabled)

    def test_trace_creates_reference_worker_even_with_both_kl_switches_off(self):
        path = ROOT / "agentflow/verl/entrypoint.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "TaskRunner")
        run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "trace_enabled" for t in n.targets))
        code = compile(ast.Module(body=run.body[start:start + 3], type_ignores=[]), str(path), "exec")
        for algorithm, enabled, expected in (("grpo+slic", True, True),
                                              ("grpo", True, False), ("slic", False, False)):
            scope = {"config": SimpleNamespace(data={"algorithm": algorithm, "trace": {"enabled": enabled}},
                         algorithm=SimpleNamespace(use_kl_in_reward=False),
                         actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(strategy="fsdp", use_kl_loss=False))),
                     "ray": SimpleNamespace(remote=lambda cls: cls), "ActorRolloutRefWorker": object,
                     "Role": SimpleNamespace(RefPolicy="ref"), "role_worker_mapping": {},
                     "mapping": {}, "global_pool_id": "pool"}
            exec(code, scope)
            self.assertEqual("ref" in scope["role_worker_mapping"], expected)


class TraceStartupTests(unittest.TestCase):
    def make_runner(self, algorithm, trace_config):
        # Execute the real startup body up to model loading, without Ray/GPU
        # dependencies. The OmegaConf shim only supplies container operations.
        path = ROOT / "agentflow/verl/entrypoint.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "TaskRunner")
        run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        run.body = [n for n in run.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        class ConfigDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        config = SimpleNamespace(data=ConfigDict(algorithm=algorithm, trace=dict(trace_config)),
                                 actor_rollout_ref=SimpleNamespace(
                                     model=ConfigDict(path="model", lora_rank=0),
                                     actor=SimpleNamespace(strategy="fsdp")))
        copy_model = Mock(side_effect=RuntimeError("model loading reached"))
        scope = {"TraceConfig": scoring.TraceConfig, "asdict": asdict,
                 "copy_to_local": copy_model, "pprint": Mock(),
                 "OmegaConf": SimpleNamespace(resolve=lambda c: None, create=dict,
                     to_container=lambda c, resolve: c)}
        exec(compile(ast.Module(body=[run], type_ignores=[]), str(path), "exec"), scope)
        return lambda: scope["run"](None, config), config, copy_model

    def test_legacy_settings_are_normalized_before_model_loading(self):
        run, config, copy_model = self.make_runner("grpo+slic", {
            "enabled": True, "pair_score_mode": "igpo", "igpo_gamma": 1.0,
            "k": 3, "gamma": 0.8, "terminal_fill_scale": 2.0,
        })
        with self.assertWarnsRegex(UserWarning, "does not enable IGPO"):
            with self.assertRaisesRegex(RuntimeError, "model loading reached"):
                run()
        copy_model.assert_called_once_with("model")
        self.assertNotIn("pair_score_mode", config.data.trace)
        self.assertNotIn("igpo_gamma", config.data.trace)
        self.assertEqual(scoring.TraceConfig.from_mapping(config.data.trace), scoring.TraceConfig())

    def test_bad_trace_config_fails_before_loading_models(self):
        for bad in ({"k": 0}, {"match_epslion": 0.05}, {"pair_score_mode": "igop"}):
            with self.subTest(bad=bad):
                run, _, copy_model = self.make_runner("grpo+slic", {"enabled": True, **bad})
                with self.assertRaises(ValueError):
                    run()
                copy_model.assert_not_called()

    def test_moving_reference_rejects_unsupported_workers_before_model_load(self):
        for strategy, lora_rank in (("megatron", 0), ("fsdp", 8), ("fsdp2", 8)):
            run, config, copy_model = self.make_runner("gigpo+slic", {
                "enabled": True, "reference_update_interval": None})
            config.actor_rollout_ref.actor.strategy = strategy
            config.actor_rollout_ref.model.lora_rank = lora_rank
            with self.assertRaisesRegex(ValueError, "reference refresh"):
                run()
            copy_model.assert_not_called()

    def test_moving_reference_accepts_full_model_fsdp_and_fsdp2(self):
        for strategy in ("fsdp", "fsdp2"):
            run, config, copy_model = self.make_runner("grpo+slic", {
                "enabled": True, "reference_update_interval": "None"})
            config.actor_rollout_ref.actor.strategy = strategy
            with self.assertRaisesRegex(RuntimeError, "model loading reached"):
                run()
            self.assertIsNone(config.data.trace["reference_update_interval"])
            copy_model.assert_called_once()

    def test_inactive_trace_settings_remain_untouched(self):
        for algorithm, enabled in (("grpo", True), ("grpo+slic", False)):
            settings = {"enabled": enabled, "unknown": 1}
            with self.subTest(algorithm=algorithm, enabled=enabled):
                run, config, _ = self.make_runner(algorithm, settings)
                with self.assertRaisesRegex(RuntimeError, "model loading reached"):
                    run()
                self.assertEqual(config.data.trace, settings)


PAIRING_ROOT = Path(__file__).resolve().parents[2]
PAIRING_SPEC = importlib.util.spec_from_file_location(
    "trace_pairing_config_test_credit", PAIRING_ROOT / "agentflow/agentflow/trace/credit.py")
PAIRING_CREDIT = importlib.util.module_from_spec(PAIRING_SPEC)
sys.modules[PAIRING_SPEC.name] = PAIRING_CREDIT
PAIRING_SPEC.loader.exec_module(PAIRING_CREDIT)

try:
    import yaml
except ImportError:
    yaml = None


def load_pairing_function(path, name, scope):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


@unittest.skipIf(yaml is None, "PyYAML is unavailable")
class PairingSwitchLaunchTests(unittest.TestCase):
    def launch_config(self, filename, environment=None, overrides=()):
        # Execute the production environment loader and argument builder without
        # importing their model clients or launching a distributed process.
        loader = load_pairing_function(PAIRING_ROOT / "train-roma/utils.py", "load_and_set_env_from_yaml",
                               {"os": os, "yaml": yaml, "sys": sys})
        utils = types.ModuleType("utils")
        utils.load_and_set_env_from_yaml = lambda _: loader(str(PAIRING_ROOT / "train-roma" / filename))
        runner = mock.Mock()
        main = load_pairing_function(PAIRING_ROOT / "train-roma/train_agent.py", "main", {
            "os": os, "sys": sys, "argparse": argparse,
            "subprocess": types.SimpleNamespace(run=runner,
                                                CalledProcessError=subprocess.CalledProcessError),
        })
        with mock.patch.dict(os.environ, environment or {}, clear=True), \
             mock.patch.dict(sys.modules, {"utils": utils}), \
             mock.patch.object(sys, "argv", ["train_agent.py", *overrides]), \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        runner.assert_called_once()
        command = runner.call_args.args[0]
        self.assertEqual(command[:3], ["python", "-m", "agentflow.verl"])
        values = yaml.safe_load((PAIRING_ROOT / "agentflow/verl/config.yaml").read_text())["data"]["trace"]
        # Hydra receives these scalar CLI literals. Preserve CLI ordering so a
        # user override appended by train_agent.main wins over the YAML default.
        for argument in command[3:]:
            if argument.startswith("data.trace."):
                key, literal = argument.split("=", 1)
                values[key.removeprefix("data.trace.")] = yaml.safe_load(literal)
        return PAIRING_CREDIT.TraceConfig.from_mapping(values), command

    @staticmethod
    def pairs(config):
        rows = [dict(index=index, uid="question", turn_type="tool_plan", turn_index=0,
                     rollout_id=f"rollout-{index}", value=.2, credit=credit,
                     valid=True, length=10, old_per_token=-1., action_key=f"action-{index}")
                for index, credit in enumerate((.4, -.1, -.2))]
        return PAIRING_CREDIT.select_pairs(rows, config)

    def test_package_default_preserves_v2_e_all_combinations(self):
        document = yaml.safe_load((PAIRING_ROOT / "agentflow/verl/config.yaml").read_text())
        self.assertIs(document["data"]["trace"]["pair_round_robin"], False)
        self.assertFalse(PAIRING_CREDIT.TraceConfig.from_mapping(document["data"]["trace"]).pair_round_robin)

    def test_both_training_yaml_files_export_the_switch(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                document = yaml.safe_load((PAIRING_ROOT / "train-roma" / filename).read_text())
                self.assertEqual(document["env"]["TRACE_PAIR_ROUND_ROBIN"],
                                 "${env:TRACE_PAIR_ROUND_ROBIN,false}")
                self.assertEqual(document["python_args"]["data.trace.pair_round_robin"],
                                 "${TRACE_PAIR_ROUND_ROBIN}")

    def test_default_launch_preserves_v2_e_all_combinations(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                config, command = self.launch_config(filename)
                self.assertFalse(config.pair_round_robin)
                self.assertIn("data.trace.pair_round_robin=false", command)
                self.assertEqual(self.pairs(config), [(0, 2), (0, 1)])

    def test_env_switch_changes_selection_without_changing_other_controls(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                enabled, enabled_command = self.launch_config(filename, {"TRACE_PAIR_ROUND_ROBIN": "true"})
                disabled, disabled_command = self.launch_config(filename, {"TRACE_PAIR_ROUND_ROBIN": "false"})
                self.assertTrue(enabled.pair_round_robin)
                self.assertFalse(disabled.pair_round_robin)
                self.assertIn("data.trace.pair_round_robin=true", enabled_command)
                self.assertIn("data.trace.pair_round_robin=false", disabled_command)
                self.assertEqual(self.pairs(enabled), [(0, 2)])
                self.assertEqual(self.pairs(disabled), [(0, 2), (0, 1)])
                self.assertEqual({key: value for key, value in asdict(enabled).items()
                                  if key != "pair_round_robin"},
                                 {key: value for key, value in asdict(disabled).items()
                                  if key != "pair_round_robin"})

    def test_explicit_cli_override_wins_over_opposite_env_value(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            for env_value, override, expected_pairs in (
                    ("true", "false", [(0, 2), (0, 1)]),
                    ("false", "true", [(0, 2)])):
                with self.subTest(filename=filename, override=override):
                    config, command = self.launch_config(
                        filename, {"TRACE_PAIR_ROUND_ROBIN": env_value},
                        ["data.trace.pair_round_robin=" + override])
                    self.assertEqual(command[-1], "data.trace.pair_round_robin=" + override)
                    self.assertIs(config.pair_round_robin, override == "true")
                    self.assertEqual(self.pairs(config), expected_pairs)


PREFIX_ROOT = Path(__file__).resolve().parents[2]
PREFIX_SPEC = importlib.util.spec_from_file_location(
    "trace_prefix_config_test_credit", PREFIX_ROOT / "agentflow/agentflow/trace/credit.py")
PREFIX_CREDIT = importlib.util.module_from_spec(PREFIX_SPEC)
sys.modules[PREFIX_SPEC.name] = PREFIX_CREDIT
PREFIX_SPEC.loader.exec_module(PREFIX_CREDIT)
TraceConfig = PREFIX_CREDIT.TraceConfig

try:
    import yaml
except ImportError:
    yaml = None


PREFIX_SETTINGS = {
    "shared_prefix_scoring": ("TRACE_SHARED_PREFIX_SCORING", True),
    "shared_prefix_max_tokens": ("TRACE_SHARED_PREFIX_MAX_TOKENS", 8192),
    "shared_prefix_max_mask_mib": ("TRACE_SHARED_PREFIX_MAX_MASK_MIB", 256),
    "shared_prefix_verify": ("TRACE_SHARED_PREFIX_VERIFY", True),
}


class SharedPrefixConfigTests(unittest.TestCase):
    def test_library_default_keeps_original_scorer_and_normalizes_round_trip(self):
        cfg = TraceConfig.from_mapping()
        self.assertFalse(cfg.shared_prefix_scoring)
        self.assertTrue(cfg.shared_prefix_verify)
        self.assertEqual(cfg.shared_prefix_max_tokens, 8192)
        self.assertEqual(cfg.shared_prefix_max_mask_mib, 256)
        self.assertEqual(TraceConfig.from_mapping(asdict(cfg)), cfg)
        self.assertIs(TraceConfig.from_mapping(cfg), cfg)

    def test_enabled_and_explicitly_disabled_controls_are_independent_of_credit(self):
        for enabled, verify in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(enabled=enabled, verify=verify):
                cfg = TraceConfig.from_mapping({
                    "shared_prefix_scoring": enabled,
                    "shared_prefix_verify": verify,
                    "shared_prefix_max_tokens": 4096,
                    "shared_prefix_max_mask_mib": 64,
                })
                self.assertIs(cfg.shared_prefix_scoring, enabled)
                self.assertIs(cfg.shared_prefix_verify, verify)
                self.assertEqual(cfg.shared_prefix_max_tokens, 4096)
                self.assertEqual(cfg.shared_prefix_max_mask_mib, 64)
                base = TraceConfig()
                for key, value in asdict(base).items():
                    if key not in PREFIX_SETTINGS:
                        self.assertEqual(getattr(cfg, key), value)

    def test_prefix_boolean_flags_reject_strings_numbers_and_null(self):
        for key in ("shared_prefix_scoring", "shared_prefix_verify"):
            for value in ("true", "false", "1", 0, 1, None, [], {}):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, re.escape("trace." + key) + " must be a boolean"
                ):
                    TraceConfig.from_mapping({key: value})

    def test_prefix_limits_require_positive_integers_even_when_disabled(self):
        for key in ("shared_prefix_max_tokens", "shared_prefix_max_mask_mib"):
            for value in (True, False, 0, -1, 2.5, "8192", None, float("inf")):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, re.escape("trace." + key) + " must be a positive integer"
                ):
                    TraceConfig.from_mapping({key: value, "shared_prefix_scoring": False})

    def test_legacy_config_warning_and_unknown_field_rejection_are_preserved(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            cfg = TraceConfig.from_mapping({"pair_score_mode": "igpo", "igpo_gamma": 1.,
                                            "shared_prefix_scoring": True})
        self.assertTrue(cfg.shared_prefix_scoring)
        self.assertEqual(len(recorded), 1)
        self.assertIn("does not enable IGPO", str(recorded[0].message))
        with self.assertRaisesRegex(ValueError, "Unknown TRACE configuration fields"):
            TraceConfig.from_mapping({"shared_prefix_scorng": True})


@unittest.skipIf(yaml is None, "PyYAML is unavailable")
class SharedPrefixYamlTests(unittest.TestCase):
    def test_package_defaults_match_the_validated_library_defaults(self):
        document = yaml.safe_load((PREFIX_ROOT / "agentflow/verl/config.yaml").read_text())
        cfg = TraceConfig.from_mapping(document["data"]["trace"])
        # Experiment pairing defaults are intentionally independent of the
        # reusable library defaults; shared-prefix settings still match.
        for field in PREFIX_SETTINGS:
            self.assertEqual(getattr(cfg, field), getattr(TraceConfig(), field))

    def test_training_entrypoints_enable_and_export_the_same_prefix_controls(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                document = yaml.safe_load((PREFIX_ROOT / "train-roma" / filename).read_text())
                for field, (variable, default) in PREFIX_SETTINGS.items():
                    default_literal = str(default).lower() if isinstance(default, bool) else str(default)
                    self.assertEqual(document["env"][variable], "${env:" + variable + "," + default_literal + "}")
                    self.assertEqual(document["python_args"]["data.trace." + field], "${" + variable + "}")

    def test_explicit_env_disable_reaches_the_original_scorer_switch(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                document = yaml.safe_load((PREFIX_ROOT / "train-roma" / filename).read_text())
                environment = {"TRACE_SHARED_PREFIX_SCORING": "false"}
                values = {}
                for field, (variable, _) in PREFIX_SETTINGS.items():
                    expression = document["env"][variable]
                    match = re.fullmatch(r"\$\{env:([A-Z_]+),([^}]+)\}", expression)
                    self.assertIsNotNone(match)
                    name, default = match.groups()
                    values[field] = yaml.safe_load(environment.get(name, default))
                cfg = TraceConfig.from_mapping(values)
                self.assertFalse(cfg.shared_prefix_scoring)
                self.assertTrue(cfg.shared_prefix_verify)
                self.assertEqual(cfg.shared_prefix_max_tokens, 8192)
                self.assertEqual(cfg.shared_prefix_max_mask_mib, 256)


if __name__ == "__main__":
    unittest.main()
