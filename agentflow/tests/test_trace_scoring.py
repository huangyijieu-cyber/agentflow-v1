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


if __name__ == "__main__":
    unittest.main()
