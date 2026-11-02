"""Exercise real trainer TRACE methods without distributed worker imports.

The reference stub scores the independently constructed, teacher-forced GT
requests.  It never reads rollout rewards or rollout-policy log probabilities.
"""

import ast
from collections import defaultdict
import contextlib
import importlib.util
import io
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CREDIT = load_source_module("trace_credit_trainer_test", ROOT / "agentflow/agentflow/trace/credit.py")
SCORING = CREDIT
METRICS = load_source_module("trace_metrics_trainer_test", ROOT / "agentflow/agentflow/trace/metrics.py")
EXECUTION = load_source_module("trace_execution_trainer_test", ROOT / "agentflow/agentflow/trace/execution.py")
ALFWORLD = load_source_module("trace_alfworld_trainer_test", ROOT / "agentflow/agentflow/trace/alfworld.py")


class Proto:
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = dict(meta_info or {})

    def __len__(self):
        return len(next(iter(self.batch.values())))

    @classmethod
    def from_dict(cls, tensors, non_tensors=None, meta_info=None):
        return cls(tensors, non_tensors, meta_info)

    @classmethod
    def from_single_dict(cls, values):
        return cls({k: v for k, v in values.items() if isinstance(v, torch.Tensor)},
                   {k: v for k, v in values.items() if not isinstance(v, torch.Tensor)})

    def select_rows(self, indices):
        return type(self)(
            {key: value[indices].clone() for key, value in self.batch.items()},
            {key: value[indices].copy() for key, value in self.non_tensor_batch.items()},
            self.meta_info,
        )

    def union(self, other):
        return type(self)({**self.batch, **other.batch},
                          {**self.non_tensor_batch, **other.non_tensor_batch},
                          {**self.meta_info, **other.meta_info})

    def select(self, batch_keys=None, non_tensor_batch_keys=None):
        return type(self)(
            {key: self.batch[key] for key in batch_keys},
            {key: self.non_tensor_batch[key] for key in non_tensor_batch_keys},
            dict(self.meta_info),
        )


def pad_dataproto_to_divisor(proto, divisor):
    padding = (-len(proto)) % divisor
    indices = list(range(len(proto))) + [i % len(proto) for i in range(padding)]
    return proto.select_rows(indices), padding


def unpad_dataproto(proto, pad_size):
    return proto.select_rows(list(range(len(proto) - pad_size)))


def load_trainer_methods():
    path = ROOT / "agentflow/verl/trainer.py"
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "AgentFlowTrainer")
    names = {"_anchors_match", "_actor_input_view", "_needs_action_ref_log_prob", "_annotate_trace_credits",
             "_trans_to_trace_batch", "_trans_to_dpo_batch"}
    methods = [node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(methods) != len(names):
        raise AssertionError("Required TRACE trainer methods were not found")
    scope = dict(
        torch=torch, np=np, random=random, defaultdict=defaultdict, DataProto=Proto, time=time,
        TraceConfig=CREDIT.TraceConfig, state_values=CREDIT.state_values,
        annotate_rollout_credits=CREDIT.annotate_rollout_credits,
        annotate_environment_credits=CREDIT.annotate_environment_credits,
        scale_outcome_advantages=CREDIT.scale_outcome_advantages,
        pair_group_key=CREDIT.pair_group_key, select_pairs=CREDIT.select_pairs,
        summarize_states=METRICS.summarize_states,
        summarize_selected_pairs=METRICS.summarize_selected_pairs,
        encode_state_requests=SCORING.encode_state_requests,
        build_score_tensors=SCORING.build_score_tensors, mean_gt_logps=SCORING.mean_gt_logps,
        pad_dataproto_to_divisor=pad_dataproto_to_divisor, unpad_dataproto=unpad_dataproto,
        FORECAST_RESPONSE=ALFWORLD.FORECAST_RESPONSE,
        FORECAST_SCORING_VERSION=ALFWORLD.FORECAST_SCORING_VERSION,
    )
    shell = ast.ClassDef(name="TrainerMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["TrainerMethods"]


TrainerMethods = load_trainer_methods()


def load_action_ref_stage():
    """Run the production stage using CPU workers, without constructing Ray."""
    path = ROOT / "agentflow/verl/trainer.py"
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "AgentFlowTrainer")
    step = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                and node.name == "_train_step")
    body = next(node.body for node in ast.walk(step) if isinstance(getattr(node, "body", None), list)
                and any(isinstance(statement, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "needs_action_ref_log_prob"
                                for target in statement.targets) for statement in node.body))
    start = next(index for index, statement in enumerate(body)
                 if isinstance(statement, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "needs_action_ref_log_prob"
                         for target in statement.targets))
    stage = ast.parse("def action_ref_stage(self, batch, metrics, timing_raw): pass").body[0]
    stage.body = body[start:start + 3] + [ast.Return(value=ast.Name(id="batch", ctx=ast.Load()))]

    @contextlib.contextmanager
    def timer(name, timing):
        timing[name] = 0.0
        yield

    scope = {"torch": torch, "_timer": timer}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stage], type_ignores=[])),
                 str(path), "exec"), scope)
    return scope[stage.name]


action_ref_stage = load_action_ref_stage()


class ConfigDict(dict):
    __getattr__ = dict.__getitem__


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 3 for character in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return self.encode(messages[0]["content"] + "\nAssistant:")

    def decode(self, ids):
        return "".join(chr(int(token) - 3) for token in ids if int(token) != self.pad_token_id)


class FrozenReference:
    world_size = 4

    def __init__(self, tokenizer, scores):
        self.tokenizer = tokenizer
        self.scores = scores
        self.calls = []

    def compute_ref_log_prob(self, probes):
        self.calls.append(probes.select_rows(list(range(len(probes)))))
        if probes.meta_info != {"trace_gt_scoring": True, "temperature": 1.0}:
            raise AssertionError("GT scoring must use its own temperature-one request")
        # Invalid values on padding deliberately catch averaging masked GT
        # padding or averaging over the decision response instead of GT tokens.
        values = torch.full_like(probes.batch["responses"], float("nan"), dtype=torch.float32)
        for index in range(len(probes)):
            context = self.tokenizer.decode(probes.batch["prompts"][index]).partition("\n\n")[0]
            valid = probes.batch["response_mask"][index].bool()
            score = self.scores[context]
            values[index, valid] = torch.as_tensor(score, dtype=torch.float32)
            if index % 2:
                values[index, ~valid] = 999.0
        return Proto.from_dict({"ref_log_prob": values})


def make_row(context, rollout, turn, *, turn_type="tool_plan", old=-1.0, reward=1.0,
             valid=True, dropped=False, uid="q", gt="AB", external_error=False,
             is_final=False, outcome_valid=False, outcome_success=None):
    return dict(context=context, rollout=rollout, turn=turn, turn_type=turn_type,
                old=old, reward=reward, valid=valid, dropped=dropped, uid=uid, gt=gt,
                external_error=external_error, is_final=is_final,
                outcome_valid=outcome_valid, outcome_success=outcome_success)


def make_batch(rows):
    size = len(rows)
    prompts = torch.arange(size * 3).reshape(size, 3) + 1000
    responses = torch.tensor([[100 + i, 200 + i, 0] for i in range(size)])
    response_mask = torch.tensor([[1, 1, 0]] * size)
    batch = dict(
        prompts=prompts, responses=responses,
        input_ids=torch.cat((prompts, responses), dim=-1),
        attention_mask=torch.cat((torch.ones_like(prompts), response_mask), dim=-1),
        position_ids=torch.arange(6).repeat(size, 1),
        response_mask=response_mask,
        old_log_probs=torch.tensor([[r["old"], r["old"], 0.] for r in rows]),
        ref_log_prob=torch.tensor([[-0.5, -0.5, 0.]] * size),
        token_level_scores=torch.tensor([[r["reward"], 0., 0.] for r in rows]),
        is_drop_mask=torch.tensor([r["dropped"] for r in rows], dtype=torch.bool),
    )
    non = {
        "uid": np.array([r["uid"] for r in rows]),
        "data_id_list": np.array([r["uid"] for r in rows]),
        "turn_index_list": np.array([r["turn"] for r in rows]),
        "trace_state_context": np.array([r["context"] for r in rows], dtype=object),
        "trace_gt": np.array([r["gt"] for r in rows], dtype=object),
        "trace_turn_type": np.array([r["turn_type"] for r in rows]),
        "trace_valid": np.array([r["valid"] for r in rows]),
        "trace_external_error": np.array([r["external_error"] for r in rows], dtype=bool),
        "trace_is_final": np.array([r["is_final"] for r in rows], dtype=bool),
        "trace_outcome_valid": np.array([r["outcome_valid"] for r in rows], dtype=bool),
        "trace_outcome_success": np.array([r["outcome_success"] for r in rows], dtype=object),
        "trace_outcome_reward": np.array([r["reward"] for r in rows], dtype=float),
        "turn_valid_list": np.array([r["valid"] for r in rows]),
        "turn_dup_list": np.zeros(size, dtype=bool),
        "anchor_list": np.array([r["turn_type"] + "\nsame anchor" for r in rows]),
    }
    for key in ("rollout_id_list", "traj_id_list", "traj_uid"):
        non[key] = np.array([r["rollout"] for r in rows])
    return Proto(batch, non, {"temperature": 0.7})


def make_trainer(scores, trace=None):
    trainer = TrainerMethods()
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
        data=ConfigDict(task="qa", algorithm="slic", max_prompt_length=1024,
                        trace={"enabled": True, "k": 1, "terminal_fill_enabled": False, **(trace or {})}),
        algorithm=ConfigDict(norm_adv_by_std_in_grpo=False, use_kl_in_reward=False),
        actor_rollout_ref=SimpleNamespace(actor=ConfigDict(use_kl_loss=False)),
    )
    trainer.algorithm = "slic"
    trainer.global_steps = 12
    trainer.use_reference_policy = True
    trainer.ref_in_actor = False
    trainer.tokenizer = Tokenizer()
    trainer.ref_policy_wg = FrozenReference(trainer.tokenizer, scores)
    return trainer


def standard_rows(chosen_reward=1.0, rejected_reward=1.0):
    # Positive action A1 and negative action B1 share their original turn,
    # while retaining each rollout's own decision prompt and state baseline.
    rows = [
        make_row("A0", "a", 0, turn_type="analysis", reward=chosen_reward),
        make_row("A1", "a", 1, old=-1, reward=chosen_reward),
        make_row("A2", "a", 2, turn_type="final_answer", reward=chosen_reward),
        make_row("B0", "b", 0, turn_type="analysis", reward=rejected_reward),
        make_row("B1", "b", 1, old=-3, reward=rejected_reward),
        make_row("B2", "b", 2, turn_type="final_answer", reward=rejected_reward),
    ]
    return rows, {"A0": -4., "A1": -4., "A2": -2., "B0": -4., "B1": -4., "B2": -8.}


def value_rollout(rollout, values, reward, *, uid="q", trusted=True, final=True):
    """Generate GT scores with prescribed V_t for end-to-end credit checks."""
    rows, scores = [], {}
    for turn, value in enumerate(values):
        terminal = final and turn == len(values) - 1
        kind = "analysis" if turn == 0 else ("final_answer" if terminal else "tool_plan")
        context = f"{uid}/{rollout}/{turn}"
        rows.append(make_row(context, rollout, turn, uid=uid, turn_type=kind,
                             reward=reward, is_final=terminal,
                             outcome_valid=trusted, outcome_success=reward > 0.5))
        scores[context] = 0.1 - 4.1 * np.exp(-value)
    return rows, scores


class ActorInputViewTests(unittest.TestCase):
    def test_driver_trace_text_is_omitted_without_changing_model_inputs(self):
        trainer = make_trainer({})
        tensors = {name: torch.arange(6).reshape(2, 3) for name in
                   ("input_ids", "responses", "old_log_probs", "advantages", "ref_log_prob", "rollout_is_weights")}
        non_tensors = {
            "uid": np.array(["a", "b"]),
            "multi_modal_inputs": np.array([{"image": "a"}, {"image": "b"}], dtype=object),
            "trace_state_context": np.array(["long state a", "long state b"]),
            "trace_next_state_context": np.array(["next a", "next b"]),
            "trace_credit": np.array([.1, -.1]),
        }
        batch = Proto(tensors, non_tensors, {"temperature": .7, "multi_turn": True, "max_response_length": 3})
        view = trainer._actor_input_view(batch)
        self.assertEqual(set(view.batch), set(batch.batch))
        for key in batch.batch:
            self.assertIs(view.batch[key], batch.batch[key])
        self.assertEqual(set(view.non_tensor_batch), {"uid", "multi_modal_inputs"})
        self.assertEqual(view.meta_info, batch.meta_info)
        self.assertIs(view.non_tensor_batch["multi_modal_inputs"], batch.non_tensor_batch["multi_modal_inputs"])
        self.assertEqual(set(batch.non_tensor_batch), set(non_tensors))
        self.assertEqual(len(view), len(batch))

    def test_pair_baselines_and_gt_temperature_metadata_are_preserved(self):
        trainer = make_trainer({})
        pair = Proto({"input_ids_a": torch.ones(1, 2), "input_ids_b": torch.zeros(1, 2)},
                     {"group_baseline": np.array([-.2]), "born_step": np.array([1]),
                      "trace_task_a": np.array(["alfworld"]), "trace_gt_b": np.array(["goal"])},
                     {"trace_gt_scoring": True, "temperature": 1., "dpo_type": "slic"})
        view = trainer._actor_input_view(pair)
        self.assertEqual(set(view.non_tensor_batch), {"group_baseline", "born_step"})
        self.assertIs(view.non_tensor_batch["group_baseline"], pair.non_tensor_batch["group_baseline"])
        self.assertEqual(view.meta_info, pair.meta_info)
        self.assertIn("trace_gt_b", pair.non_tensor_batch)

    def test_non_trace_batches_keep_the_original_object(self):
        batch = Proto({"input_ids": torch.ones(1, 2)}, {"uid": np.array(["q"])})
        self.assertIs(make_trainer({})._actor_input_view(batch), batch)

    def test_all_actor_rpc_call_sites_send_the_lightweight_view(self):
        path = ROOT / "agentflow/verl/trainer.py"
        tree = ast.parse(path.read_text())
        step = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "_train_step")
        methods = {"compute_log_prob", "compute_ref_log_prob", "update_actor", "update_actor_by_dpo"}
        calls = [node for node in ast.walk(step) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr in methods]
        self.assertEqual({call.func.attr for call in calls}, methods)
        batch = Proto({"input_ids": torch.ones(1, 2)},
                      {"uid": np.array(["q"]), "trace_state_context": np.array(["large text"])},
                      {"temperature": .7})
        worker = SimpleNamespace(**{method: lambda data: data for method in methods})
        trainer = make_trainer({})
        trainer.actor_rollout_wg = worker
        for call in calls:
            with self.subTest(method=call.func.attr):
                code = compile(ast.Expression(body=call), str(path), "eval")
                sent = eval(code, {"self": trainer, "batch": batch,
                                   "sample_batch": batch, "ref_worker": worker})
                self.assertEqual(set(sent.non_tensor_batch), {"uid"})
                self.assertIs(sent.batch["input_ids"], batch.batch["input_ids"])
                self.assertEqual(sent.meta_info, batch.meta_info)
                self.assertIn("trace_state_context", batch.non_tensor_batch)


class SharedPrefixRoutingTests(unittest.TestCase):
    def test_enabled_scoring_uses_one_shared_reference_call_and_preserves_credits(self):
        class PrefixReference(FrozenReference):
            def __init__(self, tokenizer, scores):
                super().__init__(tokenizer, scores)
                self.prefix_calls = []

            def compute_ref_prefix_log_prob(self, probes):
                self.prefix_calls.append(dict(probes.meta_info))
                native = Proto(probes.batch, probes.non_tensor_batch,
                               {"trace_gt_scoring": True, "temperature": 1.0})
                result = super().compute_ref_log_prob(native)
                result.meta_info["trace_prefix_metrics"] = {"trace/prefix_used": 1,
                                                            "trace/prefix_saved_tokens": 123}
                return result

        rows, scores = standard_rows()
        baseline, shared = make_trainer(scores), make_trainer(scores, {
            "shared_prefix_scoring": True, "shared_prefix_max_tokens": 4096,
            "shared_prefix_max_mask_mib": 64, "shared_prefix_verify": True})
        shared.ref_policy_wg = PrefixReference(shared.tokenizer, scores)
        original, optimized = make_batch(rows), make_batch(rows)
        baseline._annotate_trace_credits(original)
        metrics = shared._annotate_trace_credits(optimized)
        self.assertEqual(len(shared.ref_policy_wg.prefix_calls), 1)
        meta = shared.ref_policy_wg.prefix_calls[0]
        self.assertEqual(meta["trace_prefix_max_tokens"], 4096)
        self.assertEqual(meta["trace_prefix_max_mask_bytes"], 64 * 1024 * 1024)
        self.assertIs(meta["trace_prefix_verify"], True)
        self.assertEqual(meta["temperature"], 1.0)
        self.assertEqual(metrics["trace/prefix_requested"], 1)
        self.assertEqual(metrics["trace/prefix_used"], 1)
        self.assertEqual(metrics["trace/prefix_saved_tokens"], 123)
        for key in ("trace_gt_logp", "trace_value", "trace_local_credit", "trace_terminal_fill", "trace_credit"):
            np.testing.assert_array_equal(original.non_tensor_batch[key], optimized.non_tensor_batch[key])
        old_pairs = baseline._trans_to_trace_batch([original])
        new_pairs = shared._trans_to_trace_batch([optimized])
        self.assertEqual(len(old_pairs), len(new_pairs))
        for old, new in zip(old_pairs, new_pairs):
            for key in ("group_baseline", "turn_index_list_a", "turn_index_list_b"):
                np.testing.assert_array_equal(old.non_tensor_batch[key], new.non_tensor_batch[key])

    def test_enabled_scoring_rejects_stale_workers_instead_of_hiding_missing_support(self):
        rows, scores = standard_rows()
        trainer = make_trainer(scores, {"shared_prefix_scoring": True})
        with self.assertRaisesRegex(RuntimeError, "updated reference workers"):
            trainer._annotate_trace_credits(make_batch(rows))


class ActionReferenceTests(unittest.TestCase):
    def test_trace_without_kl_skips_action_ref_but_still_scores_gt(self):
        for task in ("qa", "alfworld"):
            for algorithm in ("slic", "grpo+slic", "gigpo+slic"):
                with self.subTest(task=task, algorithm=algorithm):
                    rows, scores = standard_rows()
                    trainer, history = make_trainer(scores), make_batch(rows)
                    trainer.config.data["task"] = task
                    trainer.algorithm = algorithm
                    history.batch.pop("ref_log_prob")
                    metrics, timing = {}, {}
                    with contextlib.redirect_stdout(io.StringIO()):
                        history = action_ref_stage(trainer, history, metrics, timing)
                    self.assertEqual(trainer.ref_policy_wg.calls, [])
                    self.assertNotIn("ref_log_prob", history.batch)
                    self.assertEqual(metrics["perf/action_ref_log_prob_skipped"], 1)
                    self.assertEqual(timing["ref"], 0.)
                    trainer._annotate_trace_credits(history)
                    self.assertEqual(len(trainer.ref_policy_wg.calls), 1)
                    pair = trainer._trans_to_trace_batch([history])[0]
                    self.assertNotIn("ref_log_prob_a", pair.batch)
                    self.assertEqual(pair.non_tensor_batch["turn_index_list_a"].tolist(), [1])
                    self.assertEqual(pair.non_tensor_batch["turn_index_list_b"].tolist(), [1])

    def test_either_kl_objective_keeps_the_real_action_reference_forward(self):
        class Reference:
            def __init__(self):
                self.calls = []

            def compute_ref_log_prob(self, batch):
                self.calls.append(batch)
                return Proto.from_dict({"ref_log_prob": torch.full_like(
                    batch.batch["responses"], -0.75, dtype=torch.float32)})

        for reward_kl, actor_kl in ((True, False), (False, True), (True, True)):
            with self.subTest(reward_kl=reward_kl, actor_kl=actor_kl):
                trainer, history = make_trainer({}), make_batch(standard_rows()[0])
                trainer.config.algorithm["use_kl_in_reward"] = reward_kl
                trainer.config.actor_rollout_ref.actor["use_kl_loss"] = actor_kl
                trainer.ref_policy_wg = Reference()
                history.batch.pop("ref_log_prob")
                metrics, timing = {}, {}
                with contextlib.redirect_stdout(io.StringIO()):
                    scored = action_ref_stage(trainer, history, metrics, timing)
                self.assertEqual(len(trainer.ref_policy_wg.calls), 1)
                torch.testing.assert_close(scored.batch["ref_log_prob"],
                                           torch.full_like(scored.batch["responses"], -0.75,
                                                           dtype=torch.float32))
                self.assertEqual(metrics["perf/action_ref_log_prob_skipped"], 0)

    def test_other_algorithms_and_disabled_trace_preserve_the_previous_requirement(self):
        trainer = make_trainer({})
        for algorithm in ("abpo", "grpo+abpo", "grpo+simpo", "slic+hspo", "unknown+slic", "grpo"):
            with self.subTest(algorithm=algorithm):
                trainer.algorithm = algorithm
                self.assertTrue(trainer._needs_action_ref_log_prob())
        trainer.algorithm = "grpo+slic"
        trainer.config.data["trace"]["enabled"] = False
        self.assertTrue(trainer._needs_action_ref_log_prob())
        trainer.use_reference_policy = False
        self.assertFalse(trainer._needs_action_ref_log_prob())

    def test_missing_kl_configuration_does_not_silently_skip_action_scores(self):
        trainer = make_trainer({})
        del trainer.config.algorithm["use_kl_in_reward"]
        self.assertTrue(trainer._needs_action_ref_log_prob())
        trainer.config.algorithm["use_kl_in_reward"] = False
        del trainer.config.actor_rollout_ref.actor["use_kl_loss"]
        self.assertTrue(trainer._needs_action_ref_log_prob())
        del trainer.config.actor_rollout_ref
        self.assertTrue(trainer._needs_action_ref_log_prob())

    def test_old_action_ref_fields_do_not_change_trace_pairs_or_baselines(self):
        rows, scores = standard_rows()
        trainer, previous = make_trainer(scores), make_batch(rows)
        trainer._annotate_trace_credits(previous)
        current = previous.select_rows(list(range(len(previous))))
        current.batch.pop("ref_log_prob")
        old_pair = trainer._trans_to_trace_batch([previous])[0]
        new_pair = trainer._trans_to_trace_batch([current])[0]
        self.assertEqual(set(old_pair.batch), set(new_pair.batch))
        for name in old_pair.batch:
            torch.testing.assert_close(old_pair.batch[name], new_pair.batch[name], equal_nan=True)
        for name in ("group_baseline", "group_size", "turn_index_list_a", "turn_index_list_b"):
            np.testing.assert_array_equal(old_pair.non_tensor_batch[name], new_pair.non_tensor_batch[name])

    def test_trace_reports_separate_cpu_and_model_stages_including_no_requests(self):
        rows, scores = standard_rows()
        for gt in ("AB", ""):
            with self.subTest(gt=gt):
                for row in rows:
                    row["gt"] = gt
                trainer, history = make_trainer(scores), make_batch(rows)
                metrics = trainer._annotate_trace_credits(history)
                self.assertEqual(metrics["trace/config/gap_epsilon"], trainer.config.data.trace.get("gap_epsilon", .02))
                for phase in ("validation", "encoding", "scoring_prepare", "scoring",
                              "scoring_postprocess", "credit"):
                    self.assertGreaterEqual(metrics["trace/" + phase + "_seconds"], 0.)
                if not gt:
                    self.assertEqual(metrics["trace/scoring_seconds"], 0.)
                    self.assertEqual(metrics["trace/scoring_prepare_seconds"], 0.)
                    self.assertEqual(metrics["trace/scoring_postprocess_seconds"], 0.)


class TraceTrainerIntegrationTests(unittest.TestCase):
    def test_qa_keeps_its_original_answer_target_despite_alfworld_forecast_metadata(self):
        rows, scores = standard_rows()
        history, trainer = make_batch(rows), make_trainer(scores)
        history.non_tensor_batch["trace_task"] = np.full(len(rows), "qa", dtype=object)
        history.non_tensor_batch["trace_goal_metadata"] = np.array(
            [{"scoring_version": "old-or-unrelated", "scoring_response": ALFWORLD.FORECAST_RESPONSE}
             for _ in rows], dtype=object)
        original_gt = history.non_tensor_batch["trace_gt"].copy()
        trainer._annotate_trace_credits(history)
        for call in trainer.ref_policy_wg.calls:
            for index in range(len(call)):
                response = trainer.tokenizer.decode(
                    call.batch["responses"][index][call.batch["response_mask"][index].bool()])
                self.assertEqual(response, "AB")
        np.testing.assert_array_equal(history.non_tensor_batch["trace_gt"], original_gt)

    def test_null_pair_limit_allows_cross_turn_and_preserves_native_inputs(self):
        for limit in (None, "None", "null"):
            with self.subTest(limit=limit):
                rows, scores = standard_rows()
                history, trainer = make_batch(rows), make_trainer(scores)
                trainer._annotate_trace_credits(history)
                history.non_tensor_batch["trace_original_turn"] = history.non_tensor_batch["turn_index_list"].copy()
                history.non_tensor_batch["trace_original_turn"][4] = 7
                self.assertIsNone(trainer._trans_to_dpo_batch([history]))
                tensors = {key: value.clone() for key, value in history.batch.items()}
                credits = history.non_tensor_batch["trace_credit"].copy()
                native_turns = history.non_tensor_batch["turn_index_list"].copy()
                trainer.config.data.trace["k_pairs_per_group"] = limit
                pair = trainer._trans_to_dpo_batch([history])[0]
                self.assertEqual(pair.non_tensor_batch["turn_index_list_a"].tolist(), [1])
                self.assertEqual(pair.non_tensor_batch["turn_index_list_b"].tolist(), [7])
                self.assertEqual(pair.non_tensor_batch["pair_turn"].tolist(), [1])
                self.assertEqual(trainer._trace_pair_metrics["trace/pair/cross_turn_rate"], 1.)
                np.testing.assert_array_equal(history.non_tensor_batch["trace_credit"], credits)
                np.testing.assert_array_equal(history.non_tensor_batch["turn_index_list"], native_turns)
                for key, value in tensors.items():
                    torch.testing.assert_close(history.batch[key], value, equal_nan=True)

    def test_null_pair_limit_removes_driver_quota_across_history_batches(self):
        rows, scores = [], {}
        for rollout in ("a", "b", "c", "d"):
            for turn in range(3):
                context = rollout + str(turn)
                rows.append(make_row(context, rollout, turn,
                                     turn_type="tool_plan" if turn < 2 else "final_answer"))
                scores[context] = -4.
        history, trainer = make_batch(rows), make_trainer(scores)
        trainer._annotate_trace_credits(history)
        history.non_tensor_batch["trace_credit"][:] = [
            .2, .2, np.nan, .3, .3, np.nan, -.2, -.2, np.nan, -.3, -.4, np.nan]
        trainer.config.data.trace["k_pairs_per_group"] = None
        pairs = trainer._trans_to_trace_batch([history, history])
        self.assertEqual(len(pairs), 8)  # Four Chosen attempts in each history.
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/selected_unique"], 8)
        self.assertGreater(trainer._trace_pair_metrics["trace/pair/cross_turn_rate"], 0.)
        trainer.config.data.trace["max_pairs_per_step"] = 3
        self.assertEqual(len(trainer._trans_to_trace_batch([history, history])), 3)

    def test_null_pair_limit_uses_the_same_cross_turn_group_for_baseline(self):
        rows, scores = standard_rows()
        rows.extend([make_row("C0", "c", 0, old=-100.),
                     make_row("C1", "c", 1, turn_type="final_answer")])
        scores.update(C0=-4., C1=-4.)
        history, trainer = make_batch(rows), make_trainer(scores)
        trainer._annotate_trace_credits(history)
        same_turn_pair = trainer._trans_to_dpo_batch([history])[0]
        self.assertEqual(same_turn_pair.non_tensor_batch["group_baseline"].tolist(), [-2.])
        trainer.config.data.trace["k_pairs_per_group"] = None
        cross_turn_pair = trainer._trans_to_dpo_batch([history])[0]
        self.assertEqual(cross_turn_pair.non_tensor_batch["group_size"].tolist(), [3])
        self.assertAlmostEqual(cross_turn_pair.non_tensor_batch["group_baseline"][0], -104. / 3, places=5)

    def test_trace_pairing_uses_original_turn_ids_and_preserves_native_batch(self):
        rows, scores = standard_rows()
        history, trainer = make_batch(rows), make_trainer(scores)
        history.non_tensor_batch["trace_original_turn"] = history.non_tensor_batch["turn_index_list"].copy()
        history.non_tensor_batch["turn_index_list"][4] = 0  # Compacted worker numbering differs.
        native_turns = history.non_tensor_batch["turn_index_list"].copy()
        trainer._annotate_trace_credits(history)
        pair = trainer._trans_to_dpo_batch([history])[0]
        self.assertEqual(pair.non_tensor_batch["turn_index_list_a"].tolist(), [1])
        self.assertEqual(pair.non_tensor_batch["turn_index_list_b"].tolist(), [1])
        self.assertEqual(pair.non_tensor_batch["pair_turn"].tolist(), [1])
        np.testing.assert_array_equal(history.non_tensor_batch["turn_index_list"], native_turns)
        history.non_tensor_batch["trace_original_turn"][4] = 7
        self.assertIsNone(trainer._trans_to_dpo_batch([history]))

    def test_trace_baseline_excludes_valid_actions_from_other_turns(self):
        rows, scores = standard_rows()
        rows.extend([make_row("C0", "c", 0, old=-100.),
                     make_row("C1", "c", 1, turn_type="final_answer")])
        scores.update(C0=-4., C1=-4.)
        history, trainer = make_batch(rows), make_trainer(scores)
        trainer._annotate_trace_credits(history)
        pair = trainer._trans_to_dpo_batch([history])[0]
        self.assertEqual(pair.non_tensor_batch["group_baseline"].tolist(), [-2.])
        self.assertEqual(pair.non_tensor_batch["group_size"].tolist(), [2])

    def test_trace_turn_quotas_are_independent_and_shared_across_history_batches(self):
        rows, scores = [], {}
        for rollout, positive in (("a", True), ("b", True), ("c", False), ("d", False)):
            for turn in range(3):
                context = rollout + str(turn)
                rows.append(make_row(context, rollout, turn,
                                     turn_type="tool_plan" if turn < 2 else "final_answer"))
                scores[context] = -4.
        history, trainer = make_batch(rows), make_trainer(scores)
        trainer._annotate_trace_credits(history)
        non = history.non_tensor_batch
        non["trace_credit"][:] = [.2, .2, np.nan, .3, .3, np.nan,
                                  -.2, -.2, np.nan, -.3, -.3, np.nan]
        pairs = trainer._trans_to_trace_batch([history])
        counts = defaultdict(int)
        for pair in pairs:
            left, right = pair.non_tensor_batch["turn_index_list_a"][0], pair.non_tensor_batch["turn_index_list_b"][0]
            self.assertEqual(left, right)
            counts[int(left)] += 1
        self.assertEqual(dict(counts), {0: 2, 1: 2})
        repeated = trainer._trans_to_trace_batch([history, history])
        self.assertEqual(len(repeated), 4)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/selected_unique"], 4)

    def test_metric_coverage_funnel_and_outcome_disagreement_are_actual(self):
        rows, scores = standard_rows(chosen_reward=0., rejected_reward=1.)
        trainer, history = make_trainer(scores), make_batch(rows)
        state_metrics = trainer._annotate_trace_credits(history)
        self.assertEqual(state_metrics["trace/score_coverage"], 1.)
        self.assertEqual(state_metrics["trace/credit_expected_turns"], 2)
        self.assertEqual(state_metrics["trace/credit_coverage"], 1.)
        self.assertGreaterEqual(state_metrics["trace/scoring_seconds"], 0.)
        self.assertEqual(state_metrics["trace/missing_gt"], 0)
        trainer._trans_to_dpo_batch([history])
        metrics = trainer._trace_pair_metrics
        self.assertEqual(metrics["trace/pair/candidates_sign"], 1)
        self.assertEqual(metrics["trace/pair/tool_value_match_rate"], 1.)
        self.assertEqual(metrics["trace/pair/selected_unique"], 1)
        self.assertEqual(metrics["trace/pair/cross_turn_rate"], 0.)
        self.assertEqual(metrics["trace/pair/question_coverage"], 1.)
        self.assertEqual(metrics["trace/pair/trace/outcome_pairs_count"], 0)
        self.assertNotIn("trace/pair/chosen_failed_rollout_rate", metrics)

    def test_analysis_and_final_use_verified_outcomes_without_gt_scores(self):
        rows = []
        for rollout, reward, success in (("a", 1., True), ("b", 0.1, False)):
            offset = 0 if rollout == "a" else 5
            rows.extend([
                make_row(rollout + "0", rollout, offset, turn_type="analysis", gt="",
                         reward=reward, outcome_valid=True, outcome_success=success),
                make_row(rollout + "1", rollout, offset + 1, turn_type="final_answer", gt="",
                         reward=reward, is_final=True, outcome_valid=True, outcome_success=success),
            ])
        trainer, history = make_trainer({}), make_batch(rows)
        metrics = trainer._annotate_trace_credits(history)
        self.assertEqual(metrics["trace/scoring_requests"], 0)
        self.assertTrue(np.isnan(history.non_tensor_batch["trace_credit"]).all())
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 2)
        self.assertEqual({p.non_tensor_batch["trace_turn_type_a"][0] for p in pairs},
                         {"analysis", "final_answer"})
        for pair in pairs:
            self.assertEqual(pair.non_tensor_batch["rollout_id_list_a"].tolist(), ["a"])
            self.assertEqual(pair.non_tensor_batch["rollout_id_list_b"].tolist(), ["b"])
            self.assertEqual(pair.non_tensor_batch["trace_label_source_a"].tolist(), ["outcome"])
            self.assertTrue(np.isfinite(pair.non_tensor_batch["group_baseline"]).all())
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/outcome_selected_unique"], 2)
        self.assertNotIn("trace/pair/credit_gap_p50", trainer._trace_pair_metrics)

    def test_pruned_metrics_keep_pair_selection_and_hide_internal_counters(self):
        rows, scores = standard_rows()
        trainer, history = make_trainer(scores), make_batch(rows)
        state_metrics = trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 1)
        metrics = trainer._trace_pair_metrics
        self.assertEqual(metrics["trace/pair/question_coverage"], 1.)
        self.assertEqual(metrics["trace/pair/tool_value_match_rate"], 1.)
        self.assertEqual(metrics["trace/pair/both_sign_group_rate"], 1.)
        for key in ("groups_eligible", "groups_both_signs", "trace_candidates_sign",
                    "trace_after_value_match", "after_value_match", "after_credit_gap",
                    "after_length_filter", "after_distinct_action", "value_match_rate",
                    "selection_rate", "questions_covered", "rows_unique"):
            self.assertNotIn("trace/pair/" + key, metrics)
        self.assertNotIn("trace/unique_pairs", metrics)
        for key in ("scoring_prompt_tokens", "scoring_gt_tokens", "scoring_dedup_rate",
                    "deduplicated_states", "effective_credit_mean", "outcome_invalid_turns"):
            self.assertNotIn("trace/" + key, state_metrics)

    def test_zero_local_range_does_not_create_tool_credit_from_outcomes(self):
        rows, scores = [], {}
        for rollout, reward, success in (("a", 1., True), ("b", 0., False)):
            for turn, turn_type in enumerate(("analysis", "tool_plan", "tool_plan", "tool_plan", "final_answer")):
                context = rollout + str(turn)
                rows.append(make_row(context, rollout, turn, turn_type=turn_type,
                                     reward=reward, is_final=turn == 4,
                                     outcome_valid=True, outcome_success=success))
                scores[context] = -4.
        trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": True})
        history = make_batch(rows)
        trainer._annotate_trace_credits(history)
        credit = history.non_tensor_batch["trace_credit"]
        self.assertTrue(np.isnan(credit[[0, 4, 5, 9]]).all())
        np.testing.assert_allclose(credit[[1, 2, 3, 6, 7, 8]], 0.)
        np.testing.assert_allclose(history.non_tensor_batch["trace_outcome_advantage"][:5], 0.5)
        np.testing.assert_allclose(history.non_tensor_batch["trace_local_credit"][[1, 2, 3]], 0.)
        self.assertEqual(history.non_tensor_batch["trace_terminal_reached"].tolist(),
                         [False, False, True, True, False] * 2)
        self.assertEqual(len(trainer._trans_to_dpo_batch([history])), 2)

    def test_k3_dynamic_fill_keeps_final_discount_and_deduplicated_outcomes(self):
        rows, scores = [], {}
        for rollout, values, reward in (("a", [0, 0, .1, .2, .3], 1.),
                                        ("b", [0, 0, -.05, -.1], 0.)):
            new_rows, new_scores = value_rollout(rollout, values, reward)
            rows.extend(new_rows)
            scores.update(new_scores)
        history = make_batch(rows)
        trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": True})
        trainer._annotate_trace_credits(history)
        non = history.non_tensor_batch
        np.testing.assert_allclose(non["trace_outcome_advantage"], [.5] * 5 + [-.5] * 4)
        np.testing.assert_allclose(non["trace_scaled_outcome_advantage"], [.1] * 5 + [-.05] * 4,
                                   atol=1e-7)
        np.testing.assert_allclose(non["trace_terminal_fill"][[1, 2, 3, 6, 7]],
                                   [0, .064, .08, -.032, -.04], atol=1e-7)
        self.assertTrue(np.isnan(non["trace_credit"][[0, 4, 5, 8]]).all())

    def test_ranges_include_nonterminal_untrusted_tools_but_do_not_cross_uid(self):
        rows, scores = [], {}
        fixtures = [("q", "a", [0, 0, .1], 1., True, True),
                    ("q", "b", [0, 0, -.08], 0., True, True),
                    ("q", "prefix", [0, 0, .8], 100., False, False),
                    ("other", "a", [0, 0, 2.], 1., True, True),
                    ("other", "b", [0, 0, -1.], 0., True, True)]
        for uid, rollout, values, reward, trusted, final in fixtures:
            new_rows, new_scores = value_rollout(rollout, values, reward, uid=uid,
                                                trusted=trusted, final=final)
            rows.extend(new_rows)
            scores.update(new_scores)
        trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": True})
        history = make_batch(rows)
        trainer._annotate_trace_credits(history)
        non = history.non_tensor_batch
        np.testing.assert_allclose(non["trace_scaled_outcome_advantage"][[1, 4, 10, 13]],
                                   [.8, -.08, 2., -1.], atol=1e-6)
        np.testing.assert_allclose(non["trace_terminal_fill"][[1, 4, 7]], [.64, -.064, 0], atol=1e-6)
        np.testing.assert_allclose(non["trace_outcome_advantage"][[1, 4]], [.5, -.5])

    def test_dynamic_fill_matches_asymmetric_ranges_and_ignores_fixed_scale(self):
        rows, scores = [], {}
        for rollout, credit, reward in (("a", .2, 1.), ("b", .1, 1.),
                                        ("c", .05, 1.), ("d", -.08, 0.)):
            new_rows, new_scores = value_rollout(rollout, [0, 0, credit], reward)
            rows.extend(new_rows)
            scores.update(new_scores)
        for scale in (0., 2., 100.):
            for norm_by_std in (False, True):
                with self.subTest(scale=scale, norm_by_std=norm_by_std):
                    trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": True,
                                                    "terminal_fill_scale": scale})
                    trainer.config.algorithm["norm_adv_by_std_in_grpo"] = norm_by_std
                    history = make_batch(rows)
                    original_rewards = history.batch["token_level_scores"].clone()
                    trainer._annotate_trace_credits(history)
                    non = history.non_tensor_batch
                    np.testing.assert_allclose(non["trace_scaled_outcome_advantage"][[1, 4, 7, 10]],
                                               [.2, .2, .2, -.08], atol=1e-7)
                    np.testing.assert_allclose(non["trace_terminal_fill"][[1, 4, 7, 10]],
                                               [.16, .16, .16, -.064], atol=1e-7)
                    torch.testing.assert_close(history.batch["token_level_scores"], original_rewards)

    def test_invalid_tool_extrema_do_not_set_the_question_range(self):
        for excluded in ("invalid", "duplicate", "drop", "empty", "external", "missing_gt"):
            with self.subTest(excluded=excluded):
                rows, scores = [], {}
                for rollout, credit, reward in (("a", .2, 1.), ("b", -.08, 0.), ("bad", 1.5, 1.)):
                    new_rows, new_scores = value_rollout(rollout, [0, 0, credit], reward)
                    rows.extend(new_rows)
                    scores.update(new_scores)
                history = make_batch(rows)
                non = history.non_tensor_batch
                if excluded == "invalid":
                    non["trace_valid"][7] = False
                elif excluded == "duplicate":
                    non["trace_dup"] = np.zeros(len(rows), dtype=bool)
                    non["trace_dup"][7] = True
                elif excluded == "drop":
                    history.batch["is_drop_mask"][7] = True
                elif excluded == "empty":
                    history.batch["response_mask"][7] = 0
                elif excluded == "external":
                    non["trace_external_error"][7] = True
                else:
                    non["trace_gt"][7] = ""
                trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": True})
                trainer._annotate_trace_credits(history)
                np.testing.assert_allclose(non["trace_scaled_outcome_advantage"][[1, 4]],
                                           [.2, -.08], atol=1e-7)

    def test_missing_tool_sign_or_disabled_fill_adds_no_terminal_credit(self):
        for enabled in (False, True):
            rows, scores = [], {}
            for rollout, credit, reward in (("a", .2, 1.), ("b", .1, 0.)):
                new_rows, new_scores = value_rollout(rollout, [0, 0, credit], reward)
                rows.extend(new_rows)
                scores.update(new_scores)
            trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": enabled})
            history = make_batch(rows)
            trainer._annotate_trace_credits(history)
            non = history.non_tensor_batch
            np.testing.assert_allclose(non["trace_terminal_fill"][[1, 4]],
                                       [.16 if enabled else 0, 0], atol=1e-7)

    def test_non_qa_tasks_keep_fixed_scale_terminal_fill(self):
        rows, scores = [], {}
        for rollout, credit, reward in (("a", .2, 1.), ("b", -.08, 0.)):
            new_rows, new_scores = value_rollout(rollout, [0, 0, credit], reward)
            rows.extend(new_rows)
            scores.update(new_scores)
        for task, metadata_task in (("webshop", None), ("rule", None), ("qa", "webshop")):
            for scale in (0., 2., 3.):
                with self.subTest(task=task, metadata_task=metadata_task, scale=scale):
                    trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": True,
                                                    "terminal_fill_scale": scale})
                    trainer.config.data["task"] = task
                    history = make_batch(rows)
                    history.non_tensor_batch["trace_task"] = np.full(len(rows), metadata_task, dtype=object)
                    trainer._annotate_trace_credits(history)
                    non = history.non_tensor_batch
                    np.testing.assert_allclose(non["trace_terminal_fill"][[1, 4]],
                                               [.4 * scale, -.4 * scale], atol=1e-7)
                    self.assertTrue(np.isnan(non["trace_scaled_outcome_advantage"]).all())

    def test_external_failure_blocks_tool_windows_and_untrusted_outcome_pairs(self):
        rows, scores = [], {}
        for rollout, reward, success, trustworthy in (("a", 1., True, True), ("b", 0., False, False)):
            for turn, turn_type in enumerate(("analysis", "tool_plan", "tool_plan", "tool_plan", "final_answer")):
                context = rollout + str(turn)
                rows.append(make_row(context, rollout, turn, turn_type=turn_type,
                                     reward=reward, is_final=turn == 4,
                                     external_error=rollout == "b" and turn == 3,
                                     outcome_valid=trustworthy, outcome_success=success))
                scores[context] = -4.
        trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": True})
        history = make_batch(rows)
        metrics = trainer._annotate_trace_credits(history)
        self.assertTrue(np.isnan(history.non_tensor_batch["trace_credit"][[6, 7, 8]]).all())
        self.assertTrue(history.non_tensor_batch["trace_credit_window_blocked"][[6, 7, 8]].all())
        self.assertEqual(metrics["trace/external_error_turns"], 1)
        self.assertIsNone(trainer._trans_to_dpo_batch([history]))

    def test_actual_evaluated_answer_can_precede_later_generated_outputs(self):
        rows = []
        for rollout, reward in (("a", 1.), ("b", 0.)):
            rows.extend([
                make_row(rollout + "0", rollout, 0, turn_type="base", gt="",
                         reward=reward, is_final=True, outcome_valid=True, outcome_success=bool(reward)),
                make_row(rollout + "1", rollout, 1, turn_type="analysis", gt="",
                         reward=reward, outcome_valid=True, outcome_success=bool(reward)),
                make_row(rollout + "2", rollout, 2, gt="", reward=reward,
                         outcome_valid=True, outcome_success=bool(reward)),
            ])
        trainer, history = make_trainer({}), make_batch(rows)
        metrics = trainer._annotate_trace_credits(history)
        self.assertEqual(metrics["trace/post_final_turns"], 4)
        self.assertEqual(history.non_tensor_batch["trace_outcome_valid"].tolist(),
                         [True, False, False] * 2)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].non_tensor_batch["trace_turn_type_a"].tolist(), ["base"])

    def test_outcome_routes_never_infer_success_from_untrusted_reward(self):
        rows = [make_row("A", "a", 0, turn_type="final_answer", is_final=True,
                         reward=1., outcome_valid=False),
                make_row("B", "b", 0, turn_type="final_answer", is_final=True,
                         reward=0., outcome_valid=True, outcome_success=False)]
        trainer, history = make_trainer(dict(A=-4., B=-4.)), make_batch(rows)
        trainer._annotate_trace_credits(history)
        self.assertIsNone(trainer._trans_to_dpo_batch([history]))

    def test_legacy_duplicate_flag_does_not_drop_recovered_trace_action(self):
        rows, scores = standard_rows()
        trainer, history = make_trainer(scores), make_batch(rows)
        # Legacy history counts the failed attempt as seen, while TRACE has
        # already certified this successful retry as valid and nonduplicate.
        history.non_tensor_batch["turn_dup_list"][1] = True
        history.non_tensor_batch["trace_dup"] = np.zeros(len(rows), dtype=bool)
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].non_tensor_batch["turn_index_list_a"].tolist(), [1])

    def test_real_terminal_metadata_flows_into_both_pair_routes(self):
        rows, scores = [], {}
        for rollout, success in (("a", True), ("b", False)):
            logs = [{"trace_turn_type": kind, "response": rollout + str(turn),
                     "finish_reason": "stop"}
                    for turn, kind in enumerate(("analysis", "tool_plan", "final_answer"))]
            EXECUTION.annotate_trace_outcome(
                logs, {"final_output": rollout + "2"}, float(success),
                {"success": success, "valid": True})
            for turn, log in enumerate(logs):
                context = rollout + str(turn)
                rows.append(make_row(context, rollout, turn, turn_type=log["trace_turn_type"],
                                     reward=log["trace_outcome_reward"],
                                     is_final=log["trace_is_final"],
                                     outcome_valid=log["trace_outcome_valid"],
                                     outcome_success=log["trace_outcome_success"],
                                     external_error=log["trace_external_error"]))
                scores[context] = (-2. if success else -8.) if turn == 2 else -4.
        trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": True})
        history = make_batch(rows)
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 3)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/trace_selected_unique"], 1)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/outcome_selected_unique"], 2)
        non = history.non_tensor_batch
        np.testing.assert_allclose(non["trace_credit"][[1, 4]], 1.8 * non["trace_local_credit"][[1, 4]])

    def test_distributed_padding_does_not_inflate_pair_statistics(self):
        rows, scores = standard_rows()
        trainer, history = make_trainer(scores), make_batch(rows)
        trainer.config.trainer.n_gpus_per_node = 8
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 8)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/selected_unique"], 1)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/padding_duplicates"], 7)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/trace_selected_unique"], 1)

    def test_global_cap_across_history_batches_preserves_funnel_counts(self):
        rows, scores = standard_rows()
        trainer = make_trainer(scores, trace={"max_pairs_per_step": 1})
        first, second = make_batch(rows), make_batch(rows)
        trainer._annotate_trace_credits(first)
        trainer._annotate_trace_credits(second)
        self.assertEqual(len(trainer._trans_to_dpo_batch([first, second])), 1)
        metrics = trainer._trace_pair_metrics
        self.assertEqual(metrics["trace/pair/candidates_sign"], 2)
        self.assertEqual(metrics["trace/pair/rejected_global_cap"], 1)
        self.assertEqual(metrics["trace/pair/selected_unique"], 1)

    def test_same_turn_pair_keeps_each_prompt_and_ignores_final_reward(self):
        for chosen_reward, rejected_reward in ((1., 1.), (0., 1.), (1., 0.)):
            with self.subTest(chosen_reward=chosen_reward, rejected_reward=rejected_reward):
                rows, scores = standard_rows(chosen_reward, rejected_reward)
                history = make_batch(rows)
                trainer = make_trainer(scores)
                trainer._annotate_trace_credits(history)
                pairs = trainer._trans_to_dpo_batch([history])
                self.assertEqual(len(pairs), 1)
                pair = pairs[0]
                self.assertEqual(pair.non_tensor_batch["rollout_id_list_a"].tolist(), ["a"])
                self.assertEqual(pair.non_tensor_batch["rollout_id_list_b"].tolist(), ["b"])
                self.assertEqual(pair.non_tensor_batch["turn_index_list_a"].tolist(), [1])
                self.assertEqual(pair.non_tensor_batch["turn_index_list_b"].tolist(), [1])
                self.assertTrue(torch.equal(pair.batch["prompts_a"], history.batch["prompts"][1:2]))
                self.assertTrue(torch.equal(pair.batch["prompts_b"], history.batch["prompts"][4:5]))
                self.assertFalse(torch.equal(pair.batch["prompts_a"], pair.batch["prompts_b"]))
                self.assertGreater(pair.non_tensor_batch["trace_credit_a"][0], 0)
                self.assertLess(pair.non_tensor_batch["trace_credit_b"][0], 0)
                self.assertEqual(float(pair.non_tensor_batch["group_baseline"][0]), -2.)
                self.assertEqual(pair.meta_info["temperature"], 0.7)
                self.assertEqual(pair.batch["token_level_scores_a"][0, 0], chosen_reward)
                self.assertEqual(pair.batch["token_level_scores_b"][0, 0], rejected_reward)

    def test_dropped_action_does_not_pollute_baseline(self):
        rows, scores = standard_rows()
        rows.extend([make_row("C0", "c", 0, old=-100, dropped=True),
                     make_row("C1", "c", 1, turn_type="final_answer")])
        scores.update(C0=-4., C1=-4.)
        history = make_batch(rows)
        trainer = make_trainer(scores)
        trainer._annotate_trace_credits(history)
        pair = trainer._trans_to_dpo_batch([history])[0]
        self.assertEqual(pair.non_tensor_batch["group_baseline"].tolist(), [-2.])
        self.assertEqual(pair.non_tensor_batch["group_size"].tolist(), [2])
        # Action filtering must not delete its state observation before credit.
        self.assertEqual(history.non_tensor_batch["trace_credit"][6], 0.)

    def test_invalid_next_action_still_supplies_current_actions_next_state(self):
        rows = [make_row("A0", "a", 0), make_row("A1", "a", 1, valid=False),
                make_row("A2", "a", 2, turn_type="final_answer"),
                make_row("B0", "b", 0), make_row("B1", "b", 1, turn_type="final_answer")]
        trainer = make_trainer(dict(A0=-4., A1=-2., A2=-8., B0=-4., B1=-8.))
        history = make_batch(rows)
        trainer._annotate_trace_credits(history)
        self.assertGreater(history.non_tensor_batch["trace_credit"][0], 0)
        self.assertLess(history.non_tensor_batch["trace_credit"][1], 0)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].non_tensor_batch["turn_index_list_a"].tolist(), [0])
        self.assertEqual(pairs[0].non_tensor_batch["rollout_id_list_b"].tolist(), ["b"])

    def test_no_negative_credit_skips_even_when_rewards_differ(self):
        rows, scores = standard_rows(chosen_reward=1., rejected_reward=0.)
        scores["B2"] = -2.
        trainer = make_trainer(scores)
        history = make_batch(rows)
        trainer._annotate_trace_credits(history)
        self.assertIsNone(trainer._trans_to_dpo_batch([history]))
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/groups_no_negative"], 1)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/selected_unique"], 0)
        self.assertNotIn("trace/pair/credit_gap_p50", trainer._trace_pair_metrics)

    def test_missing_initial_state_and_original_turn_gaps_do_not_create_credit(self):
        rows = [make_row("M1", "missing_initial", 1), make_row("M2", "missing_initial", 2),
                make_row("G0", "gap", 0), make_row("G2", "gap", 2), make_row("G3", "gap", 3),
                make_row("O0", "missing_observation", 0),
                make_row(None, "missing_observation", 1), make_row("O2", "missing_observation", 2)]
        trainer = make_trainer(dict(M1=-4., M2=-2., G0=-4., G2=-2., G3=-8., O0=-4., O2=-2.))
        history = make_batch(rows)
        metrics = trainer._annotate_trace_credits(history)
        values, credit = history.non_tensor_batch["trace_value"], history.non_tensor_batch["trace_credit"]
        self.assertTrue(np.isnan(values[:2]).all())
        self.assertTrue(np.isnan(credit[:3]).all())
        self.assertLess(credit[3], 0)  # G2 -> G3 is a genuine adjacent pair.
        self.assertTrue(np.isnan(credit[4:]).all())
        self.assertEqual(values[5], 0)
        self.assertTrue(np.isnan(values[6]))
        self.assertGreater(values[7], 0)
        self.assertEqual(metrics["trace/missing_state"], 1)
        self.assertIsNone(trainer._trans_to_dpo_batch([history]))

    def test_gt_mask_only_temperature_one_and_probe_padding_do_not_change_rollout(self):
        rows = [make_row("A", "a", 0, gt="AB"), make_row("B", "a", 1, gt="C"),
                make_row("C", "a", 2, gt="D")]
        trainer = make_trainer({"A": [-2., -4.], "B": [-5.], "C": [-6.]})
        history = make_batch(rows)
        original = {key: value.clone() for key, value in history.batch.items()}
        metrics = trainer._annotate_trace_credits(history)
        np.testing.assert_allclose(history.non_tensor_batch["trace_gt_logp"], [-3., -5., -6.])
        self.assertEqual(len(history), 3)
        for key, value in original.items():
            self.assertTrue(torch.equal(history.batch[key], value), key)
        self.assertEqual(history.meta_info, {"temperature": 0.7})
        self.assertEqual(metrics["trace/scoring_requests"], 3)
        probe = trainer.ref_policy_wg.calls[0]
        self.assertEqual(len(probe), 4)  # scorer world-size padding was removed.
        self.assertEqual(probe.meta_info, {"trace_gt_scoring": True, "temperature": 1.0})
        self.assertEqual(probe.batch["response_mask"].tolist(), [[1, 1], [1, 0], [1, 0], [1, 1]])
        for index, gt in enumerate(("AB", "C", "D")):
            self.assertEqual(probe.batch["responses"][index][probe.batch["response_mask"][index].bool()].tolist(),
                             trainer.tokenizer.encode(gt))

    def test_reference_inside_actor_uses_same_independent_gt_probe(self):
        rows = [make_row("A", "a", 0), make_row("B", "a", 1)]
        trainer = make_trainer(dict(A=-4., B=-2.))
        trainer.ref_in_actor = True
        trainer.actor_rollout_wg = FrozenReference(trainer.tokenizer, dict(A=-4., B=-2.))
        history = make_batch(rows)
        trainer._annotate_trace_credits(history)
        self.assertEqual(len(trainer.actor_rollout_wg.calls), 1)
        self.assertEqual(trainer.ref_policy_wg.calls, [])
        self.assertGreater(history.non_tensor_batch["trace_credit"][0], 0)

    def test_trace_requires_frozen_reference(self):
        trainer = make_trainer({})
        trainer.use_reference_policy = False
        with self.assertRaisesRegex(RuntimeError, "frozen reference"):
            trainer._annotate_trace_credits(make_batch([make_row("A", "a", 0)]))

    def test_disabled_trace_uses_real_legacy_path_without_trace_metadata(self):
        trainer = make_trainer({}, trace={"enabled": False})
        trainer._trans_to_trace_batch = lambda _: self.fail("Disabled TRACE called its new pairing path")
        history = make_batch([make_row("A", "a", 0, old=-1., reward=1.),
                              make_row("B", "b", 0, old=-3., reward=0.)])
        for key in list(history.non_tensor_batch):
            if key.startswith("trace_"):
                del history.non_tensor_batch[key]
        with contextlib.redirect_stdout(io.StringIO()):
            pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].non_tensor_batch["rollout_id_list_a"].tolist(), ["a"])
        self.assertEqual(pairs[0].non_tensor_batch["rollout_id_list_b"].tolist(), ["b"])
        self.assertEqual(pairs[0].non_tensor_batch["group_baseline"].tolist(), [-2.])


class LegacySlicTurnTests(unittest.TestCase):
    def trainer(self, task, algorithm="grpo+slic"):
        trainer = make_trainer({}, trace={"enabled": False})
        trainer.algorithm = algorithm
        trainer.config.data["algorithm"] = algorithm
        trainer.config.data["task"] = task
        return trainer

    def pairs(self, trainer, histories):
        with contextlib.redirect_stdout(io.StringIO()):
            return trainer._trans_to_dpo_batch(histories)

    def history(self, task, positive=3, negative=3):
        rows = []
        for index in range(positive + negative):
            success = index < positive
            for turn in (0, 1):
                rows.append(make_row(f"state-{index}-{turn}", f"traj-{index}", turn,
                                     reward=float(success), old=-1. - index))
        history = make_batch(rows)
        history.non_tensor_batch["anchor_list"][:] = "tool\nsame state" if task == "qa" else "same state"
        return history

    def test_slic_without_trace_limits_each_task_turn_independently(self):
        for task in ("qa", "alfworld"):
            with self.subTest(task=task):
                trainer, history = self.trainer(task), self.history(task)
                pairs = self.pairs(trainer, [history])
                counts = defaultdict(int)
                for pair in pairs:
                    left, right = pair.non_tensor_batch["turn_index_list_a"][0], pair.non_tensor_batch["turn_index_list_b"][0]
                    self.assertEqual(left, right)
                    counts[int(left)] += 1
                self.assertEqual(dict(counts), {0: 2, 1: 2})
                self.assertEqual(len(self.pairs(trainer, [history, history])), 4)

    def test_slic_without_trace_rejects_cross_turn_positive_negative_actions(self):
        for task in ("qa", "alfworld"):
            with self.subTest(task=task):
                history = make_batch([make_row("positive", "a", 0, reward=1.),
                                      make_row("negative", "b", 1, reward=0.)])
                history.non_tensor_batch["anchor_list"][:] = "tool\nsame state" if task == "qa" else "same state"
                self.assertIsNone(self.pairs(self.trainer(task), [history]))

    def test_legacy_slic_uses_original_turn_ids_not_compacted_ids(self):
        for task in ("qa", "alfworld"):
            with self.subTest(task=task):
                history = make_batch([make_row("positive", "a", 0, reward=1.),
                                      make_row("negative", "b", 0, reward=0.)])
                history.non_tensor_batch["anchor_list"][:] = "tool\nsame state" if task == "qa" else "same state"
                history.non_tensor_batch["trace_original_turn"] = np.array([3, 4])
                self.assertIsNone(self.pairs(self.trainer(task), [history]))
                history.non_tensor_batch["trace_original_turn"][:] = 3
                pair = self.pairs(self.trainer(task), [history])[0]
                self.assertEqual(pair.non_tensor_batch["turn_index_list_a"].tolist(), [3])
                self.assertEqual(pair.non_tensor_batch["turn_index_list_b"].tolist(), [3])
                np.testing.assert_array_equal(history.non_tensor_batch["turn_index_list"], [0, 0])

    def test_qa_analysis_final_legacy_grouping_and_no_cap_default_are_preserved(self):
        rows, anchors = [], []
        for index in range(6):
            success = index < 3
            for stage, turn in (("analysis", index), ("final", index + 10)):
                rows.append(make_row(f"{stage}-{index}", f"traj-{index}", turn,
                                     reward=float(success), old=-1. - index))
                anchors.append(stage + "\nsame state")
        history = make_batch(rows)
        history.non_tensor_batch["anchor_list"] = np.array(anchors)
        pairs = self.pairs(self.trainer("qa"), [history])
        self.assertEqual(len(pairs), 6)  # The original default None cap remains for outcomes.
        self.assertTrue(any(pair.non_tensor_batch["turn_index_list_a"][0]
                            != pair.non_tensor_batch["turn_index_list_b"][0] for pair in pairs))

    def test_other_preference_algorithms_keep_their_original_cross_turn_path(self):
        for task in ("qa", "alfworld"):
            with self.subTest(task=task):
                history = make_batch([make_row("positive", "a", 0, reward=1.),
                                      make_row("negative", "b", 1, reward=0.)])
                history.non_tensor_batch["anchor_list"][:] = "tool\nsame state" if task == "qa" else "same state"
                history.non_tensor_batch["trace_original_turn"] = np.array([50, 90])
                pairs = self.pairs(self.trainer(task, "grpo+abpo"), [history])
                self.assertEqual(len(pairs), 1)
                self.assertEqual(pairs[0].non_tensor_batch["turn_index_list_a"].tolist(), [0])
                self.assertEqual(pairs[0].non_tensor_batch["turn_index_list_b"].tolist(), [1])

    def test_other_slic_tasks_keep_the_original_explicit_none_cap(self):
        trainer = self.trainer("webshop")
        trainer.config.data["replay_filter"] = {"k_pairs_per_group": None}
        pairs = self.pairs(trainer, [self.history("webshop")])
        self.assertEqual(len(pairs), 6)


if __name__ == "__main__":
    unittest.main()
