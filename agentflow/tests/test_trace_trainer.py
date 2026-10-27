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


CREDIT = load_source_module("trace_credit_trainer_test", ROOT / "agentflow/agentflow/trace_credit.py")
SCORING = load_source_module("trace_scoring_trainer_test", ROOT / "agentflow/agentflow/trace_scoring.py")
METRICS = load_source_module("trace_metrics_trainer_test", ROOT / "agentflow/agentflow/trace_metrics.py")
EXECUTION = load_source_module("trace_execution_trainer_test", ROOT / "agentflow/agentflow/trace_execution.py")
IGPO = load_source_module("igpo_credit_trainer_test", ROOT / "agentflow/agentflow/igpo_credit.py")


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
    names = {"_anchors_match", "_annotate_trace_credits", "_trans_to_trace_batch", "_trans_to_dpo_batch"}
    methods = [node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(methods) != len(names):
        raise AssertionError("Required TRACE trainer methods were not found")
    scope = dict(
        torch=torch, np=np, random=random, defaultdict=defaultdict, DataProto=Proto, time=time,
        TraceConfig=CREDIT.TraceConfig, state_values=CREDIT.state_values,
        annotate_rollout_credits=CREDIT.annotate_rollout_credits, select_pairs=CREDIT.select_pairs,
        compute_igpo_advantages=IGPO.compute_igpo_advantages,
        summarize_states=METRICS.summarize_states,
        summarize_selected_pairs=METRICS.summarize_selected_pairs,
        summarize_igpo=METRICS.summarize_igpo,
        encode_state_requests=SCORING.encode_state_requests,
        build_score_tensors=SCORING.build_score_tensors, mean_gt_logps=SCORING.mean_gt_logps,
        pad_dataproto_to_divisor=pad_dataproto_to_divisor, unpad_dataproto=unpad_dataproto,
    )
    shell = ast.ClassDef(name="TrainerMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[shell], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["TrainerMethods"]


TrainerMethods = load_trainer_methods()


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
                        trace={"enabled": True, "pair_score_mode": "trace_fill", "k": 1,
                               "terminal_fill_enabled": False, **(trace or {})}),
        algorithm=ConfigDict(norm_adv_by_std_in_grpo=False),
    )
    trainer.algorithm = "slic"
    trainer.global_steps = 12
    trainer.use_reference_policy = True
    trainer.ref_in_actor = False
    trainer.tokenizer = Tokenizer()
    trainer.ref_policy_wg = FrozenReference(trainer.tokenizer, scores)
    return trainer


def standard_rows(chosen_reward=1.0, rejected_reward=1.0):
    # Positive action A1 and negative action B0 have the same value but
    # different original turn numbers and different decision prompts.
    rows = [
        make_row("A0", "a", 0, turn_type="analysis", reward=chosen_reward),
        make_row("A1", "a", 1, old=-1, reward=chosen_reward),
        make_row("A2", "a", 2, turn_type="final_answer", reward=chosen_reward),
        make_row("B0", "b", 0, old=-3, reward=rejected_reward),
        make_row("B1", "b", 1, turn_type="final_answer", reward=rejected_reward),
    ]
    return rows, {"A0": -4., "A1": -4., "A2": -2., "B0": -4., "B1": -8.}


class TraceTrainerIntegrationTests(unittest.TestCase):
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
        self.assertEqual(metrics["trace/pair/cross_turn_rate"], 1.)
        self.assertEqual(metrics["trace/pair/question_coverage"], 1.)
        self.assertEqual(metrics["trace/pair/trace/outcome_pairs_count"], 0)
        self.assertNotIn("trace/pair/chosen_failed_rollout_rate", metrics)

    def test_analysis_and_final_use_verified_outcomes_without_gt_scores(self):
        rows = []
        for rollout, reward, success in (("a", 1., True), ("b", 0.1, False)):
            rows.extend([
                make_row(rollout + "0", rollout, 0, turn_type="analysis", gt="",
                         reward=reward, outcome_valid=True, outcome_success=success),
                make_row(rollout + "1", rollout, 1, turn_type="final_answer", gt="",
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

    def test_k3_fill_uses_final_decision_and_unique_rollout_outcomes(self):
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
        np.testing.assert_allclose(credit[[1, 2, 3]], [0., 0.64, 0.8])
        np.testing.assert_allclose(credit[[6, 7, 8]], [0., -0.64, -0.8])
        np.testing.assert_allclose(history.non_tensor_batch["trace_outcome_advantage"][:5], 0.5)
        np.testing.assert_allclose(history.non_tensor_batch["trace_local_credit"][[1, 2, 3]], 0.)
        self.assertEqual(history.non_tensor_batch["trace_terminal_reached"].tolist(),
                         [False, False, True, True, False] * 2)
        self.assertEqual(len(trainer._trans_to_dpo_batch([history])), 4)

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
                scores[context] = -4.
        trainer = make_trainer(scores, {"k": 3, "terminal_fill_enabled": True})
        history = make_batch(rows)
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 3)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/trace_selected_unique"], 1)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/outcome_selected_unique"], 2)
        np.testing.assert_allclose(history.non_tensor_batch["trace_credit"][[1, 4]], [0.8, -0.8])

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

    def test_cross_turn_pair_keeps_each_prompt_and_ignores_final_reward(self):
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
                self.assertEqual(pair.non_tensor_batch["turn_index_list_b"].tolist(), [0])
                self.assertTrue(torch.equal(pair.batch["prompts_a"], history.batch["prompts"][1:2]))
                self.assertTrue(torch.equal(pair.batch["prompts_b"], history.batch["prompts"][3:4]))
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
        self.assertEqual(history.non_tensor_batch["trace_credit"][5], 0.)

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
        scores["B1"] = -2.
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


def igpo_rows():
    rows, scores = [], {}
    for rollout, reward, success, tool_scores in (
            ("a", 1., True, (-8., -6., -4., -2.)),
            ("b", 0., False, (-8., -7., -6., -5.))):
        for turn, turn_type in enumerate(("analysis", "tool_plan", "tool_plan", "tool_plan", "final_answer")):
            context = rollout + str(turn)
            rows.append(make_row(context, rollout, turn, turn_type=turn_type,
                                 old=-1. if rollout == "a" else -3., reward=reward,
                                 is_final=turn == 4, outcome_valid=True, outcome_success=success))
            scores[context] = -8. if turn == 0 else tool_scores[turn - 1]
    return rows, scores


class IgpoTrainerIntegrationTests(unittest.TestCase):
    def trainer(self, scores, **overrides):
        return make_trainer(scores, {"pair_score_mode": "igpo", **overrides})

    def test_full_return_uses_population_z_and_does_not_apply_legacy_fill(self):
        rows, scores = igpo_rows()
        trainer = self.trainer(scores, k=1, terminal_fill_enabled=True, terminal_fill_scale=999.)
        history = make_batch(rows)
        history.batch["advantages"] = torch.full((len(rows), 3), 42.)
        original = {key: value.clone() for key, value in history.batch.items()}
        metrics = trainer._annotate_trace_credits(history)
        non = history.non_tensor_batch
        unit = 1. / (1. + 2e-6)
        np.testing.assert_allclose(non["trace_igpo_advantage"][[1, 2, 3]],
                                   np.array([4., 3., 2.]) * unit)
        np.testing.assert_allclose(non["trace_igpo_advantage"][[6, 7, 8]],
                                   -np.array([4., 3., 2.]) * unit)
        np.testing.assert_allclose(non["trace_igpo_terminal_return"][[1, 2, 3]], unit)
        self.assertTrue(np.isnan(non["trace_igpo_advantage"][[0, 4, 5, 9]]).all())
        for key in ("trace_credit", "trace_local_credit", "trace_terminal_fill", "trace_outcome_advantage"):
            self.assertTrue(np.isnan(non[key]).all(), key)
        for key, before in original.items():
            self.assertTrue(torch.equal(history.batch[key], before), key)
        self.assertNotIn("trace/credit_mean", metrics)
        self.assertNotIn("trace/terminal_fill_mean", metrics)
        self.assertEqual(metrics["trace/igpo/advantage_coverage"], 1.)

    def test_relative_advantage_can_pair_two_absolutely_positive_ig_actions(self):
        rows, scores = igpo_rows()
        trainer, history = self.trainer(scores), make_batch(rows)
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_dpo_batch([history])
        tool_pairs = [pair for pair in pairs if pair.non_tensor_batch["trace_turn_type_a"][0] == "tool_plan"]
        self.assertGreater(len(tool_pairs), 0)
        first = tool_pairs[0]
        self.assertGreater(first.non_tensor_batch["trace_igpo_raw_ig_a"][0], 0)
        self.assertGreater(first.non_tensor_batch["trace_igpo_raw_ig_b"][0], 0)
        self.assertGreater(first.non_tensor_batch["trace_igpo_advantage_a"][0], 0)
        self.assertLess(first.non_tensor_batch["trace_igpo_advantage_b"][0], 0)
        self.assertEqual(first.non_tensor_batch["group_baseline"].tolist(), [-2.])
        self.assertEqual(first.non_tensor_batch["trace_pair_score_mode_a"].tolist(), ["igpo"])
        self.assertTrue(torch.equal(first.batch["prompts_a"], history.batch["prompts"][1:2]))
        self.assertTrue(torch.equal(first.batch["prompts_b"], history.batch["prompts"][6:7]))
        self.assertIn("trace/pair/igpo_advantage_gap_p50", trainer._trace_pair_metrics)
        self.assertNotIn("trace/pair/credit_gap_p50", trainer._trace_pair_metrics)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/outcome_selected_unique"], 2)

    def test_discount_uses_original_distance_and_reaches_earliest_tool(self):
        rows, scores = igpo_rows()
        trainer, history = self.trainer(scores, igpo_gamma=0.8), make_batch(rows)
        trainer._annotate_trace_credits(history)
        unit = 1. / (1. + 2e-6)
        self.assertAlmostEqual(history.non_tensor_batch["trace_igpo_advantage"][1],
                               (1. + .8 + .8**2 + .8**3) * unit)
        self.assertAlmostEqual(history.non_tensor_batch["trace_igpo_terminal_return"][1],
                               .8**3 * unit)

    def test_all_failed_rollouts_still_get_igpo_tool_pairs(self):
        rows, scores = igpo_rows()
        for row in rows:
            row.update(reward=0., outcome_success=False)
        trainer, history = self.trainer(scores), make_batch(rows)
        trainer._annotate_trace_credits(history)
        np.testing.assert_allclose(history.non_tensor_batch["trace_igpo_outcome_z"], 0.)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertGreater(len(pairs), 0)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/outcome_selected_unique"], 0)
        self.assertGreater(trainer._trace_pair_metrics["trace/pair/trace_selected_unique"], 0)

    def test_constant_ig_does_not_fall_back_to_outcome_only_tool_pairs(self):
        rows, scores = igpo_rows()
        for turn, score in enumerate((-8., -8., -6., -4., -2.)):
            scores["b" + str(turn)] = score
        trainer, history = self.trainer(scores), make_batch(rows)
        trainer._annotate_trace_credits(history)
        self.assertFalse(history.non_tensor_batch["trace_igpo_valid"].any())
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 2)  # Trusted analysis and final routes remain.
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/trace_selected_unique"], 0)

    def test_invalid_future_turn_blocks_prefix_but_preserves_clean_suffix(self):
        rows, scores = igpo_rows()
        rows[2]["valid"] = False
        trainer, history = self.trainer(scores), make_batch(rows)
        trainer._annotate_trace_credits(history)
        non = history.non_tensor_batch
        self.assertTrue(non["trace_igpo_blocked"][[1, 2]].all())
        self.assertTrue(np.isnan(non["trace_igpo_advantage"][[1, 2]]).all())
        self.assertTrue(non["trace_igpo_valid"][[3, 6, 7, 8]].all())
        self.assertFalse(non["trace_igpo_external_blocked"].any())

    def test_untrusted_terminal_and_duplicates_do_not_create_mixed_returns(self):
        rows, scores = igpo_rows()
        for row in rows[5:]:
            row["outcome_valid"] = False
        trainer, history = self.trainer(scores), make_batch(rows)
        trainer._annotate_trace_credits(history)
        self.assertFalse(history.non_tensor_batch["trace_igpo_valid"].any())
        self.assertIsNone(trainer._trans_to_dpo_batch([history]))
        # A duplicate inside a future return is a barrier even with trusted R.
        rows, scores = igpo_rows()
        history = make_batch(rows)
        history.non_tensor_batch["trace_dup"] = np.array([False] * len(rows))
        history.non_tensor_batch["trace_dup"][2] = True
        trainer = self.trainer(scores)
        trainer._annotate_trace_credits(history)
        self.assertTrue(history.non_tensor_batch["trace_igpo_blocked"][[1, 2]].all())
        self.assertTrue(history.non_tensor_batch["trace_igpo_valid"][3])

    def test_absolute_cumulative_guard_is_optional_and_never_changes_population(self):
        rows, scores = igpo_rows()
        for row in rows:
            row.update(reward=0., outcome_success=False)
        guarded = self.trainer(scores, igpo_absolute_ig_guard="cumulative")
        history = make_batch(rows)
        guarded._annotate_trace_credits(history)
        self.assertTrue(history.non_tensor_batch["trace_igpo_valid"][[1, 2, 3, 6, 7, 8]].all())
        self.assertIsNone(guarded._trans_to_dpo_batch([history]))

    def test_outcome_routes_work_without_gt_in_igpo_mode(self):
        rows = []
        for rollout, reward, success in (("a", 1., True), ("b", 0., False)):
            rows.extend([
                make_row(rollout + "0", rollout, 0, turn_type="analysis", gt="",
                         reward=reward, outcome_valid=True, outcome_success=success),
                make_row(rollout + "1", rollout, 1, turn_type="final_answer", gt="",
                         reward=reward, is_final=True, outcome_valid=True, outcome_success=success),
            ])
        trainer, history = self.trainer({}), make_batch(rows)
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 2)
        self.assertTrue(np.isnan(history.non_tensor_batch["trace_igpo_advantage"]).all())


if __name__ == "__main__":
    unittest.main()
