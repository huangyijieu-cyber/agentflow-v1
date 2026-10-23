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
        discounted_credits=CREDIT.discounted_credits, select_pairs=CREDIT.select_pairs,
        summarize_states=METRICS.summarize_states,
        summarize_selected_pairs=METRICS.summarize_selected_pairs,
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


def make_row(context, rollout, turn, *, turn_type="tool", old=-1.0, reward=1.0,
             valid=True, dropped=False, uid="q", gt="AB"):
    return dict(context=context, rollout=rollout, turn=turn, turn_type=turn_type,
                old=old, reward=reward, valid=valid, dropped=dropped, uid=uid, gt=gt)


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
                        trace={"enabled": True, **(trace or {})}),
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
        make_row("A2", "a", 2, turn_type="final", reward=chosen_reward),
        make_row("B0", "b", 0, old=-3, reward=rejected_reward),
        make_row("B1", "b", 1, turn_type="final", reward=rejected_reward),
    ]
    return rows, {"A0": -4., "A1": -4., "A2": -2., "B0": -4., "B1": -8.}


class TraceTrainerIntegrationTests(unittest.TestCase):
    def test_metric_coverage_funnel_and_outcome_disagreement_are_actual(self):
        rows, scores = standard_rows(chosen_reward=0., rejected_reward=1.)
        trainer, history = make_trainer(scores), make_batch(rows)
        state_metrics = trainer._annotate_trace_credits(history)
        self.assertEqual(state_metrics["trace/score_coverage"], 1.)
        self.assertEqual(state_metrics["trace/credit_expected_turns"], 3)
        self.assertEqual(state_metrics["trace/credit_coverage"], 1.)
        self.assertEqual(state_metrics["trace/terminal_turns"], 2)
        self.assertGreaterEqual(state_metrics["trace/scoring_seconds"], 0.)
        self.assertEqual(state_metrics["trace/missing_gt"], 0)
        trainer._trans_to_dpo_batch([history])
        metrics = trainer._trace_pair_metrics
        self.assertEqual(metrics["trace/pair/candidates_sign"], 1)
        self.assertEqual(metrics["trace/pair/after_value_match"], 1)
        self.assertEqual(metrics["trace/pair/selected_unique"], 1)
        self.assertEqual(metrics["trace/pair/cross_turn_rate"], 1.)
        self.assertEqual(metrics["trace/pair/question_coverage"], 1.)
        self.assertEqual(metrics["trace/pair/chosen_failed_rollout_rate"], 1.)
        self.assertEqual(metrics["trace/pair/rejected_successful_rollout_rate"], 1.)

    def test_distributed_padding_does_not_inflate_pair_statistics(self):
        rows, scores = standard_rows()
        trainer, history = make_trainer(scores), make_batch(rows)
        trainer.config.trainer.n_gpus_per_node = 8
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_dpo_batch([history])
        self.assertEqual(len(pairs), 8)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/selected_unique"], 1)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/padding_duplicates"], 7)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/credit_gap_count"], 1)

    def test_global_cap_across_history_batches_preserves_funnel_counts(self):
        rows, scores = standard_rows()
        trainer = make_trainer(scores, trace={"max_pairs_per_step": 1})
        first, second = make_batch(rows), make_batch(rows)
        trainer._annotate_trace_credits(first)
        trainer._annotate_trace_credits(second)
        self.assertEqual(len(trainer._trans_to_dpo_batch([first, second])), 1)
        metrics = trainer._trace_pair_metrics
        self.assertEqual(metrics["trace/pair/after_distinct_action"], 2)
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
                     make_row("C1", "c", 1, turn_type="final")])
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
                make_row("A2", "a", 2, turn_type="final"),
                make_row("B0", "b", 0), make_row("B1", "b", 1, turn_type="final")]
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
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/negative_rows"], 0)
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


if __name__ == "__main__":
    unittest.main()
