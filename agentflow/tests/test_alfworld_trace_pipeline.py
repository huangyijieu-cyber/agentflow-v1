"""ALFWorld metadata contracts across real rollout, daemon and TRACE trainer code.

The simulator/frozen reference are CPU stubs; the production adapter methods
are extracted without importing distributed or ALFWorld runtime dependencies.
"""
import copy
import ast
import asyncio
import re
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

import test_trace_states as states
import test_trace_trainer as training


EXECUTION = training.EXECUTION
GOAL = "A clean mug is in a cabinet."
INSTRUCTION = training.ALFWORLD.FORECAST_INSTRUCTION
PROVENANCE = {"goal_spec_hash": "actual-game-v1", "source_sha256": {"gamefile": "official-game"},
              "scoring_version": training.ALFWORLD.FORECAST_SCORING_VERSION,
              "scoring_response": training.ALFWORLD.FORECAST_RESPONSE}


class AlfworldPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        states.DaemonMetadataTests.setUpClass()
        cls.daemon_builder = states.DaemonMetadataTests()

    def logs(self, name, *, success=True, invalid=None, external=None):
        logs = []
        for index in range(4):
            log = {
                "prompt": f"actor visible observation {index}", "response": f"<action>act {index}</action>",
                "prompt_token_ids": [10, index + 11], "response_token_ids": [100 + index],
                "finish_reason": "stop", "trace_original_turn": index,
                "trace_gt": GOAL, "trace_answer_instruction": INSTRUCTION,
                "trace_goal_metadata": copy.deepcopy(PROVENANCE), "trace_state_view": "full_state",
                "trace_goal_check_before": False, "trace_goal_check_valid_before": True,
                "trace_scoring_valid": True,
            }
            final = index == 3
            EXECUTION.annotate_alfworld_step(log, name + str(index), final, {
                "trace_state_context": name + str(index + 1),
                "trace_env_action": f"act {index}", "terminated": final and success,
                "truncated": final and not success, "won": final and success,
                "trace_goal_check": final and success,
                "trace_goal_check_valid": index != invalid,
                "trace_scoring_valid": index != invalid,
                "env_error": index == external,
            })
            logs.append(log)
        EXECUTION.annotate_trace_outcome(logs, {"base_response": logs[-1]["response"]},
                                          float(success), None, "alfworld")
        return logs

    def encode(self, logs, path):
        encoder = states.extract(path, "encode_logs", {"Triplet": NS})
        return encoder(None, logs) if "profiled" in path else encoder(logs)

    def batch(self, path, *, invalid=None, external=None):
        completed, originals = {}, {}
        flags = states.extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        for name, success in (("A", True), ("B", False)):
            logs = self.logs(name, success=success, invalid=invalid, external=external)
            completed[name] = NS(triplets=self.encode(logs, path), metadata={"turn_flags": flags(logs, "alfworld")},
                                 reward=float(success))
            originals[name] = {"data_id": "same-question", "result": GOAL}  # Prepared Parquet is authoritative.
        daemon = NS(is_train=True, _completed_rollouts=completed, _total_tasks_queued=2,
                    pad_token_id=0, _task_id_to_original_sample=originals,
                    _fillna_reward=lambda rollout: rollout.reward,
                    _get_format_metrics=lambda prefix: {})
        batch, _ = self.daemon_builder.get_batch(daemon, 4, 4, "cpu")
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"].copy()
        batch.non_tensor_batch["traj_uid"] = batch.non_tensor_batch["traj_id_list"].copy()
        batch.batch["response_mask"] = batch.batch["attention_mask"][:, -4:].clone()
        batch.batch["old_log_probs"] = torch.full_like(batch.batch["responses"], -1., dtype=torch.float32)
        batch.batch["ref_log_prob"] = torch.full_like(batch.batch["responses"], -1., dtype=torch.float32)
        batch.meta_info = {"temperature": .7}
        return batch

    def trainer(self):
        scores = {"A0": -4., "A1": -3., "A2": -2., "A3": -1., "A4": -.5,
                  "B0": -4., "B1": -5., "B2": -6., "B3": -7., "B4": -8.}
        trainer = training.make_trainer(scores, {"k": 3, "gamma": .8, "epsilon": .1,
                                               "terminal_fill_enabled": True, "terminal_fill_scale": 2.})
        trainer.config.data["task"] = "alfworld"
        return trainer

    def test_both_entrypoints_use_fixed_gt_full_state_and_uniform_action_credit(self):
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            with self.subTest(path=path):
                history, trainer = self.batch(path), self.trainer()
                before = {key: value.clone() for key, value in history.batch.items()}
                non = history.non_tensor_batch
                self.assertEqual(non["trace_gt"].tolist(), [GOAL] * 8)
                self.assertEqual(non["trace_goal_metadata"].shape, (8,))
                self.assertEqual(non["trace_answer_instruction"].tolist(), [INSTRUCTION] * 8)
                self.assertEqual(non["trace_state_view"].tolist(), ["full_state"] * 8)
                self.assertEqual(non["trace_goal_check"].tolist(), [False, False, False, True] + [False] * 4)
                self.assertTrue(non["trace_goal_check_valid"].all())
                metrics = trainer._annotate_trace_credits(history)
                self.assertEqual(metrics["trace/scoring_requests"], 10)  # Two rollouts, T+1 states each.
                self.assertEqual(metrics["trace/credit_expected_turns"], 8)
                self.assertEqual(non["trace_label_source"].tolist(), ["trace"] * 8)
                self.assertEqual(non["trace_turn_type"].tolist(), ["tool_plan"] * 8)
                self.assertFalse(non["trace_is_final"].any())
                self.assertEqual(non["trace_is_terminal"].tolist(), [False, False, False, True] * 2)
                self.assertTrue(np.isfinite(non["trace_credit"][[3, 7]]).all())
                self.assertEqual(history.batch["responses"].shape[0], 8)  # Terminal post-states do not create actor samples.
                np.testing.assert_allclose(non["trace_outcome_advantage"], [.5] * 4 + [-.5] * 4)
                cfg = training.CREDIT.TraceConfig(k=3, gamma=.8, epsilon=.1)
                for start in (0, 4):
                    expected = training.CREDIT.annotate_environment_credits(
                        non["trace_value"][start:start + 4], non["trace_next_value"][start:start + 4],
                        [0, 1, 2, 3], [False] * 4, [False] * 3 + [True],
                        outcome_valid=True, outcome_advantage=.5 if start == 0 else -.5, config=cfg)
                    np.testing.assert_allclose(non["trace_credit"][start:start + 4], expected["effective_credit"], equal_nan=True)
                self.assertEqual(non["trace_terminal_reached"].tolist(), [False, True, True, True] * 2)
                np.testing.assert_allclose(non["trace_terminal_fill"][:4], [0., .64, .8, 1.])
                np.testing.assert_allclose(non["trace_next_value"][:3], non["trace_value"][1:4])
                expected_last = np.log(4.1 / .6) - np.log(4.1 / 1.1)
                self.assertAlmostEqual(non["trace_local_credit"][3], expected_last)
                for key, value in before.items():
                    self.assertTrue(torch.equal(history.batch[key], value), key)
                for call in trainer.ref_policy_wg.calls:
                    for row in range(len(call)):
                        prompt = trainer.tokenizer.decode(call.batch["prompts"][row])
                        self.assertIn(INSTRUCTION, prompt)
                        self.assertNotIn("actor visible", prompt)
                        response = trainer.tokenizer.decode(
                            call.batch["responses"][row][call.batch["response_mask"][row].bool()])
                        self.assertEqual(response, training.ALFWORLD.FORECAST_RESPONSE)
                pairs = trainer._trans_to_trace_batch([history])
                self.assertTrue(pairs)
                self.assertTrue(all(pair.non_tensor_batch["trace_label_source_a"][0] == "trace" for pair in pairs))
                self.assertTrue(all(pair.non_tensor_batch["trace_turn_type_a"][0] == "tool_plan" for pair in pairs))
                for pair in pairs:
                    self.assertEqual(pair.non_tensor_batch["trace_gt_a"].tolist(), [GOAL])
                    self.assertEqual(pair.non_tensor_batch["trace_task_a"].tolist(), ["alfworld"])

    def test_skipping_action_reference_keeps_real_alfworld_gt_credit_and_pairs(self):
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            with self.subTest(path=path):
                reference, optimized = self.batch(path), self.batch(path)
                optimized.batch.pop("ref_log_prob")
                original_trainer, optimized_trainer = self.trainer(), self.trainer()
                original_trainer._annotate_trace_credits(reference)
                metrics, timing = {}, {}
                optimized = training.action_ref_stage(optimized_trainer, optimized, metrics, timing)
                self.assertEqual(optimized_trainer.ref_policy_wg.calls, [])
                optimized_trainer._annotate_trace_credits(optimized)
                self.assertEqual(len(optimized_trainer.ref_policy_wg.calls), 1)
                self.assertEqual(metrics["perf/action_ref_log_prob_skipped"], 1)
                for key in ("trace_gt_logp", "trace_next_gt_logp", "trace_value", "trace_next_value",
                            "trace_local_credit", "trace_terminal_fill", "trace_credit"):
                    np.testing.assert_array_equal(reference.non_tensor_batch[key], optimized.non_tensor_batch[key])
                original_pairs = original_trainer._trans_to_trace_batch([reference])
                optimized_pairs = optimized_trainer._trans_to_trace_batch([optimized])
                self.assertTrue(optimized_pairs)
                self.assertEqual(len(original_pairs), len(optimized_pairs))
                for old, new in zip(original_pairs, optimized_pairs):
                    self.assertNotIn("ref_log_prob_a", new.batch)
                    for key in old.batch:
                        self.assertTrue(torch.equal(old.batch[key], new.batch[key]), key)
                    for key in ("group_baseline", "group_size", "turn_index_list_a", "turn_index_list_b"):
                        np.testing.assert_array_equal(old.non_tensor_batch[key], new.non_tensor_batch[key])

    def test_invalid_goal_state_blocks_reference_credit_and_terminal_labels(self):
        history, trainer = self.batch("train-roma/rollout.py", invalid=1), self.trainer()
        trainer._annotate_trace_credits(history)
        non = history.non_tensor_batch
        self.assertIsNone(non["trace_state_context"][1])
        self.assertTrue(np.isnan(non["trace_gt_logp"][1]))
        self.assertTrue(np.isnan(non["trace_credit"][0]))
        self.assertTrue(np.isnan(non["trace_credit"][1]))
        self.assertFalse(non["trace_outcome_valid"].any())
        self.assertFalse(non["trace_valid"][1])
        # Environment terminal identity only bounds the action window/fill.
        self.assertFalse(non["trace_is_final"][3])
        self.assertTrue(non["trace_is_terminal"][3])
        self.assertEqual(non["trace_turn_type"][3], "tool_plan")

    def test_shared_prefix_route_keeps_alfworld_post_state_credits_and_pairs(self):
        class SharedReference(training.FrozenReference):
            def compute_ref_prefix_log_prob(self, probes):
                self.prefix_meta = dict(probes.meta_info)
                native = training.Proto(probes.batch, probes.non_tensor_batch,
                                        {"trace_gt_scoring": True, "temperature": 1.0})
                scored = super().compute_ref_log_prob(native)
                scored.meta_info["trace_prefix_metrics"] = {"trace/prefix_used": 1}
                return scored

        original, shared = self.batch("train-roma/rollout.py"), self.batch("train-roma/rollout.py")
        old_trainer, new_trainer = self.trainer(), self.trainer()
        new_trainer.config.data.trace["shared_prefix_scoring"] = True
        scores = new_trainer.ref_policy_wg.scores
        new_trainer.ref_policy_wg = SharedReference(new_trainer.tokenizer, scores)
        old_trainer._annotate_trace_credits(original)
        metrics = new_trainer._annotate_trace_credits(shared)
        self.assertEqual(metrics["trace/prefix_used"], 1)
        self.assertEqual(metrics["trace/scoring_requests"], 10)
        self.assertEqual(len(new_trainer.ref_policy_wg.calls), 1)
        self.assertTrue(new_trainer.ref_policy_wg.prefix_meta["trace_prefix_verify"])
        self.assertEqual(shared.batch["responses"].shape[0], 8)
        for key in ("trace_gt_logp", "trace_next_gt_logp", "trace_value", "trace_next_value",
                    "trace_local_credit", "trace_terminal_fill", "trace_credit"):
            np.testing.assert_array_equal(original.non_tensor_batch[key], shared.non_tensor_batch[key])
        old_pairs, new_pairs = old_trainer._trans_to_trace_batch([original]), new_trainer._trans_to_trace_batch([shared])
        self.assertEqual(len(old_pairs), len(new_pairs))
        for old, new in zip(old_pairs, new_pairs):
            for key in ("group_baseline", "turn_index_list_a", "turn_index_list_b"):
                np.testing.assert_array_equal(old.non_tensor_batch[key], new.non_tensor_batch[key])

    def test_trainer_also_masks_invalid_state_without_daemon_cleanup(self):
        history, trainer = self.batch("train-roma/rollout.py"), self.trainer()
        history.non_tensor_batch["trace_scoring_valid"][0] = False
        trainer._annotate_trace_credits(history)
        self.assertTrue(np.isnan(history.non_tensor_batch["trace_gt_logp"][0]))
        self.assertFalse(history.non_tensor_batch["trace_outcome_valid"][:4].any())

    def test_different_actual_game_with_same_uid_fails_before_model_call(self):
        history, trainer = self.batch("train-roma/rollout.py"), self.trainer()
        history.non_tensor_batch["trace_goal_metadata"][4] = {"goal_spec_hash": "other-game"}
        with self.assertRaisesRegex(ValueError, "different actual games"):
            trainer._annotate_trace_credits(history)
        self.assertEqual(trainer.ref_policy_wg.calls, [])

    def test_v2b_rejects_legacy_or_changed_scoring_contract_before_model_call(self):
        for change in ({"scoring_version": None}, {"scoring_version": "alfworld-v2a"},
                       {"scoring_response": GOAL}, {"scoring_response": None}):
            with self.subTest(change=change):
                history, trainer = self.batch("train-roma/rollout.py"), self.trainer()
                history.non_tensor_batch["trace_goal_metadata"][0] = dict(PROVENANCE, **change)
                with self.assertRaisesRegex(ValueError, "v2b future-success"):
                    trainer._annotate_trace_credits(history)
                self.assertEqual(trainer.ref_policy_wg.calls, [])

    def test_v2b_forecast_scoring_preserves_original_goal_and_provenance(self):
        history, trainer = self.batch("train-roma/rollout.py"), self.trainer()
        original_goals = history.non_tensor_batch["trace_gt"].copy()
        original_provenance = copy.deepcopy(history.non_tensor_batch["trace_goal_metadata"])
        trainer._annotate_trace_credits(history)
        np.testing.assert_array_equal(history.non_tensor_batch["trace_gt"], original_goals)
        np.testing.assert_array_equal(history.non_tensor_batch["trace_goal_metadata"], original_provenance)
        for call in trainer.ref_policy_wg.calls:
            for index in range(len(call)):
                response = trainer.tokenizer.decode(
                    call.batch["responses"][index][call.batch["response_mask"][index].bool()])
                self.assertEqual(response, training.ALFWORLD.FORECAST_RESPONSE)

    def test_missing_actual_gt_cannot_fall_back_to_parquet(self):
        logs = self.logs("A")
        logs[0]["trace_gt"] = None
        with self.assertRaisesRegex(ValueError, "differs from the verified"):
            self.daemon_builder.build_batch(self.encode(logs, "train-roma/rollout.py"), ground_truth=GOAL)

    def test_empty_original_result_cannot_fall_back_to_environment_gt(self):
        with self.assertRaisesRegex(ValueError, "nonempty prepared Parquet result"):
            self.daemon_builder.build_batch(self.encode(self.logs("A"), "train-roma/rollout.py"), ground_truth="")

    def test_original_result_mismatch_is_rejected_in_both_entrypoints(self):
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "differs from the verified"):
                self.daemon_builder.build_batch(self.encode(self.logs("A"), path), ground_truth="Another task is complete.")

    def test_repeated_action_at_different_states_remains_trace_eligible(self):
        flags = states.extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        logs = self.logs("A")[:3]
        for log in logs:
            log["response"] = "<action>look</action>"
        logs[2]["trace_state_context"] = logs[0]["trace_state_context"]
        logs[2]["trace_action_key"] = logs[0]["trace_action_key"]
        result = flags(logs, "alfworld")
        self.assertEqual([row["dup"] for row in result], [False, True, True])
        self.assertEqual([row["trace_dup"] for row in result], [False, False, True])

    def test_terminal_action_cannot_pair_with_an_earlier_turn(self):
        history, trainer = self.batch("train-roma/rollout.py"), self.trainer()
        trainer.ref_policy_wg.scores.update({"A1": -4., "A2": -4., "A3": -4., "A4": -1.})
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_trace_batch([history])
        self.assertIsNone(pairs)  # The only value-matched sign pair would cross turn IDs.

    def test_terminal_actions_can_pair_at_the_same_turn(self):
        history, trainer = self.batch("train-roma/rollout.py"), self.trainer()
        trainer.ref_policy_wg.scores.update({"A1": -4., "A2": -4., "A3": -4., "A4": -1.,
                                             "B1": -4., "B2": -4., "B3": -4.})
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_trace_batch([history])
        self.assertTrue(any(pair.non_tensor_batch["trace_is_terminal_a"][0]
                            and pair.non_tensor_batch["trace_is_terminal_b"][0] for pair in pairs))
        self.assertTrue(all(pair.non_tensor_batch["turn_index_list_a"][0]
                            == pair.non_tensor_batch["turn_index_list_b"][0] for pair in pairs))
        self.assertTrue(all(pair.non_tensor_batch["trace_label_source_a"][0] == "trace" for pair in pairs))

    def nullable_pairing_history(self, path):
        original = self.batch(path)
        # Four actual environment trajectories supply multiple eligible pairs at
        # each turn. Copy identities, while reusing the same scored states.
        history = training.Proto(
            {key: torch.cat([value, value], dim=0) for key, value in original.batch.items()},
            {key: np.concatenate([value, value]) for key, value in original.non_tensor_batch.items()},
            original.meta_info,
        )
        for key in ("rollout_id_list", "traj_id_list", "traj_uid"):
            identities = original.non_tensor_batch[key].tolist()
            history.non_tensor_batch[key] = np.array(
                identities + [str(identity) + "_copy" for identity in identities], dtype=object)
        return history

    def nullable_pairing_trainer(self, cap, *, max_pairs=256):
        trainer = self.trainer()
        trainer.config.data.trace.update(k_pairs_per_group=cap, max_pairs_per_step=max_pairs)
        # All action pre-states match in V. Only the final post-states differ,
        # creating real signed local credit plus unchanged terminal filling.
        trainer.ref_policy_wg.scores.update({
            **{name + str(turn): -4. for name in ("A", "B") for turn in range(4)},
            "A4": -1., "B4": -8.,
        })
        return trainer

    def test_nullable_pair_quota_changes_only_pair_selection_in_both_entrypoints(self):
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            with self.subTest(path=path):
                limited = self.nullable_pairing_history(path)
                unlimited = self.nullable_pairing_history(path)
                limited_trainer = self.nullable_pairing_trainer(2)
                unlimited_trainer = self.nullable_pairing_trainer(None)
                before = {key: value.clone() for key, value in unlimited.batch.items()}
                original_turns = unlimited.non_tensor_batch["trace_original_turn"].copy()
                limited_trainer._annotate_trace_credits(limited)
                unlimited_trainer._annotate_trace_credits(unlimited)
                for key in ("trace_gt_logp", "trace_next_gt_logp", "trace_value", "trace_next_value",
                            "trace_local_credit", "trace_terminal_fill", "trace_credit"):
                    np.testing.assert_array_equal(limited.non_tensor_batch[key], unlimited.non_tensor_batch[key])
                same_turn_pairs = limited_trainer._trans_to_trace_batch([limited])
                counts = {}
                for pair in same_turn_pairs:
                    left = int(pair.non_tensor_batch["turn_index_list_a"][0])
                    right = int(pair.non_tensor_batch["turn_index_list_b"][0])
                    self.assertEqual(left, right)
                    counts[left] = counts.get(left, 0) + 1
                self.assertEqual(counts, {1: 2, 2: 2, 3: 2})
                cross_turn_pairs = unlimited_trainer._trans_to_trace_batch([unlimited])
                self.assertEqual(len(cross_turn_pairs), 6)  # One attempt per positive action.
                identities = [(pair.non_tensor_batch["rollout_id_list_a"][0],
                               pair.non_tensor_batch["turn_index_list_a"][0]) for pair in cross_turn_pairs]
                self.assertEqual(len(identities), len(set(identities)))
                # These symmetric trajectories rank corresponding turns alike,
                # while their baseline still sees the full cross-turn group.
                self.assertEqual({int(pair.non_tensor_batch["group_size"][0]) for pair in same_turn_pairs}, {4})
                self.assertEqual({int(pair.non_tensor_batch["group_size"][0]) for pair in cross_turn_pairs}, {16})
                self.assertEqual(unlimited_trainer._trace_pair_metrics["trace/pair/selected_unique"], 6)
                for pair in cross_turn_pairs:
                    self.assertEqual(pair.non_tensor_batch["trace_task_a"].tolist(), ["alfworld"])
                    self.assertEqual(pair.non_tensor_batch["trace_label_source_a"].tolist(), ["trace"])
                np.testing.assert_array_equal(unlimited.non_tensor_batch["trace_original_turn"], original_turns)
                for key, value in before.items():
                    self.assertTrue(torch.equal(unlimited.batch[key], value), key)

    def test_nullable_pair_quota_still_obeys_the_global_pair_limit(self):
        history = self.nullable_pairing_history("train-roma/rollout.py")
        trainer = self.nullable_pairing_trainer(None, max_pairs=5)
        trainer._annotate_trace_credits(history)
        pairs = trainer._trans_to_trace_batch([history])
        self.assertEqual(len(pairs), 5)
        self.assertEqual(trainer._trace_pair_metrics["trace/pair/selected_unique"], 5)
        self.assertGreater(trainer._trace_pair_metrics["trace/pair/rejected_global_cap"], 0)

    def test_single_action_episodes_score_terminal_post_state_without_extra_turn(self):
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            with self.subTest(path=path):
                original, trainer = self.batch(path), self.trainer()
                history = training.Proto(
                    {key: value[[0, 4]].clone() for key, value in original.batch.items()},
                    {key: value[[0, 4]].copy() for key, value in original.non_tensor_batch.items()},
                    original.meta_info,
                )
                history.non_tensor_batch["trace_is_terminal"][:] = True
                trainer._annotate_trace_credits(history)
                non = history.non_tensor_batch
                self.assertEqual(history.batch["responses"].shape[0], 2)
                self.assertTrue(np.isfinite(non["trace_credit"]).all())
                np.testing.assert_allclose(non["trace_value"], [0., 0.])
                np.testing.assert_allclose(non["trace_local_credit"], [np.log(4.1 / 3.1), np.log(4.1 / 5.1)])
                np.testing.assert_allclose(non["trace_terminal_fill"], [1., -1.])
                self.assertEqual(non["trace_label_source"].tolist(), ["trace", "trace"])
                pairs = trainer._trans_to_trace_batch([history])
                self.assertTrue(pairs)
                self.assertTrue(all(pair.non_tensor_batch["trace_is_terminal_a"][0] for pair in pairs))

    def test_missing_terminal_post_state_blocks_last_credit_and_all_terminal_fill(self):
        history, trainer = self.batch("train-roma/rollout.py"), self.trainer()
        history.non_tensor_batch["trace_next_state_context"][3] = None
        trainer._annotate_trace_credits(history)
        non = history.non_tensor_batch
        self.assertTrue(np.isnan(non["trace_next_gt_logp"][3]))
        self.assertTrue(np.isnan(non["trace_credit"][3]))
        self.assertFalse(non["trace_outcome_valid"][:4].any())
        self.assertTrue(np.isnan(non["trace_credit"][[1, 2, 3]]).all())
        self.assertTrue(np.isfinite(non["trace_credit"][0]))
        np.testing.assert_allclose(non["trace_terminal_fill"][0], 0.)

    def test_discontinuous_real_boundary_state_is_rejected_before_reference_scoring(self):
        history, trainer = self.batch("train-roma/rollout.py"), self.trainer()
        history.non_tensor_batch["trace_next_state_context"][0] = "different-game-or-state"
        with self.assertRaisesRegex(ValueError, "post-state and next pre-state disagree"):
            trainer._annotate_trace_credits(history)
        self.assertEqual(trainer.ref_policy_wg.calls, [])

    def test_missing_initial_turn_cannot_establish_a_new_environment_baseline(self):
        original, trainer = self.batch("train-roma/rollout.py"), self.trainer()
        kept = [1, 2, 3]
        history = training.Proto(
            {key: value[kept].clone() for key, value in original.batch.items()},
            {key: value[kept].copy() for key, value in original.non_tensor_batch.items()},
            original.meta_info,
        )
        trainer._annotate_trace_credits(history)
        self.assertTrue(np.isnan(history.non_tensor_batch["trace_value"]).all())
        self.assertTrue(np.isnan(history.non_tensor_batch["trace_next_value"]).all())
        self.assertTrue(np.isnan(history.non_tensor_batch["trace_credit"]).all())
        self.assertIsNone(trainer._trans_to_trace_batch([history]))

    def test_old_outcome_routed_environment_cache_is_rejected(self):
        history, trainer = self.batch("train-roma/rollout.py"), self.trainer()
        history.non_tensor_batch["trace_turn_type"][3] = "base"
        history.non_tensor_batch["trace_is_final"][3] = True
        with self.assertRaisesRegex(ValueError, "uniform environment-action metadata"):
            trainer._annotate_trace_credits(history)
        self.assertEqual(trainer.ref_policy_wg.calls, [])

    def test_external_failure_never_creates_a_verified_failure_outcome(self):
        logs = self.logs("A", external=1)
        self.assertFalse(any(log["trace_outcome_valid"] for log in logs))
        self.assertTrue(logs[1]["trace_external_error"])


class ValidationScopeTests(unittest.TestCase):
    def run_until_solve(self, path, *, task_name="alfworld", is_train=False):
        """Execute the real runner up to model solve, without distributed imports."""
        tree = ast.parse((states.ROOT / path).read_text())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.AsyncFunctionDef) and node.name == "_solve_and_evaluate")
        validation_env = NS(trace_enabled=True)
        training_env = NS(trace_enabled=True)
        bound_tasks = []
        validation_env.set_trace_task = bound_tasks.append
        rollout = NS(solver=NS(env=validation_env, executor=NS(tool_instances_cache={})),
                     solve=lambda **kwargs: None)
        snapshots = []

        class ReachedSolve(Exception):
            pass

        async def to_thread(function, **kwargs):
            snapshots.append((validation_env.trace_enabled, dict(kwargs)))
            raise ReachedSolve

        scope = {"Any": object, "AgentFlowRollout": object,
                 "asyncio": NS(to_thread=to_thread), "profiling_enabled": lambda: False,
                 "time": NS(perf_counter=lambda: 0)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec"), scope)
        owner = NS(task=task_name, training_agent=NS(solver=NS(env=training_env)))
        with self.assertRaises(ReachedSolve):
            asyncio.run(scope["_solve_and_evaluate"](
                owner, "rollout", rollout, {"question": "val_unseen_all_alfworld_0", "result": ""},
                0, val=not is_train, is_train=is_train))
        self.assertTrue(training_env.trace_enabled)
        self.assertEqual(len(snapshots), 1)
        return snapshots[0], bound_tasks

    def test_both_entrypoints_disable_only_alfworld_validation_trace_before_solve(self):
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            with self.subTest(path=path):
                (enabled, arguments), bound_tasks = self.run_until_solve(path)
                self.assertFalse(enabled)
                self.assertEqual(bound_tasks, [])
                self.assertEqual(arguments["question"], "val_unseen_all_alfworld_0")
                self.assertEqual(arguments["ground_truth"], "")

    def test_both_entrypoints_leave_training_trace_and_other_tasks_unchanged(self):
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            with self.subTest(path=path):
                (enabled, _), bound_tasks = self.run_until_solve(path, is_train=True)
                self.assertTrue(enabled)
                self.assertEqual(len(bound_tasks), 1)
                self.assertEqual(bound_tasks[0]["question"], "val_unseen_all_alfworld_0")
                (enabled, _), bound_tasks = self.run_until_solve(path, task_name="webshop")
                self.assertTrue(enabled)
                self.assertEqual(bound_tasks, [])


if __name__ == "__main__":
    unittest.main()
