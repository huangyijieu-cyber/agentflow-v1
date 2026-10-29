"""ALFWorld metadata contracts across real rollout, daemon and TRACE trainer code.

The simulator/frozen reference are CPU stubs; the production adapter methods
are extracted without importing distributed or ALFWorld runtime dependencies.
"""
import copy
import re
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

import test_trace_states as states
import test_trace_trainer as training


EXECUTION = training.EXECUTION
GOAL = "A clean mug is in a cabinet."
INSTRUCTION = "Describe the current full state relative to the success conditions."
PROVENANCE = {"goal_spec_hash": "actual-game-v1", "source_sha256": {"gamefile": "official-game"}}


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
            originals[name] = {"data_id": "same-question", "result": ""}  # ALFWorld parquet has no GT.
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
        scores = {"A0": -4., "A1": -3., "A2": -2., "A3": -1.,
                  "B0": -4., "B1": -5., "B2": -6., "B3": -7.}
        trainer = training.make_trainer(scores, {"k": 3, "gamma": .8, "epsilon": .1,
                                               "terminal_fill_enabled": True, "terminal_fill_scale": 2.})
        trainer.config.data.task = "alfworld"
        return trainer

    def test_both_entrypoints_use_fixed_gt_full_state_and_qa_credit_routing(self):
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
                trainer._annotate_trace_credits(history)
                self.assertEqual(non["trace_label_source"].tolist(), ["trace"] * 3 + ["outcome"] + ["trace"] * 3 + ["outcome"])
                np.testing.assert_allclose(non["trace_outcome_advantage"], [.5] * 4 + [-.5] * 4)
                cfg = training.CREDIT.TraceConfig(k=3, gamma=.8, epsilon=.1)
                for start in (0, 4):
                    expected = training.CREDIT.annotate_rollout_credits(
                        non["trace_value"][start:start + 4], [0, 1, 2, 3],
                        ["tool_plan"] * 3 + ["base"], [False] * 4, [False] * 3 + [True],
                        outcome_valid=True, outcome_advantage=.5 if start == 0 else -.5, config=cfg)
                    np.testing.assert_allclose(non["trace_credit"][start:start + 4], expected["effective_credit"], equal_nan=True)
                self.assertEqual(non["trace_terminal_reached"].tolist(), [False, True, True, False] * 2)
                for key, value in before.items():
                    self.assertTrue(torch.equal(history.batch[key], value), key)
                for call in trainer.ref_policy_wg.calls:
                    for row in range(len(call)):
                        prompt = trainer.tokenizer.decode(call.batch["prompts"][row])
                        self.assertIn(INSTRUCTION, prompt)
                        self.assertNotIn("actor visible", prompt)
                pairs = trainer._trans_to_trace_batch([history])
                self.assertTrue(pairs)
                self.assertTrue(any(pair.non_tensor_batch["trace_label_source_a"][0] == "outcome" for pair in pairs))
                for pair in pairs:
                    self.assertEqual(pair.non_tensor_batch["trace_gt_a"].tolist(), [GOAL])
                    self.assertEqual(pair.non_tensor_batch["trace_task_a"].tolist(), ["alfworld"])

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
        # Actual terminal identity remains base/outcome even when untrusted.
        self.assertTrue(non["trace_is_final"][3])
        self.assertEqual(non["trace_turn_type"][3], "base")

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

    def test_missing_actual_gt_cannot_fall_back_to_parquet(self):
        logs = self.logs("A")
        logs[0]["trace_gt"] = None
        with self.assertRaisesRegex(ValueError, "deterministic goal response"):
            self.daemon_builder.build_batch(self.encode(logs, "train-roma/rollout.py"), ground_truth="guessed answer")

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

    def test_external_failure_never_creates_a_verified_failure_outcome(self):
        logs = self.logs("A", external=1)
        self.assertFalse(any(log["trace_outcome_valid"] for log in logs))
        self.assertTrue(logs[1]["trace_external_error"])


if __name__ == "__main__":
    unittest.main()
