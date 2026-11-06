"""Real rollout encoding, daemon and trainer with CPU environment/reference stubs."""
import copy
import re
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

import test_trace_states as states
import test_trace_trainer as training


GT = "The current candidate product meets the required attributes."
INSTRUCTION = "Describe the current shopping facts relative to the task requirements."
PROVENANCE = {"goal_sha256": "a" * 64, "manifest_sha256": "b" * 64,
              "task_key": "frozen-corpus:goal:6684", "template_version": "webshop-candidate-state-v1"}


class WebshopPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        states.DaemonMetadataTests.setUpClass()
        cls.daemon_builder = states.DaemonMetadataTests()

    def logs(self, name, reward):
        logs = []
        for index, action in enumerate(("search[bottle]", "click[item]", "click[blue]", "click[buy now]")):
            log = {"prompt": f"original actor {name}{index}", "response": f"<action>{action}</action>",
                   "prompt_token_ids": [10, index + 11], "response_token_ids": [100 + index],
                   "finish_reason": "stop", "trace_original_turn": index,
                   "trace_gt": GT, "trace_answer_instruction": INSTRUCTION,
                   "trace_goal_metadata": copy.deepcopy(PROVENANCE), "trace_state_view": "webshop_state",
                   "trace_goal_check_valid_before": True, "trace_scoring_valid": True}
            terminal = index == 3
            training.EXECUTION.annotate_webshop_step(log, name + str(index), terminal, {
                "trace_state_context": name + str(index + 1), "trace_env_action": action,
                "terminated": terminal, "truncated": False, "won": terminal and reward == 1.,
                "task_score": reward if terminal else 0.,
                "trace_goal_check_valid": True, "trace_scoring_valid": True,
            })
            logs.append(log)
        # This deliberately supplies the unchanged binary GRPO reward.
        training.EXECUTION.annotate_trace_outcome(
            logs, {"base_response": logs[-1]["response"]}, float(reward == 1.), None, "webshop")
        return logs

    def encode(self, logs, path):
        encoder = states.extract(path, "encode_logs", {"Triplet": NS})
        return encoder(None, logs) if "profiled" in path else encoder(logs)

    def batch(self, path="train-roma/rollout.py", rewards=(.8, .2)):
        completed, originals = {}, {}
        flags = states.extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        for name, reward in zip(("A", "B"), rewards):
            logs = self.logs(name, reward)
            completed[name] = NS(triplets=self.encode(logs, path),
                                 metadata={"turn_flags": flags(logs, "webshop")}, reward=float(reward == 1.))
            originals[name] = {"data_id": "same-question", "result": GT}
        daemon = NS(is_train=True, _completed_rollouts=completed, _total_tasks_queued=2,
                    pad_token_id=0, _task_id_to_original_sample=originals,
                    _fillna_reward=lambda rollout: rollout.reward,
                    _get_format_metrics=lambda prefix: {})
        batch, _ = self.daemon_builder.get_batch(daemon, 4, 4, "cpu")
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"].copy()
        batch.non_tensor_batch["traj_uid"] = batch.non_tensor_batch["traj_id_list"].copy()
        batch.batch["response_mask"] = batch.batch["attention_mask"][:, -4:].clone()
        batch.batch["old_log_probs"] = torch.full_like(batch.batch["responses"], -1., dtype=torch.float32)
        batch.meta_info = {"temperature": .7}
        return batch

    def trainer(self):
        scores = {"A0": -4., "A1": -3., "A2": -2., "A3": -1., "A4": -1.,
                  "B0": -4., "B1": -5., "B2": -6., "B3": -7., "B4": -7.}
        trainer = training.make_trainer(scores, {"k": 3, "gamma": .8, "epsilon": .1,
                                               "terminal_fill_enabled": True,
                                               "k_pairs_per_group": None, "pair_round_robin": False})
        trainer.config.data["task"] = "webshop"
        return trainer

    def test_both_entrypoints_preserve_continuous_reward_and_terminal_post_state(self):
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            with self.subTest(path=path):
                batch, trainer = self.batch(path), self.trainer()
                before = {key: value.clone() for key, value in batch.batch.items()}
                metrics = trainer._annotate_trace_credits(batch)
                non = batch.non_tensor_batch
                np.testing.assert_allclose(non["trace_outcome_reward"], [.8] * 4 + [.2] * 4)
                np.testing.assert_allclose(non["trace_outcome_advantage"], [.3] * 4 + [-.3] * 4)
                self.assertTrue(non["trace_outcome_valid"].all())
                self.assertFalse(non["trace_outcome_success"].any())  # .8 is not full success.
                self.assertEqual(non["trace_is_terminal"].tolist(), [False, False, False, True] * 2)
                self.assertFalse(non["trace_is_final"].any())
                self.assertEqual(metrics["trace/scoring_requests"], 10)
                self.assertEqual(batch.batch["responses"].shape[0], 8)
                np.testing.assert_allclose(non["trace_next_value"][:3], non["trace_value"][1:4])
                np.testing.assert_allclose(non["trace_local_credit"][[3, 7]], [0., 0.])
                positive = max(non["trace_local_credit"])
                negative = min(non["trace_local_credit"])
                np.testing.assert_allclose(non["trace_terminal_fill"],
                    np.r_[np.array([0., .64, .8, 1.]) * positive,
                          np.array([0., .64, .8, 1.]) * negative])
                self.assertGreater(non["trace_credit"][3], 0.)
                self.assertLess(non["trace_credit"][7], 0.)
                for key, value in before.items():
                    self.assertTrue(torch.equal(batch.batch[key], value), key)
                pairs = trainer._trans_to_trace_batch([batch])
                self.assertTrue(pairs)
                for pair in pairs:
                    self.assertEqual(pair.non_tensor_batch["trace_task_a"].tolist(), ["webshop"])
                    self.assertEqual(pair.non_tensor_batch["trace_gt_a"].tolist(), [GT])
                    self.assertFalse(any(key.startswith("trace_") for key in trainer._actor_input_view(pair).non_tensor_batch))

    def test_wrong_goal_manifest_template_or_gt_within_uid_fails_before_scoring(self):
        for key in ("goal_sha256", "manifest_sha256", "task_key", "template_version", "trace_gt"):
            with self.subTest(key=key):
                batch, trainer = self.batch(), self.trainer()
                if key == "trace_gt":
                    batch.non_tensor_batch[key][4] = "Another fixed target"
                else:
                    batch.non_tensor_batch["trace_goal_metadata"][4][key] = "different"
                with self.assertRaises(ValueError):
                    trainer._annotate_trace_credits(batch)
                self.assertEqual(trainer.ref_policy_wg.calls, [])

    def test_stale_metadata_and_missing_state_view_fail_closed(self):
        for key, value in (("trace_task", None), ("trace_state_view", "full_state")):
            batch, trainer = self.batch(), self.trainer()
            batch.non_tensor_batch[key][0] = value
            with self.assertRaises(ValueError):
                trainer._annotate_trace_credits(batch)

    def test_broken_boundary_rejected_and_missing_post_state_blocks_credit(self):
        batch, trainer = self.batch(), self.trainer()
        batch.non_tensor_batch["trace_next_state_context"][0] = "unrelated-state"
        with self.assertRaisesRegex(ValueError, "post-state and next pre-state"):
            trainer._annotate_trace_credits(batch)
        batch, trainer = self.batch(), self.trainer()
        batch.non_tensor_batch["trace_next_scoring_valid"][3] = False
        trainer._annotate_trace_credits(batch)
        self.assertFalse(batch.non_tensor_batch["trace_outcome_valid"][:4].any())
        self.assertTrue(np.isnan(batch.non_tensor_batch["trace_credit"][1:4]).all())

    def test_partial_reward_cannot_use_boolean_success_threshold(self):
        for reward, success in ((.8, True), (1., False), (.2, None), (1.2, True)):
            batch, trainer = self.batch(), self.trainer()
            batch.non_tensor_batch["trace_outcome_reward"][:4] = reward
            batch.non_tensor_batch["trace_outcome_success"][:4] = success
            trainer._annotate_trace_credits(batch)
            self.assertFalse(batch.non_tensor_batch["trace_outcome_valid"][:4].any())

    def test_parquet_gt_mismatch_is_rejected_by_daemon(self):
        for gt in ("", "wrong goal"):
            with self.subTest(gt=gt), self.assertRaises(ValueError):
                self.daemon_builder.build_batch(self.encode(self.logs("A", .8), "train-roma/rollout.py"), ground_truth=gt)

    def test_same_buy_action_at_different_states_is_not_removed(self):
        flags = states.extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        logs = [self.logs("A", .8)[-1], self.logs("B", .2)[-1]]
        logs.append(copy.deepcopy(logs[0]))
        result = flags(logs, "webshop")
        self.assertEqual([row["trace_dup"] for row in result], [False, False, True])


if __name__ == "__main__":
    unittest.main()
