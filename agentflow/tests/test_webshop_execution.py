"""Native WebShop outcomes remain continuous and require real state boundaries."""
import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "webshop_execution_test", ROOT / "agentflow/agentflow/trace/execution.py")
execution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(execution)


class WebShopExecutionTests(unittest.TestCase):
    def step(self, index=0, *, done=False, score=0., action="search[bottle]", **changes):
        log = {"response": f"<action>{action}</action>", "finish_reason": "stop",
               "trace_goal_check_valid_before": True, "trace_scoring_valid": True}
        info = {"trace_state_context": f"state {index + 1}", "trace_env_action": action,
                "trace_goal_check_valid": True, "trace_scoring_valid": True,
                "terminated": done, "won": score == 1., "task_score": score}
        info.update(changes)
        execution.annotate_webshop_step(log, f"state {index}", done, info)
        return log

    def logs(self, score=.75):
        return [self.step(), self.step(1, done=True, score=score, action="click[buy now]")]

    def outcome(self, logs, binary_reward=0.):
        return execution.annotate_trace_outcome(
            logs, {"base_response": logs[-1]["response"]}, binary_reward, None, "webshop")

    def test_partial_native_reward_survives_binary_grpo_reward(self):
        logs = self.logs(.75)
        result = self.outcome(logs, 0.)
        self.assertTrue(result["trace_outcome_valid"])
        self.assertFalse(result["trace_outcome_success"])
        self.assertEqual([log["trace_outcome_reward"] for log in logs], [.75, .75])
        self.assertEqual([log["trace_turn_type"] for log in logs], ["tool_plan"] * 2)
        self.assertEqual([log["trace_is_final"] for log in logs], [False, False])
        self.assertEqual([log["trace_is_terminal"] for log in logs], [False, True])
        self.assertEqual(logs[-1]["trace_next_state_context"], "state 2")

    def test_full_success_uses_only_exact_native_one(self):
        for score, success in ((0., False), (.999, False), (1., True)):
            logs = self.logs(score)
            self.assertEqual(self.outcome(logs, 1.)["trace_outcome_success"], success)

    def test_verified_budget_zero_is_a_terminal_outcome(self):
        logs = [self.step(done=True, terminated=False, truncated_by_max_step=True)]
        self.assertTrue(self.outcome(logs)["trace_outcome_valid"])
        self.assertEqual(logs[0]["trace_outcome_reward"], 0.)
        for score in (.2, None, float("nan"), float("inf"), -1., 1.1):
            logs = [self.step(done=True, score=score, terminated=False, truncated_by_max_step=True)]
            self.assertFalse(self.outcome(logs)["trace_outcome_valid"])

    def test_unverified_stop_or_nonpurchase_termination_has_no_outcome(self):
        for fields in ({}, {"terminated": False, "truncated": True}):
            logs = [self.step(done=True, **fields)]
            self.assertFalse(self.outcome(logs)["trace_outcome_valid"])

    def test_missing_binding_state_external_error_or_cutoff_blocks_fill(self):
        changes = (("trace_goal_check_valid", False), ("trace_goal_check_valid_before", False),
                   ("trace_scoring_valid", False), ("trace_next_scoring_valid", False),
                   ("trace_external_error", True), ("finish_reason", "length"))
        for field, value in changes:
            for index in (0, 1):
                with self.subTest(field=field, index=index):
                    logs = self.logs()
                    logs[index][field] = value
                    self.assertFalse(self.outcome(logs)["trace_outcome_valid"])
                    self.assertTrue(all(log["trace_outcome_reward"] is None for log in logs))

    def test_invalid_noop_keeps_state_transition_without_external_failure(self):
        log = self.step(format_error=True, invalid_action=True)
        self.assertTrue(log["trace_scoring_valid"])
        self.assertTrue(log["trace_next_scoring_valid"])
        self.assertFalse(log["extraction_valid"])
        self.assertTrue(log["trace_tool_failed"])
        self.assertFalse(log["trace_external_error"])
        self.assertEqual(log["trace_next_state_context"], "state 1")

    def test_state_and_action_both_contribute_to_duplicate_identity(self):
        a = self.step(action="click[buy now]")
        b = self.step(1, action="click[buy now]")
        self.assertNotEqual(a["trace_action_key"], b["trace_action_key"])
        self.assertEqual(a["trace_action_key"], self.step(action=" CLICK[BUY NOW] ")["trace_action_key"])

    def test_only_the_attributable_prefix_receives_outcome_labels(self):
        logs = self.logs()
        output = logs[-1]["response"]
        logs.append(self.step(2, action="search[later]"))
        result = execution.annotate_trace_outcome(
            logs, {"base_response": output}, 0., None, "webshop")
        self.assertTrue(result["trace_outcome_valid"])
        self.assertEqual([item["trace_outcome_valid"] for item in logs], [True, True, False])
        self.assertIsNone(logs[-1]["trace_outcome_reward"])
        duplicate = copy.deepcopy(logs[:2])
        duplicate.append(copy.deepcopy(duplicate[-1]))
        self.assertFalse(self.outcome(duplicate)["trace_outcome_valid"])


if __name__ == "__main__":
    unittest.main()
