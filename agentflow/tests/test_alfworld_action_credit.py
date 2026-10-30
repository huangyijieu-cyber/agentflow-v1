"""Every ALFWorld action has a real successor state and the same TRACE route."""
import importlib.util
import math
from pathlib import Path
import unittest

import test_trace_credit as shared


CREDIT = shared.trace_credit
SPEC = importlib.util.spec_from_file_location(
    "alfworld_action_execution_test", Path(__file__).resolve().parents[1] / "agentflow/trace/execution.py")
EXECUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXECUTION)


class EnvironmentCreditTests(unittest.TestCase):
    def annotate(self, values=None, next_values=None, indices=None, errors=None,
                 terminals=None, outcome_valid=True, advantage=.5, config=None):
        values = [0., 1., 3., 6.] if values is None else values
        next_values = [1., 3., 6., 10.] if next_values is None else next_values
        size = len(values)
        return CREDIT.annotate_environment_credits(
            values, next_values, list(range(size)) if indices is None else indices,
            [False] * size if errors is None else errors,
            [False] * (size - 1) + [True] if terminals is None else terminals,
            outcome_valid=outcome_valid, outcome_advantage=advantage, config=config)

    def test_single_terminal_action_uses_its_post_state_and_distance_zero_fill(self):
        result = self.annotate(values=[0.], next_values=[2.])
        self.assertEqual(result["local_credit"], [2.])
        self.assertEqual(result["terminal_fill"], [1.])
        self.assertEqual(result["effective_credit"], [3.])
        self.assertEqual(result["terminal_reached"], [True])
        self.assertEqual(len(result["effective_credit"]), 1)

    def test_k_three_uses_real_terminal_delta_and_fills_last_three_actions(self):
        result = self.annotate()
        self.assertAlmostEqual(result["local_credit"][0], (1 + .8 * 2 + .64 * 3) / 2.44)
        self.assertAlmostEqual(result["local_credit"][1], (2 + .8 * 3 + .64 * 4) / 2.44)
        self.assertAlmostEqual(result["local_credit"][2], (3 + .8 * 4) / 1.8)
        self.assertEqual(result["local_credit"][3], 4)
        for actual, expected in zip(result["terminal_fill"], [0., .64, .8, 1.]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["terminal_reached"], [False, True, True, True])

    def test_terminal_reward_does_not_replace_missing_successor(self):
        result = self.annotate(next_values=[1., 3., 6., None])
        self.assertTrue(math.isfinite(result["effective_credit"][0]))
        for index in (1, 2, 3):
            self.assertTrue(math.isnan(result["effective_credit"][index]))
            self.assertTrue(result["window_blocked"][index])

    def test_missing_pre_state_blocks_every_window_that_needs_its_delta(self):
        result = self.annotate(values=[0., 1., None, 6.])
        for index in (0, 1, 2):
            self.assertTrue(math.isnan(result["effective_credit"][index]))
        self.assertEqual(result["effective_credit"][3], 5.)

    def test_external_failure_blocks_only_windows_containing_that_action(self):
        result = self.annotate(errors=[False, True, False, False])
        self.assertEqual(result["external_error_blocked"], [True, True, False, False])
        self.assertTrue(all(math.isnan(result["effective_credit"][i]) for i in (0, 1)))
        self.assertTrue(all(math.isfinite(result["effective_credit"][i]) for i in (2, 3)))

    def test_turn_gaps_are_not_joined_but_one_step_can_use_explicit_successor(self):
        result = self.annotate(indices=[0, 1, 3, 4])
        self.assertTrue(math.isnan(result["effective_credit"][0]))
        self.assertTrue(math.isnan(result["effective_credit"][1]))
        self.assertTrue(math.isfinite(result["effective_credit"][2]))
        one = self.annotate(indices=[0, 1, 3, 4], config={"k": 1})
        self.assertEqual(one["local_credit"], [1., 2., 3., 4.])

    def test_terminal_closes_window_after_its_delta_and_excludes_later_actions(self):
        result = self.annotate(terminals=[False, True, False, False])
        self.assertAlmostEqual(result["local_credit"][0], (1 + .8 * 2) / 1.8)
        self.assertEqual(result["local_credit"][1], 2.)
        self.assertEqual(result["terminal_reached"], [True, True, False, False])
        self.assertTrue(all(math.isnan(result["effective_credit"][i]) for i in (2, 3)))

    def test_missing_or_untrusted_terminal_keeps_local_credit_without_fill(self):
        for options in ({"terminals": [False] * 4}, {"outcome_valid": False, "advantage": None},
                        {"config": {"terminal_fill_enabled": False}, "advantage": None}):
            with self.subTest(options=options):
                result = self.annotate(**options)
                self.assertEqual(result["terminal_fill"], [0.] * 4)
                self.assertEqual(result["effective_credit"][-1], 4.)

    def test_multiple_terminals_and_invalid_original_turn_fail_closed(self):
        result = self.annotate(terminals=[False, True, False, True])
        self.assertTrue(all(math.isnan(value) for value in result["effective_credit"]))
        for bad in (None, -1, True, 1.5):
            result = self.annotate(indices=[bad, 1, 2, 3])
            self.assertTrue(math.isnan(result["effective_credit"][0]))

    def test_signed_fill_and_zero_discount(self):
        result = self.annotate(values=[0.], next_values=[.1], advantage=-.5)
        self.assertAlmostEqual(result["effective_credit"][0], -.9)
        zero = self.annotate(config={"gamma": 0})
        self.assertEqual(zero["local_credit"], [1., 2., 3., 4.])
        self.assertEqual(zero["terminal_fill"], [0., 0., 0., 1.])

    def test_input_lengths_are_validated(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            self.annotate(next_values=[1.])
        self.assertEqual(self.annotate(values=[], next_values=[], terminals=[])["effective_credit"], [])

    def test_first_and_terminal_action_share_type_but_cannot_cross_turn_ids(self):
        first = shared.row(0, credit=.2, turn_type="tool_plan", turn_index=0,
                           label_source="trace", is_final=False, is_terminal=False)
        terminal = shared.row(1, credit=-.2, turn_type="tool_plan", turn_index=7,
                              label_source="trace", is_final=False, is_terminal=True)
        self.assertEqual(CREDIT.select_pairs([first, terminal]), [])
        # Terminal status does not create a separate route: actions from
        # different rollouts can still pair when their original turns agree.
        first["turn_index"] = terminal["turn_index"]
        self.assertEqual(CREDIT.select_pairs([first, terminal]), [(0, 1)])


class UniformEnvironmentRouteTests(unittest.TestCase):
    def action(self, index, *, terminal=False, success=False, finish="stop", external=False,
               post_context=True, goal_valid=True):
        log = {"response": f"<action>act {index}</action>", "finish_reason": finish,
               "trace_scoring_valid": True, "trace_goal_check_valid_before": True}
        info = {"trace_env_action": f"act {index}", "terminated": terminal,
                "truncated": False, "won": success, "env_error": external,
                "trace_goal_check": success, "trace_goal_check_valid": goal_valid,
                "trace_scoring_valid": goal_valid}
        if post_context:
            info["trace_state_context"] = f"private state {index + 1}"
        EXECUTION.annotate_alfworld_step(log, f"private state {index}", terminal, info)
        return log

    def annotate(self, logs, reward=1.):
        return EXECUTION.annotate_trace_outcome(
            logs, {"base_response": logs[-1]["response"]}, reward, None, "alfworld")

    def test_all_actions_share_trace_type_and_terminal_is_only_a_boundary(self):
        logs = [self.action(0), self.action(1, terminal=True, success=True)]
        result = self.annotate(logs)
        self.assertEqual([item["trace_turn_type"] for item in logs], ["tool_plan"] * 2)
        self.assertEqual([item["trace_is_final"] for item in logs], [False] * 2)
        self.assertEqual([item["trace_is_terminal"] for item in logs], [False, True])
        self.assertEqual([item["trace_next_state_context"] for item in logs],
                         ["private state 1", "private state 2"])
        self.assertTrue(result["trace_outcome_valid"])
        self.assertIsNone(result["trace_final_turn"])
        self.assertEqual(result["trace_terminal_turn"], 1)

    def test_missing_or_invalid_post_state_cannot_supply_terminal_fill(self):
        for options in ({"post_context": False}, {"goal_valid": False}):
            with self.subTest(options=options):
                logs = [self.action(0, terminal=True, success=True, **options)]
                self.assertFalse(logs[0]["trace_next_scoring_valid"])
                result = self.annotate(logs)
                self.assertFalse(result["trace_outcome_valid"])
                self.assertTrue(logs[0]["trace_is_terminal"])
                self.assertFalse(logs[0]["trace_is_final"])

    def test_truncated_model_output_or_external_crash_is_not_trusted_completion(self):
        for options in ({"finish": "length"}, {"external": True}):
            with self.subTest(options=options):
                logs = [self.action(0, terminal=True, success=False, **options)]
                result = self.annotate(logs, reward=0.)
                self.assertFalse(result["trace_outcome_valid"])
                self.assertFalse(logs[0]["trace_is_terminal"])
                self.assertFalse(logs[0]["trace_is_final"])
                self.assertEqual(logs[0]["trace_turn_type"], "tool_plan")


if __name__ == "__main__":
    unittest.main()
