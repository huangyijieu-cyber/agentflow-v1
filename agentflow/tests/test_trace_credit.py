"""CPU-only tests for TRACE credit and strict SLIC selection."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "agentflow" / "trace" / "credit.py"
SPEC = importlib.util.spec_from_file_location("trace_credit", MODULE_PATH)
trace_credit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trace_credit
SPEC.loader.exec_module(trace_credit)

TraceConfig = trace_credit.TraceConfig
state_values = trace_credit.state_values
discounted_credits = trace_credit.discounted_credits
annotate_rollout_credits = trace_credit.annotate_rollout_credits
select_pairs = trace_credit.select_pairs


def row(index, credit=0.1, **kwargs):
    result = dict(index=index, uid="q", turn_type="tool", rollout_id="r",
                  turn_index=index, value=0.2, credit=credit, valid=True,
                  length=10, old_per_token=-1.0)
    result.update(kwargs)
    return result


class TraceValueTests(unittest.TestCase):
    def test_value_sign_and_adjacent_credit(self):
        values = state_values([-4.0, -2.0, -8.0])
        self.assertEqual(values[0], 0)
        self.assertAlmostEqual(values[1], math.log((4.0 + 0.1) / (2.0 + 0.1)))
        self.assertLess(values[2], 0)
        credit = discounted_credits(values)
        self.assertAlmostEqual(credit[0], values[1] - values[0])
        self.assertAlmostEqual(credit[1], values[2] - values[1])
        self.assertTrue(math.isnan(credit[2]))

    def test_training_gap_offset_limits_near_zero_nll_credit(self):
        # With the paper's training offset, NLL 0.1 -> 0 closes half
        # the offset-adjusted gap, rather than producing unbounded credit.
        credit = discounted_credits(state_values([-0.1, 0.0]))
        self.assertAlmostEqual(credit[0], math.log(2.0))
        diagnostic_value = state_values([-0.1, 0.0], epsilon=1e-3)[1]
        self.assertAlmostEqual(diagnostic_value, math.log(101.0))
        self.assertLess(credit[0], diagnostic_value)

    def test_missing_baseline_invalidates_entire_rollout(self):
        for baseline in (None, math.nan, math.inf, 0.1):
            self.assertTrue(all(math.isnan(value) for value in state_values([baseline, -2.0])))

    def test_missing_observation_keeps_original_adjacency(self):
        values = state_values([-4.0, None, -2.0, -1.0])
        credit = discounted_credits(values)
        self.assertTrue(math.isnan(credit[0]))
        self.assertTrue(math.isnan(credit[1]))
        self.assertAlmostEqual(credit[2], values[3] - values[2])

    def test_turn_gaps_and_duplicates_are_not_joined(self):
        credit = discounted_credits([0, 1, 3, 6], [0, 2, 2, 3])
        self.assertTrue(math.isnan(credit[0]))
        self.assertTrue(math.isnan(credit[1]))
        self.assertEqual(credit[2], 3)

    def test_discount_and_tail_weight_normalization(self):
        credit = discounted_credits([0, 1, 4, 10], k=2, gamma=0.5)
        self.assertAlmostEqual(credit[0], (1 + 0.5 * 3) / 1.5)
        self.assertAlmostEqual(credit[1], (3 + 0.5 * 6) / 1.5)
        self.assertEqual(credit[2], 6)
        self.assertTrue(math.isnan(credit[3]))

    def test_missing_value_invalidates_whole_requested_window(self):
        credit = discounted_credits([0, 1, None, 4], k=2)
        self.assertTrue(all(math.isnan(value) for value in credit))

    def test_zero_discount_uses_only_immediate_delta(self):
        credit = discounted_credits([0, 1, None], k=2, gamma=0)
        self.assertEqual(credit[0], 1)
        self.assertTrue(math.isnan(credit[1]))

    def test_empty_and_singleton_inputs(self):
        self.assertEqual(state_values([]), [])
        self.assertEqual(discounted_credits([]), [])
        self.assertTrue(math.isnan(discounted_credits([0])[0]))


class RoutedCreditTests(unittest.TestCase):
    def annotate(self, *, values=None, indices=None, types=None, errors=None,
                 finals=None, outcome_valid=True, advantage=0.5, config=None):
        values = [0, 0.1, 0.2, 0.3, 0.4, 0.5] if values is None else values
        size = len(values)
        return annotate_rollout_credits(
            values, list(range(size)) if indices is None else indices,
            ["analysis", "tool_plan", "tool_plan", "tool_plan", "tool_plan", "final_answer"]
            if types is None else types,
            [False] * size if errors is None else errors,
            [False] * (size - 1) + [True] if finals is None else finals,
            outcome_valid=outcome_valid, outcome_advantage=advantage, config=config)

    def test_k_three_contains_final_and_only_last_two_tools_receive_fill(self):
        result = self.annotate()
        self.assertTrue(math.isnan(result["effective_credit"][0]))
        self.assertTrue(math.isnan(result["effective_credit"][-1]))
        for position in range(1, 5):
            self.assertAlmostEqual(result["local_credit"][position], 0.1)
        self.assertEqual(result["terminal_fill"][1:3], [0, 0])
        self.assertAlmostEqual(result["terminal_fill"][3], 2 * 0.8 ** 2 * 0.5)
        self.assertAlmostEqual(result["terminal_fill"][4], 2 * 0.8 * 0.5)
        self.assertEqual(result["terminal_reached"], [False, False, False, True, True, False])

    def test_fill_uses_signed_advantage_and_can_reverse_effective_credit(self):
        result = self.annotate(advantage=-0.5)
        self.assertAlmostEqual(result["effective_credit"][4], 0.1 - 0.8)
        self.assertLess(result["effective_credit"][4], 0)
        self.assertTrue(math.isnan(result["effective_credit"][0]))

    def test_external_failure_invalidates_all_windows_containing_transition(self):
        result = self.annotate(errors=[False, False, False, True, False, False])
        for position in (1, 2, 3):
            self.assertTrue(math.isnan(result["effective_credit"][position]))
            self.assertTrue(result["external_error_blocked"][position])
        self.assertAlmostEqual(result["effective_credit"][4], 0.9)

    def test_one_step_does_not_include_next_actions_failure_or_final_decision(self):
        result = self.annotate(errors=[False, False, True, False, False, False], config={"k": 1})
        self.assertAlmostEqual(result["effective_credit"][1], 0.1)
        self.assertTrue(math.isnan(result["effective_credit"][2]))
        self.assertEqual(result["terminal_fill"][4], 0)

    def test_no_final_marker_means_no_terminal_fill(self):
        result = self.annotate(values=[0, 0.1, 0.2],
                               types=["analysis", "tool_plan", "tool_plan"], finals=[False] * 3)
        self.assertAlmostEqual(result["effective_credit"][1], 0.1)
        self.assertEqual(result["terminal_fill"][1], 0)
        self.assertTrue(math.isnan(result["effective_credit"][2]))

    def test_untrusted_outcome_or_disabled_fill_preserves_only_local_credit(self):
        for kwargs in ({"outcome_valid": False, "advantage": None},
                       {"config": {"terminal_fill_enabled": False}, "advantage": None},
                       {"config": {"terminal_fill_scale": 0}, "advantage": None}):
            with self.subTest(kwargs=kwargs):
                result = self.annotate(**kwargs)
                self.assertAlmostEqual(result["effective_credit"][4], 0.1)
                self.assertEqual(result["terminal_fill"][4], 0)

    def test_missing_gt_cannot_be_replaced_with_terminal_advantage(self):
        result = self.annotate(values=[0, 0.1, 0.2, 0.3, 0.4, math.nan])
        self.assertTrue(math.isnan(result["effective_credit"][4]))
        self.assertTrue(result["window_blocked"][4])

    def test_missing_original_turn_is_not_bridged(self):
        result = self.annotate(values=[0, 0.1, 0.3, 0.4], indices=[0, 1, 3, 4],
                               types=["analysis", "tool_plan", "tool_plan", "final_answer"])
        self.assertTrue(math.isnan(result["effective_credit"][1]))
        self.assertAlmostEqual(result["effective_credit"][2], 0.9)

    def test_decisions_after_actual_final_do_not_receive_credit(self):
        result = self.annotate(values=[0, 0.1, 0.2, 0.3],
                               types=["analysis", "tool_plan", "direct", "tool_plan"],
                               finals=[False, False, True, False])
        self.assertAlmostEqual(result["effective_credit"][1], 0.9)
        self.assertTrue(math.isnan(result["effective_credit"][3]))

    def test_multiple_terminal_markers_fail_closed(self):
        result = self.annotate(finals=[False, False, False, False, True, True])
        self.assertTrue(all(math.isnan(value) for value in result["effective_credit"]))


class TracePairTests(unittest.TestCase):
    def test_same_question_type_pairs_across_turns_and_rollouts(self):
        rows = [row(0, rollout_id="a", turn_index=1),
                row(1, -0.1, rollout_id="b", turn_index=7)]
        self.assertEqual(select_pairs(rows), [(0, 1)])
        rows[1]["rollout_id"] = "a"
        self.assertEqual(select_pairs(rows), [(0, 1)])

    def test_different_question_or_type_cannot_pair(self):
        self.assertEqual(select_pairs([row(0), row(1, -0.1, uid="other")]), [])
        self.assertEqual(select_pairs([row(0), row(1, -0.1, turn_type="answer")]), [])

    def test_strict_credit_thresholds_and_no_reward_fallback(self):
        self.assertEqual(select_pairs([row(0, 0.01, reward=1), row(1, -0.1, reward=0)]), [])
        self.assertEqual(select_pairs([row(0, 0.1), row(1, -0.01)]), [])
        self.assertEqual(select_pairs([row(0, 0.1, reward=1), row(1, 0, reward=0)]), [])
        self.assertEqual(select_pairs([row(0, reward=0), row(1, -0.1, reward=1)]), [(0, 1)])

    def test_value_and_credit_gap_limits(self):
        self.assertEqual(select_pairs([row(0), row(1, -0.1, value=0.31)]), [])
        self.assertEqual(select_pairs([row(0), row(1, -0.1)], {"gap_epsilon": 0.21}), [])
        self.assertEqual(select_pairs([row(0), row(1, -0.1)], {"gap_epsilon": 0.2}), [(0, 1)])

    def test_duplicate_actions_empty_and_invalid_responses(self):
        self.assertEqual(select_pairs([row(0, action_key="same"), row(1, -0.1, action_key="same")]), [])
        for overrides in ({"valid": False}, {"length": 0}, {"value": math.nan},
                          {"credit": math.nan}, {"old_per_token": math.inf}):
            self.assertEqual(select_pairs([row(0, **overrides), row(1, -0.1)]), [])

    def test_length_filter_and_absence_of_old_logp_margin_filter(self):
        self.assertEqual(select_pairs([row(0, length=19), row(1, -0.1, length=10)]), [])
        self.assertEqual(select_pairs([row(0, length=18), row(1, -0.1, length=10)]), [(0, 1)])
        self.assertEqual(select_pairs([row(0, old_per_token=-1), row(1, -0.1, old_per_token=-100)]), [(0, 1)])

    def test_priority_and_group_and_global_caps(self):
        rows = [row(0, 0.1), row(1, -0.1, value=0.21), row(2, -0.2, value=0.2),
                row(3, -0.3, value=0.2), row(4, 0.1, uid="q2"), row(5, -0.2, uid="q2")]
        self.assertEqual(select_pairs(rows), [(0, 3), (0, 2), (4, 5)])
        self.assertEqual(select_pairs(list(reversed(rows)), {"max_pairs_per_step": 2}), [(0, 3), (0, 2)])
        self.assertEqual(select_pairs(rows, {"k_pairs_per_group": 1}), [(0, 3), (4, 5)])

    def test_credit_gap_priority_keeps_value_match_as_hard_filter(self):
        rows = [row(0, 0.1), row(1, -0.1),
                row(2, -0.4, value=0.28), row(3, -0.8, value=0.4)]
        config = {"k_pairs_per_group": 1, "match_epsilon": 0.1}
        # The stronger contrast wins within the matched neighborhood; the
        # largest contrast is ineligible because its state is too far away.
        self.assertEqual(select_pairs(rows, config), [(0, 2)])
        self.assertEqual(select_pairs(list(reversed(rows)), config), [(0, 2)])

    def test_equal_credit_gaps_prefer_closer_states(self):
        rows = [row(0, 0.1), row(1, -0.2, value=0.24),
                row(2, -0.2, value=0.21)]
        self.assertEqual(select_pairs(rows, {"k_pairs_per_group": 1}), [(0, 2)])

    def test_no_negative_and_disabled_return_no_pairs(self):
        self.assertEqual(select_pairs([row(0), row(1)]), [])
        self.assertEqual(select_pairs([row(0), row(1, -0.1)], {"enabled": False}), [])

    def outcome_row(self, index, reward, **overrides):
        result = row(index, math.nan, turn_type="analysis", value=math.nan,
                     label_source="outcome", outcome_valid=True,
                     outcome_success=reward > 0.5, outcome_reward=reward)
        result.update(overrides)
        return result

    def test_analysis_outcome_pair_does_not_require_gt_or_trace_thresholds(self):
        rows = [self.outcome_row(0, 1), self.outcome_row(1, 0)]
        config = {"positive_threshold": 10, "negative_threshold": 10, "gap_epsilon": 10}
        self.assertEqual(select_pairs(rows, config), [(0, 1)])
        self.assertEqual(select_pairs(rows, {"outcome_match_value": True}), [])
        rows[0]["value"], rows[1]["value"] = 0, 0.01
        self.assertEqual(select_pairs(rows, {"outcome_match_value": True}), [(0, 1)])

    def test_only_actual_terminal_answers_use_outcome_labels(self):
        for turn_type in ("final_answer", "direct", "base"):
            with self.subTest(turn_type=turn_type):
                rows = [self.outcome_row(0, 1, turn_type=turn_type),
                        self.outcome_row(1, 0, turn_type=turn_type)]
                self.assertEqual(select_pairs(rows), [])
                for item in rows:
                    item["is_final"] = True
                self.assertEqual(select_pairs(rows), [(0, 1)])

    def test_untrusted_and_external_outcomes_cannot_pair(self):
        rows = [self.outcome_row(0, 1), self.outcome_row(1, 0)]
        for side in (0, 1):
            for updates in ({"outcome_valid": False}, {"external_error": True},
                            {"outcome_reward": math.nan}, {"outcome_success": "unknown"}):
                with self.subTest(side=side, updates=updates):
                    candidate = [dict(item) for item in rows]
                    candidate[side].update(updates)
                    self.assertEqual(select_pairs(candidate), [])

    def test_verifier_success_overrides_numeric_reward_threshold(self):
        rows = [self.outcome_row(0, 0.1, outcome_success=True),
                self.outcome_row(1, 0, outcome_success=False)]
        self.assertEqual(select_pairs(rows), [(0, 1)])
        rows[0]["outcome_success"] = rows[1]["outcome_success"] = None
        self.assertEqual(select_pairs(rows), [])

    def test_outcome_reward_gap_ranking_and_no_cross_route_fallback(self):
        rows = [self.outcome_row(0, 0.7), self.outcome_row(1, 0),
                self.outcome_row(2, 0.9)]
        self.assertEqual(select_pairs(rows, {"k_pairs_per_group": 1}), [(2, 1)])
        rows = [self.outcome_row(0, 1, turn_type="tool_plan", label_source="outcome"),
                self.outcome_row(1, 0, turn_type="tool_plan", label_source="outcome")]
        self.assertEqual(select_pairs(rows), [])

    def test_external_failure_tool_excluded_on_both_sides(self):
        for side in (0, 1):
            rows = [row(0), row(1, -0.1)]
            rows[side]["external_error"] = True
            self.assertEqual(select_pairs(rows), [])

    def test_metrics_funnel_is_ordered_and_conserves_candidates(self):
        rows = [
            row(0, uid="value", value=0, length=100, action_key="same"),
            row(1, -0.1, uid="value", value=0.5, length=1, action_key="same"),
            row(2, 0.06, uid="gap", length=100, action_key="same"),
            row(3, -0.06, uid="gap", length=1, action_key="same"),
            row(4, uid="length", length=19, action_key="same"),
            row(5, -0.1, uid="length", length=10, action_key="same"),
            row(6, uid="same", action_key="same"),
            row(7, -0.1, uid="same", action_key="same"),
            row(8, uid="pass"), row(9, -0.1, uid="pass"),
            row(10, uid="positive"), row(11, -0.1, uid="negative"),
            row(12, 0.01, uid="neutral"), row(13, -0.01, uid="neutral"),
            row(14, uid="invalid", valid=False),
        ]
        metrics = {}
        self.assertEqual(select_pairs(rows, {"gap_epsilon": 0.15}, metrics), [(8, 9)])
        expected = {
            "groups_eligible": 8,
            "groups_no_positive": 2, "groups_no_negative": 2,
            "groups_both_signs": 5,
            "candidates_sign": 5, "rejected_value_match": 1,
            "rejected_credit_gap": 1, "rejected_length": 1,
            "rejected_same_action": 1, "rejected_group_cap": 0,
            "rejected_global_cap": 0, "selected_unique": 1,
            "trace_candidates_sign": 5, "trace_after_value_match": 4,
        }
        self.assertEqual(metrics, expected)
        reasons = ["rejected_value_match", "rejected_credit_gap",
                   "rejected_length", "rejected_same_action", "rejected_group_cap",
                   "rejected_global_cap"]
        self.assertEqual(metrics["candidates_sign"],
                         metrics["selected_unique"] + sum(metrics[key] for key in reasons))

    def test_metrics_empty_calls_reset_all_keys(self):
        metrics = {"unrelated": 7}
        select_pairs([row(0), row(1, -0.1)], metrics=metrics)
        expected_keys = set(metrics) - {"unrelated"}
        self.assertEqual(select_pairs([], metrics=metrics), [])
        self.assertEqual(set(metrics) - {"unrelated"}, expected_keys)
        self.assertEqual(metrics["unrelated"], 7)
        self.assertTrue(all(metrics[key] == 0 for key in expected_keys))
        select_pairs([row(0)], metrics=metrics)
        self.assertEqual(metrics["groups_no_negative"], 1)
        self.assertEqual(metrics["candidates_sign"], 0)
        self.assertEqual(metrics["selected_unique"], 0)

    def test_metrics_invalid_and_nan_rows_do_not_enter_eligible_population(self):
        rows = [row(0), row(1, -0.1),
                row(2, value=math.nan), row(3, credit=math.nan),
                row(4, old_per_token=math.inf), row(5, length=math.nan),
                row(6, length=0), row(7, valid=False, uid="invalid-only")]
        metrics = {}
        self.assertEqual(select_pairs(rows, metrics=metrics), [(0, 1)])
        self.assertEqual(metrics["groups_eligible"], 1)
        self.assertEqual(metrics["groups_both_signs"], 1)
        self.assertEqual(metrics["groups_no_positive"], 0)
        self.assertEqual(metrics["groups_no_negative"], 0)
        self.assertEqual(metrics["candidates_sign"], 1)

    def test_metrics_omit_redundant_counts_and_keep_tool_match_denominator(self):
        rows = [row(0), row(1, -0.1), row(2, -0.1, value=0.5),
                self.outcome_row(3, 1), self.outcome_row(4, 0)]
        metrics = {}
        self.assertEqual(select_pairs(rows, metrics=metrics), [(3, 4), (0, 1)])
        self.assertEqual(metrics["candidates_sign"], 3)
        self.assertEqual(metrics["trace_candidates_sign"], 2)
        self.assertEqual(metrics["trace_after_value_match"], 1)
        self.assertEqual(metrics["rejected_value_match"], 1)
        removed = {
            "rows_total", "rows_eligible", "groups_total", "positive_rows",
            "negative_rows", "neutral_rows", "after_value_match", "after_credit_gap",
            "after_length_filter", "after_distinct_action", "outcome_candidates_sign",
            "outcome_after_value_match",
        }
        self.assertTrue(removed.isdisjoint(metrics))

    def test_metrics_caps_keep_original_selection_and_count_all_survivors(self):
        rows = [row(0), row(1, -0.1, value=0.21), row(2, -0.2),
                row(3, -0.3), row(4, uid="q2"), row(5, -0.2, uid="q2")]
        for group_cap, global_cap, expected in (
                (1, 1, [(0, 3)]), (1, 2, [(0, 3), (4, 5)]),
                (2, 2, [(0, 3), (0, 2)]),
                (2, 256, [(0, 3), (0, 2), (4, 5)])):
            config = {"k_pairs_per_group": group_cap, "max_pairs_per_step": global_cap}
            with self.subTest(config=config):
                metrics = {}
                self.assertEqual(select_pairs(rows, config), expected)
                self.assertEqual(select_pairs(rows, config, metrics), expected)
                self.assertEqual(select_pairs(list(reversed(rows)), config, {}), expected)
                self.assertEqual(metrics["candidates_sign"], 4)
                self.assertEqual(metrics["selected_unique"], len(expected))
                self.assertEqual(4, metrics["selected_unique"]
                                 + metrics["rejected_group_cap"]
                                 + metrics["rejected_global_cap"])
        metrics = {}
        select_pairs(rows, {"k_pairs_per_group": 1, "max_pairs_per_step": 1}, metrics)
        self.assertEqual(metrics["rejected_group_cap"], 2)
        self.assertEqual(metrics["rejected_global_cap"], 1)


class TraceConfigTests(unittest.TestCase):
    def test_defaults_and_mapping(self):
        self.assertEqual(TraceConfig.from_mapping().k, 3)
        self.assertEqual(TraceConfig.from_mapping().match_epsilon, 0.05)
        self.assertEqual(TraceConfig.from_mapping({"match_epsilon": 0.2}).match_epsilon, 0.2)

    def test_reject_invalid_configuration(self):
        for config in ({"k": 0}, {"k": 1.5}, {"gamma": 1.1}, {"epsilon": 0},
                       {"match_epsilon": -1}, {"positive_threshold": -1},
                       {"negative_threshold": -1}, {"gap_epsilon": math.nan},
                       {"length_ratio_max": 0.9}, {"max_pairs_per_step": 0},
                       {"answer_instruction": ""}, {"gt_alias_strategy": "mean"},
                       {"enabled": "true"}, {"terminal_fill_enabled": "true"},
                       {"outcome_match_value": 1}, {"terminal_fill_scale": -1},
                       {"terminal_fill_scale": math.inf},
                       {"outcome_success_threshold": math.nan}, {"unknown": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                TraceConfig.from_mapping(config)


if __name__ == "__main__":
    unittest.main()
