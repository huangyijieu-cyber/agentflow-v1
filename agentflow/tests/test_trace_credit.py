"""CPU-only tests for TRACE credit and strict SLIC selection."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "agentflow" / "trace_credit.py"
SPEC = importlib.util.spec_from_file_location("trace_credit", MODULE_PATH)
trace_credit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trace_credit
SPEC.loader.exec_module(trace_credit)

TraceConfig = trace_credit.TraceConfig
state_values = trace_credit.state_values
discounted_credits = trace_credit.discounted_credits
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

    def test_no_negative_and_disabled_return_no_pairs(self):
        self.assertEqual(select_pairs([row(0), row(1)]), [])
        self.assertEqual(select_pairs([row(0), row(1, -0.1)], {"enabled": False}), [])

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
            "rows_total": 15, "rows_eligible": 14,
            "groups_total": 9, "groups_eligible": 8,
            "groups_no_positive": 2, "groups_no_negative": 2,
            "groups_both_signs": 5, "positive_rows": 6,
            "negative_rows": 6, "neutral_rows": 2,
            "candidates_sign": 5, "after_value_match": 4,
            "after_credit_gap": 3, "after_length_filter": 2,
            "after_distinct_action": 1, "rejected_value_match": 1,
            "rejected_credit_gap": 1, "rejected_length": 1,
            "rejected_same_action": 1, "rejected_group_cap": 0,
            "rejected_global_cap": 0, "selected_unique": 1,
        }
        self.assertEqual(metrics, expected)
        stages = ["candidates_sign", "after_value_match", "after_credit_gap",
                  "after_length_filter", "after_distinct_action"]
        reasons = ["rejected_value_match", "rejected_credit_gap",
                   "rejected_length", "rejected_same_action"]
        for before, after, rejected in zip(stages, stages[1:], reasons):
            self.assertEqual(metrics[before], metrics[after] + metrics[rejected])
        self.assertEqual(metrics["after_distinct_action"],
                         metrics["selected_unique"] + metrics["rejected_group_cap"]
                         + metrics["rejected_global_cap"])

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
        self.assertEqual(metrics["rows_total"], 8)
        self.assertEqual(metrics["rows_eligible"], 2)
        self.assertEqual(metrics["groups_total"], 2)
        self.assertEqual(metrics["groups_eligible"], 1)
        self.assertEqual(metrics["positive_rows"], 1)
        self.assertEqual(metrics["negative_rows"], 1)
        self.assertEqual(metrics["neutral_rows"], 0)
        self.assertEqual(metrics["candidates_sign"], 1)

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
                self.assertEqual(metrics["after_distinct_action"], 4)
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
        self.assertEqual(TraceConfig.from_mapping().k, 1)
        self.assertEqual(TraceConfig.from_mapping({"match_epsilon": 0.2}).match_epsilon, 0.2)

    def test_reject_invalid_configuration(self):
        for config in ({"k": 0}, {"k": 1.5}, {"gamma": 1.1}, {"epsilon": 0},
                       {"match_epsilon": -1}, {"positive_threshold": -1},
                       {"negative_threshold": -1}, {"gap_epsilon": math.nan},
                       {"length_ratio_max": 0.9}, {"max_pairs_per_step": 0},
                       {"answer_instruction": ""}, {"gt_alias_strategy": "mean"},
                       {"enabled": "true"}, {"unknown": 1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                TraceConfig.from_mapping(config)


if __name__ == "__main__":
    unittest.main()
