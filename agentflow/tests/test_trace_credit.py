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
        self.assertAlmostEqual(values[1], math.log((4.0 + 1e-8) / (2.0 + 1e-8)))
        self.assertLess(values[2], 0)
        credit = discounted_credits(values)
        self.assertAlmostEqual(credit[0], values[1] - values[0])
        self.assertAlmostEqual(credit[1], values[2] - values[1])
        self.assertTrue(math.isnan(credit[2]))

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
