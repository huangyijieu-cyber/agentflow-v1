"""CPU-only diagnostics tests, independent of torch and distributed workers."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "agentflow" / "trace_metrics.py"
SPEC = importlib.util.spec_from_file_location("trace_metrics", MODULE_PATH)
trace_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_metrics)

CONFIG = SimpleNamespace(k=1, positive_threshold=0.01, negative_threshold=0.01)


def summarize(logps, values, credits, types, rollouts, turns):
    return trace_metrics.summarize_states(logps, values, credits, types, rollouts, turns, CONFIG)


def row(index, **overrides):
    result = dict(index=index, uid="q", rollout_id="r", turn_index=index,
                  value=0.2, credit=0.1, length=10)
    result.update(overrides)
    return result


class TraceStateMetricTests(unittest.TestCase):
    def test_terminal_states_do_not_reduce_credit_coverage(self):
        metrics = summarize([-2, -1, -3, -2, -1], [0, 1, 0, 1, 2],
                            [1, math.nan, -1, 0, math.nan],
                            ["analysis", "final", "analysis", "tool", "final"],
                            [("q", "a"), ("q", "a"), ("q", "b"), ("q", "b"), ("q", "b")],
                            [0, 1, 0, 1, 2])
        self.assertEqual(metrics["trace/terminal_turns"], 2)
        self.assertEqual(metrics["trace/credit_expected_turns"], 3)
        self.assertEqual(metrics["trace/credit_coverage"], 1)
        self.assertEqual(metrics["trace/credit_missing_nonterminal"], 0)
        self.assertEqual(metrics["trace/credit_count"], 3)
        self.assertAlmostEqual(metrics["trace/positive_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["trace/negative_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["trace/neutral_rate"], 1 / 3)
        self.assertEqual(metrics["trace/analysis/credit_count"], 2)
        self.assertEqual(metrics["trace/analysis/positive_rate"], 0.5)
        self.assertEqual(metrics["trace/final/credit_count"], 0)
        self.assertNotIn("trace/final/credit_mean", metrics)
        self.assertEqual(metrics["trace/tool/neutral_rate"], 1)

    def test_missing_intermediate_scores_count_as_missing_credit(self):
        metrics = summarize([-1, None, 0.5, math.inf, -2], [0, math.nan, 2, None, 3],
                            [None, math.nan, -0.01, math.inf, math.nan],
                            ["tool"] * 5, ["r"] * 5, [0, 1, 2, 3, 4])
        self.assertEqual(metrics["trace/score_coverage"], 2 / 5)
        self.assertEqual(metrics["trace/gt_logp_count"], 2)
        self.assertEqual(metrics["trace/credit_expected_turns"], 4)
        self.assertEqual(metrics["trace/credit_missing_nonterminal"], 3)
        self.assertEqual(metrics["trace/credit_coverage"], 0.25)
        self.assertEqual(metrics["trace/neutral_turns"], 1)
        self.assertEqual(metrics["trace/negative_turns"], 0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_terminal_identification_uses_largest_original_turn(self):
        metrics = summarize([-1] * 3, [0, 1, 2], [math.nan, 1, 1], ["tool"] * 3,
                            [("q", "r")] * 3, [7, 1, 4])
        self.assertEqual(metrics["trace/credit_coverage"], 1)
        self.assertEqual(metrics["trace/credit_missing_nonterminal"], 0)

    def test_tuple_ids_keep_questions_with_shared_rollout_names_separate(self):
        metrics = summarize([-1] * 4, [0, 1, 0, 1], [1, math.nan, -1, math.nan],
                            ["tool"] * 4, [("q1", "r"), ("q1", "r"), ("q2", "r"), ("q2", "r")],
                            [0, 1, 0, 1])
        self.assertEqual(metrics["trace/terminal_turns"], 2)
        self.assertEqual(metrics["trace/credit_expected_turns"], 2)

    def test_empty_batch_and_singletons_have_no_fake_quantiles(self):
        metrics = summarize([], [], [], [], [], [])
        self.assertEqual(metrics["trace/turns_total"], 0)
        self.assertEqual(metrics["trace/credit_count"], 0)
        self.assertNotIn("trace/credit_p50", metrics)
        self.assertNotIn("trace/value_mean", metrics)
        self.assertEqual(metrics["trace/credit_coverage"], 0)
        metrics = summarize([-1], [0], [math.nan], ["direct"], ["r"], [0])
        self.assertEqual(metrics["trace/credit_expected_turns"], 0)
        self.assertEqual(metrics["trace/credit_missing_nonterminal"], 0)

    def test_large_finite_values_remain_finite(self):
        metrics = summarize([-1e308, -1e308], [-1e308, 1e308], [-1e308, 1e308],
                            ["tool", "tool"], ["r", "r"], [0, 1])
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["trace/value_std"], 1e308)

    def test_mismatched_inputs_raise(self):
        with self.assertRaises(ValueError):
            summarize([-1], [], [], [], [], [])


class TracePairMetricTests(unittest.TestCase):
    def test_pair_gaps_cross_turn_and_unique_row_coverage(self):
        rows = [row(10, value=0.2, credit=0.1, turn_index=1),
                row(20, value=0.24, credit=-0.1, turn_index=3),
                row(30, value=0.26, credit=-0.2, turn_index=1, rollout_id="other", length=15)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(10, 20), (10, 30)])
        self.assertEqual(metrics["trace/pair/selected_unique"], 2)
        self.assertEqual(metrics["trace/pair/chosen_rows_unique"], 1)
        self.assertEqual(metrics["trace/pair/rejected_rows_unique"], 2)
        self.assertEqual(metrics["trace/pair/rows_unique"], 3)
        self.assertEqual(metrics["trace/pair/questions_covered"], 1)
        self.assertEqual(metrics["trace/pair/cross_turn_rate"], 0.5)
        self.assertEqual(metrics["trace/pair/cross_rollout_rate"], 0.5)
        self.assertAlmostEqual(metrics["trace/pair/value_gap_p50"], 0.05)
        self.assertAlmostEqual(metrics["trace/pair/value_gap_max"], 0.06)
        self.assertAlmostEqual(metrics["trace/pair/credit_gap_min"], 0.2)
        self.assertAlmostEqual(metrics["trace/pair/chosen_credit_mean"], 0.1)
        self.assertAlmostEqual(metrics["trace/pair/rejected_credit_mean"], -0.15)
        self.assertAlmostEqual(metrics["trace/pair/length_ratio_p90"], 1.45)
        self.assertEqual(metrics["trace/pair/outcome_pairs_count"], 0)
        self.assertNotIn("trace/pair/chosen_failed_rollout_rate", metrics)

    def test_only_selected_unpadded_pairs_affect_counts(self):
        rows = [row(0), row(1, credit=-0.1), row(2, uid="other"), row(3, uid="other", credit=-0.2)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1)])
        self.assertEqual(metrics["trace/pair/selected_unique"], 1)
        self.assertEqual(metrics["trace/pair/questions_covered"], 1)
        self.assertEqual(metrics["trace/pair/rows_unique"], 2)

    def test_optional_outcomes_describe_credit_outcome_disagreement(self):
        rows = [row(0, outcome_reward=0), row(1, outcome_reward=1),
                row(2, outcome_reward=1), row(3, outcome_reward=1),
                row(4, outcome_reward=math.nan)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1), (2, 3), (0, 4)])
        self.assertEqual(metrics["trace/pair/outcome_pairs_count"], 2)
        self.assertEqual(metrics["trace/pair/chosen_failed_rollout_rate"], 0.5)
        self.assertEqual(metrics["trace/pair/rejected_successful_rollout_rate"], 1)
        self.assertEqual(metrics["trace/pair/same_outcome_rate"], 0.5)

    def test_empty_pairs_and_missing_values_omit_distribution_metrics(self):
        metrics = trace_metrics.summarize_selected_pairs([], [])
        self.assertEqual(metrics["trace/pair/selected_unique"], 0)
        self.assertEqual(metrics["trace/pair/value_gap_count"], 0)
        self.assertNotIn("trace/pair/value_gap_p50", metrics)
        rows = [row(0, value=math.nan, credit=None, length=0),
                row(1, value=math.inf, credit=math.nan)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1)])
        self.assertEqual(metrics["trace/pair/chosen_credit_count"], 0)
        self.assertNotIn("trace/pair/chosen_credit_mean", metrics)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_duplicate_row_indices_raise(self):
        with self.assertRaises(ValueError):
            trace_metrics.summarize_selected_pairs([row(0), row(0)], [])


if __name__ == "__main__":
    unittest.main()
