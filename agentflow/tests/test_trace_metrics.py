"""CPU-only diagnostics tests, independent of torch and distributed workers."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "agentflow" / "trace" / "metrics.py"
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
        self.assertEqual(metrics["trace/credit_expected_turns"], 3)
        self.assertEqual(metrics["trace/credit_coverage"], 1)
        self.assertEqual(metrics["trace/credit_count"], 3)
        self.assertAlmostEqual(metrics["trace/positive_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["trace/negative_rate"], 1 / 3)
        self.assertNotIn("trace/final/credit_mean", metrics)

    def test_missing_intermediate_scores_count_as_missing_credit(self):
        metrics = summarize([-1, None, 0.5, math.inf, -2], [0, math.nan, 2, None, 3],
                            [None, math.nan, -0.01, math.inf, math.nan],
                            ["tool"] * 5, ["r"] * 5, [0, 1, 2, 3, 4])
        self.assertEqual(metrics["trace/score_coverage"], 2 / 5)
        self.assertEqual(metrics["trace/gt_logp_count"], 2)
        self.assertEqual(metrics["trace/credit_expected_turns"], 4)
        self.assertEqual(metrics["trace/credit_coverage"], 0.25)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_terminal_identification_uses_largest_original_turn(self):
        metrics = summarize([-1] * 3, [0, 1, 2], [math.nan, 1, 1], ["tool"] * 3,
                            [("q", "r")] * 3, [7, 1, 4])
        self.assertEqual(metrics["trace/credit_coverage"], 1)

    def test_tuple_ids_keep_questions_with_shared_rollout_names_separate(self):
        metrics = summarize([-1] * 4, [0, 1, 0, 1], [1, math.nan, -1, math.nan],
                            ["tool"] * 4, [("q1", "r"), ("q1", "r"), ("q2", "r"), ("q2", "r")],
                            [0, 1, 0, 1])
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

    def test_large_finite_values_remain_finite(self):
        metrics = summarize([-1e308, -1e308], [-1e308, 1e308], [-1e308, 1e308],
                            ["tool", "tool"], ["r", "r"], [0, 1])
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["trace/gt_logp_mean"], -1e308)

    def test_mismatched_inputs_raise(self):
        with self.assertRaises(ValueError):
            summarize([-1], [], [], [], [], [])

    def test_tool_credit_components_and_coverage_exclude_outcome_rows(self):
        metrics = trace_metrics.summarize_states(
            [-8, -7, -6, -5], [0, 0.1, 0.2, 0.3],
            [math.nan, 0.15, 0.2, math.nan],
            ["analysis", "tool_plan", "tool_plan", "final_answer"],
            ["r"] * 4, [0, 1, 2, 3], CONFIG,
            local_credits=[math.nan, 0.1, 0.2, math.nan],
            terminal_fills=[math.nan, 0.05, 0, math.nan],
            credit_expected_mask=[False, True, True, False],
            label_sources=["outcome", "trace", "trace", "outcome"],
        )
        self.assertEqual(metrics["trace/credit_expected_turns"], 2)
        self.assertEqual(metrics["trace/credit_coverage"], 1)
        self.assertAlmostEqual(metrics["trace/local_credit_mean"], 0.15)
        self.assertAlmostEqual(metrics["trace/terminal_fill_mean"], 0.025)
        self.assertAlmostEqual(metrics["trace/credit_mean"], 0.175)
        self.assertEqual(metrics["trace/terminal_fill_nonzero_turns"], 1)

    def test_external_windows_and_invalid_outcomes_have_explicit_counts(self):
        metrics = trace_metrics.summarize_states(
            [-8] * 4, [0] * 4, [math.nan] * 4,
            ["analysis", "tool_plan", "tool_plan", "final_answer"],
            [("q", "r")] * 4, [0, 1, 2, 3], CONFIG,
            local_credits=[math.nan] * 4, terminal_fills=[math.nan] * 4,
            window_blocked=[False, True, True, False],
            external_errors=[False, False, True, False],
            outcome_valid=[False] * 4,
            credit_expected_mask=[False, True, True, False],
        )
        self.assertEqual(metrics["trace/external_error_turns"], 1)
        self.assertEqual(metrics["trace/credit_windows_blocked_external"], 2)
        self.assertEqual(metrics["trace/outcome_invalid_rollouts"], 1)
        self.assertEqual(metrics["trace/local_credit_count"], 0)
        self.assertNotIn("trace/local_credit_mean", metrics)

    def test_optional_state_observations_require_aligned_lengths(self):
        with self.assertRaises(ValueError):
            trace_metrics.summarize_states([-1], [0], [math.nan], ["tool"], ["r"], [0],
                                           CONFIG, local_credits=[])

    def test_post_final_outputs_do_not_invalidate_trusted_rollouts(self):
        metrics = trace_metrics.summarize_states(
            [-8] * 6, [0] * 6, [math.nan] * 6,
            ["base", "direct", "final_answer"] * 2,
            [("q", "a")] * 3 + [("q", "b")] * 3,
            [0, 1, 2] * 2, CONFIG,
            # The actual evaluated final is index 0; later optional outputs
            # cannot inherit its label, but both rollouts remain trusted.
            outcome_valid=[True, False, False, True, False, False],
            credit_expected_mask=[False] * 6,
        )
        self.assertEqual(metrics["trace/outcome_invalid_rollouts"], 0)

    def test_only_rollouts_without_any_trusted_outcome_are_invalid(self):
        metrics = trace_metrics.summarize_states(
            [-8] * 4, [0] * 4, [math.nan] * 4, ["base", "direct"] * 2,
            [("q1", "r")] * 2 + [("q2", "r")] * 2,
            [0, 1] * 2, CONFIG, outcome_valid=[True, False, False, False],
        )
        self.assertEqual(metrics["trace/outcome_invalid_rollouts"], 1)

    def test_state_schema_omits_duplicate_and_misleading_statistics(self):
        metrics = trace_metrics.summarize_states(
            [-8, -7, -6, -5], [0, 0.1, 0.2, 0.3],
            [math.nan, 0.15, -0.2, math.nan],
            ["analysis", "tool_plan", "tool_plan", "final_answer"],
            ["r"] * 4, [0, 1, 2, 3], CONFIG,
            local_credits=[math.nan, 0.1, -0.2, math.nan],
            terminal_fills=[math.nan, 0.05, 0, math.nan],
            window_blocked=[False] * 4, external_errors=[False] * 4,
            outcome_valid=[True] * 4,
            label_sources=["outcome", "trace", "trace", "outcome"],
            credit_expected_mask=[False, True, True, False],
        )
        expected = {
            "trace/turns_total", "trace/score_coverage", "trace/credit_expected_turns",
            "trace/credit_coverage", "trace/gt_logp_count", "trace/gt_logp_mean",
            "trace/positive_rate", "trace/negative_rate", "trace/terminal_fill_nonzero_turns",
            "trace/credit_windows_blocked_external", "trace/external_error_turns",
            "trace/outcome_invalid_rollouts",
        }
        expected.update(f"trace/{name}_{suffix}"
                        for name in ("credit", "local_credit", "terminal_fill")
                        for suffix in ("count", "mean", "p10", "p50", "p90"))
        self.assertEqual(set(metrics), expected)
        self.assertEqual(metrics["trace/credit_count"], 2)
        self.assertAlmostEqual(metrics["trace/credit_mean"], -0.025)

    def test_igpo_mode_reports_state_health_without_fake_trace_credit(self):
        metrics = trace_metrics.summarize_states(
            [-8, -7], [0, 0.1], [math.nan] * 2,
            ["tool_plan", "final_answer"], [("q", "r")] * 2, [0, 1], CONFIG,
            local_credits=[math.nan] * 2, terminal_fills=[math.nan] * 2,
            window_blocked=[True, False], external_errors=[True, False],
            outcome_valid=[False, False], include_credit=False,
        )
        self.assertEqual(metrics["trace/score_coverage"], 1)
        self.assertEqual(metrics["trace/external_error_turns"], 1)
        self.assertEqual(metrics["trace/outcome_invalid_rollouts"], 1)
        self.assertFalse(any("credit" in key or "terminal_fill" in key for key in metrics))
        self.assertNotIn("trace/positive_rate", metrics)


class TracePairMetricTests(unittest.TestCase):
    def test_pair_gaps_cross_turn_and_question_coverage(self):
        rows = [row(10, value=0.2, credit=0.1, turn_index=1),
                row(20, value=0.24, credit=-0.1, turn_index=3),
                row(30, value=0.26, credit=-0.2, turn_index=1, rollout_id="other", length=15)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(10, 20), (10, 30)])
        self.assertEqual(metrics["trace/pair/selected_unique"], 2)
        self.assertEqual(metrics["trace/pair/question_coverage"], 1)
        self.assertEqual(metrics["trace/pair/cross_turn_rate"], 0.5)
        self.assertAlmostEqual(metrics["trace/pair/value_gap_p50"], 0.05)
        self.assertAlmostEqual(metrics["trace/pair/value_gap_max"], 0.06)
        self.assertAlmostEqual(metrics["trace/pair/credit_gap_min"], 0.2)
        self.assertEqual(metrics["trace/pair/trace/outcome_pairs_count"], 0)
        self.assertNotIn("trace/pair/trace/chosen_failed_rollout_rate", metrics)

    def test_only_selected_unpadded_pairs_affect_counts(self):
        rows = [row(0), row(1, credit=-0.1), row(2, uid="other"), row(3, uid="other", credit=-0.2)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1)])
        self.assertEqual(metrics["trace/pair/selected_unique"], 1)
        self.assertEqual(metrics["trace/pair/question_coverage"], 0.5)

    def test_optional_outcomes_describe_credit_outcome_disagreement(self):
        rows = [row(0, outcome_reward=0), row(1, outcome_reward=1),
                row(2, outcome_reward=1), row(3, outcome_reward=1),
                row(4, outcome_reward=math.nan)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1), (2, 3), (0, 4)])
        self.assertEqual(metrics["trace/pair/trace/outcome_pairs_count"], 2)
        self.assertEqual(metrics["trace/pair/trace/chosen_failed_rollout_rate"], 0.5)
        self.assertEqual(metrics["trace/pair/trace/rejected_successful_rollout_rate"], 1)
        self.assertEqual(metrics["trace/pair/trace/same_outcome_rate"], 0.5)

    def test_explicit_numeric_verifier_flags_are_supported(self):
        rows = [row(0, label_source="trace", outcome_success=1, outcome_valid=True),
                row(1, label_source="trace", outcome_success=0, outcome_valid=True)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1)])
        self.assertEqual(metrics["trace/pair/trace/outcome_pairs_count"], 1)
        self.assertEqual(metrics["trace/pair/trace/rejected_successful_rollout_rate"], 0)

    def test_empty_pairs_and_missing_values_omit_distribution_metrics(self):
        metrics = trace_metrics.summarize_selected_pairs([], [])
        self.assertEqual(metrics["trace/pair/selected_unique"], 0)
        self.assertNotIn("trace/pair/value_gap_count", metrics)
        self.assertNotIn("trace/pair/value_gap_p50", metrics)
        rows = [row(0, value=math.nan, credit=None, length=0),
                row(1, value=math.inf, credit=math.nan)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1)])
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_duplicate_row_indices_raise(self):
        with self.assertRaises(ValueError):
            trace_metrics.summarize_selected_pairs([row(0), row(0)], [])

    def test_outcome_pairs_do_not_become_gt_credit_or_value_samples(self):
        rows = [row(0, label_source="outcome", outcome_valid=True,
                    outcome_success=True, outcome_reward=1, credit=1),
                row(1, label_source="outcome", outcome_valid=True,
                    outcome_success=False, outcome_reward=0.1, credit=-1)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1)])
        self.assertEqual(metrics["trace/pair/outcome_selected_unique"], 1)
        self.assertEqual(metrics["trace/pair/trace_selected_unique"], 0)
        self.assertNotIn("trace/pair/credit_gap_p50", metrics)
        self.assertNotIn("trace/pair/value_gap_p50", metrics)
        self.assertEqual(metrics["trace/pair/trace/outcome_pairs_count"], 0)
        self.assertNotIn("trace/pair/trace/rejected_successful_rollout_rate", metrics)

    def test_mixed_sources_keep_trace_distributions_separate(self):
        rows = [row(0, label_source="trace", outcome_success=True, outcome_valid=True),
                row(1, label_source="trace", credit=-0.1, value=0.24,
                    outcome_success=True, outcome_valid=True),
                row(2, label_source="outcome", value=math.nan, credit=math.nan,
                    outcome_success=True, outcome_valid=True, outcome_reward=1),
                row(3, label_source="outcome", value=math.nan, credit=math.nan,
                    outcome_success=False, outcome_valid=True, outcome_reward=0)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1), (2, 3)])
        self.assertEqual(metrics["trace/pair/selected_unique"], 2)
        self.assertEqual(metrics["trace/pair/trace_selected_unique"], 1)
        self.assertEqual(metrics["trace/pair/outcome_selected_unique"], 1)
        self.assertNotIn("trace/pair/value_gap_count", metrics)
        self.assertNotIn("trace/pair/credit_gap_count", metrics)
        self.assertAlmostEqual(metrics["trace/pair/value_gap_max"], 0.04)
        self.assertEqual(metrics["trace/pair/trace/outcome_pairs_count"], 1)
        self.assertEqual(metrics["trace/pair/trace/same_outcome_rate"], 1)

    def test_unverified_positive_format_reward_is_not_success(self):
        rows = [row(0, label_source="outcome", outcome_reward=1),
                row(1, label_source="outcome", outcome_reward=0.1),
                row(2, label_source="trace", outcome_success=False, outcome_valid=False),
                row(3, label_source="outcome", outcome_success=True)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1), (0, 2)])
        self.assertEqual(metrics["trace/pair/trace/outcome_pairs_count"], 0)
        self.assertNotIn("trace/pair/trace/same_outcome_rate", metrics)
        self.assertEqual(trace_metrics.summarize_selected_pairs(rows, [(3, 1)])[
            "trace/pair/trace/outcome_pairs_count"], 0)

    def test_pair_schema_omits_duplicate_counts_and_outcome_tautologies(self):
        rows = [row(0, label_source="trace", outcome_valid=True, outcome_success=False),
                row(1, label_source="trace", value=0.24, credit=-0.2,
                    outcome_valid=True, outcome_success=True),
                row(2, label_source="outcome", outcome_valid=True,
                    outcome_success=True, outcome_reward=1),
                row(3, label_source="outcome", outcome_valid=True,
                    outcome_success=False, outcome_reward=0)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1), (2, 3)])
        expected = {
            "trace/pair/selected_unique", "trace/pair/trace_selected_unique",
            "trace/pair/outcome_selected_unique", "trace/pair/question_coverage",
            "trace/pair/cross_turn_rate", "trace/pair/trace/outcome_pairs_count",
            "trace/pair/trace/chosen_failed_rollout_rate",
            "trace/pair/trace/rejected_successful_rollout_rate",
            "trace/pair/trace/same_outcome_rate",
            "trace/pair/value_gap_p50", "trace/pair/value_gap_p90", "trace/pair/value_gap_max",
            "trace/pair/credit_gap_p50", "trace/pair/credit_gap_p90", "trace/pair/credit_gap_min",
        }
        self.assertEqual(set(metrics), expected)
        self.assertEqual(metrics["trace/pair/trace/outcome_pairs_count"], 1)
        self.assertEqual(metrics["trace/pair/trace/chosen_failed_rollout_rate"], 1)

    def test_igpo_pair_gap_uses_advantage_and_keeps_outcome_separate(self):
        rows = [row(0, label_source="trace", credit=99, igpo_advantage=2, igpo_valid=True),
                row(1, label_source="trace", credit=-99, igpo_advantage=-1, igpo_valid=True),
                row(2, label_source="outcome", outcome_valid=True,
                    outcome_success=True, outcome_reward=1),
                row(3, label_source="outcome", outcome_valid=True,
                    outcome_success=False, outcome_reward=0)]
        config = SimpleNamespace(pair_score_mode="igpo")
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1), (2, 3)], config)
        self.assertEqual(metrics["trace/pair/igpo_advantage_gap_min"], 3)
        self.assertEqual(metrics["trace/pair/trace_selected_unique"], 1)
        self.assertEqual(metrics["trace/pair/outcome_selected_unique"], 1)
        self.assertNotIn("trace/pair/credit_gap_p50", metrics)

    def test_igpo_pair_has_no_legacy_credit_fallback(self):
        rows = [row(0, pair_score_mode="igpo", credit=1, igpo_advantage=2, igpo_valid=False),
                row(1, pair_score_mode="igpo", credit=-1, igpo_advantage=-1, igpo_valid=True)]
        metrics = trace_metrics.summarize_selected_pairs(rows, [(0, 1)])
        self.assertNotIn("trace/pair/igpo_advantage_gap_p50", metrics)
        self.assertNotIn("trace/pair/credit_gap_p50", metrics)


def igpo_annotations(size, **overrides):
    fields = ("raw_ig", "normalized_ig", "outcome_z", "raw_ig_return",
              "ig_return", "terminal_return", "advantage", "remaining_turns")
    result = {name: [math.nan] * size for name in fields}
    result.update({name: [False] * size for name in ("valid", "blocked", "external_blocked")})
    result["metrics"] = {}
    result.update(overrides)
    return result


class IgpoMetricTests(unittest.TestCase):
    def test_zero_ig_and_relative_sign_flip_are_reported_without_analysis(self):
        rows = [row(0, turn_type="tool_plan"), row(1, turn_type="tool_plan"),
                row(2, turn_type="tool_plan"), row(3, turn_type="analysis")]
        annotated = igpo_annotations(
            4, raw_ig=[0, 1, -1, 999], normalized_ig=[0, 1, -1, 999],
            ig_return=[1, -2, 1, 999], terminal_return=[0, 0, 0, 999],
            advantage=[1, -2, 1, 999], remaining_turns=[3, 2, 1, 999],
            valid=[True, True, True, True], metrics={"groups_valid": 1},
        )
        metrics = trace_metrics.summarize_igpo(rows, annotated)
        self.assertEqual(metrics["trace/igpo/raw_ig_count"], 3)
        self.assertEqual(metrics["trace/igpo/raw_ig_mean"], 0)
        self.assertEqual(metrics["trace/igpo/advantage_count"], 3)
        self.assertEqual(metrics["trace/igpo/advantage_coverage"], 1)
        self.assertEqual(metrics["trace/igpo/sign_compared_turns"], 2)
        self.assertEqual(metrics["trace/igpo/raw_ig_advantage_sign_flip_rate"], 1)
        self.assertEqual(metrics["trace/igpo/groups_valid"], 1)
        self.assertNotIn("trace/credit_mean", metrics)

    def test_outcome_normalization_population_is_one_vote_per_rollout(self):
        rows = [row(0, turn_type="analysis", outcome_valid=True),
                row(1, turn_type="final_answer", is_final=True, outcome_valid=True),
                row(2, turn_type="final_answer", is_final=True, outcome_valid=True),
                row(3, turn_type="final_answer", uid="q2", is_final=True, outcome_valid=True),
                row(4, turn_type="final_answer", rollout_id="untrusted", is_final=True,
                    outcome_valid=False)]
        annotated = igpo_annotations(5, outcome_z=[999, -1, -1, 1, 999])
        metrics = trace_metrics.summarize_igpo(rows, annotated)
        self.assertEqual(metrics["trace/igpo/outcome_z_count"], 2)
        self.assertEqual(metrics["trace/igpo/outcome_z_mean"], 0)

    def test_invalid_returns_are_not_described_as_eligible_advantages(self):
        rows = [row(0, turn_type="tool_plan"), row(1, turn_type="tool_plan")]
        annotated = igpo_annotations(
            2, raw_ig=[1, -1], normalized_ig=[1, -1],
            ig_return=[999, 2], terminal_return=[999, 1],
            advantage=[999, 3], valid=[False, True], remaining_turns=[999, 1],
        )
        metrics = trace_metrics.summarize_igpo(rows, annotated)
        self.assertEqual(metrics["trace/igpo/advantage_count"], 1)
        self.assertEqual(metrics["trace/igpo/advantage_mean"], 3)
        self.assertEqual(metrics["trace/igpo/terminal_return_mean"], 1)
        self.assertEqual(metrics["trace/igpo/advantage_coverage"], 0.5)
        self.assertEqual(metrics["trace/igpo/advantage_remaining_turns_corr"], 0)

    def test_post_final_tools_do_not_reduce_advantage_coverage(self):
        rows = [row(0, turn_type="tool_plan", within_final_prefix=True),
                row(1, turn_type="tool_plan", within_final_prefix=False)]
        annotated = igpo_annotations(2, advantage=[1, math.nan], valid=[True, False])
        metrics = trace_metrics.summarize_igpo(rows, annotated)
        self.assertEqual(metrics["trace/igpo/advantage_coverage"], 1)

    def test_length_correlation_is_finite_for_constants_and_extreme_values(self):
        rows = [row(0, turn_type="tool_plan"), row(1, turn_type="tool_plan")]
        annotated = igpo_annotations(
            2, advantage=[-1e308, 1e308], valid=[True, True], remaining_turns=[1, 2])
        metrics = trace_metrics.summarize_igpo(rows, annotated)
        self.assertAlmostEqual(metrics["trace/igpo/advantage_remaining_turns_corr"], 1)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        annotated["advantage"] = [2, 2]
        metrics = trace_metrics.summarize_igpo(rows, annotated)
        self.assertEqual(metrics["trace/igpo/advantage_remaining_turns_corr"], 0)

    def test_empty_input_omits_fake_distributions_and_merges_helper_counters(self):
        annotated = igpo_annotations(0, metrics={
            "groups_total": 0, "igpo/groups_valid": 0, "trace/igpo/groups_low_ig_std": 0})
        metrics = trace_metrics.summarize_igpo([], annotated)
        self.assertEqual(metrics["trace/igpo/advantage_count"], 0)
        self.assertEqual(metrics["trace/igpo/advantage_coverage"], 0)
        self.assertEqual(metrics["trace/igpo/groups_low_ig_std"], 0)
        self.assertNotIn("trace/igpo/advantage_p50", metrics)

    def test_misaligned_igpo_arrays_raise(self):
        with self.assertRaises(ValueError):
            trace_metrics.summarize_igpo([row(0)], igpo_annotations(0))


if __name__ == "__main__":
    unittest.main()
