"""CPU regression tests for the IGPO-style SLIC score construction."""

import importlib.util
import math
from pathlib import Path
import random
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "igpo_credit_test", ROOT / "agentflow/agentflow/igpo_credit.py")
IGPO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IGPO)


def rollout(name, deltas, reward=0.0, *, uid="q", analysis=True):
    rows = []
    logp = -10.0
    if analysis:
        rows.append(dict(turn_index=0, turn_type="analysis", gt_logp=logp))
    offset = len(rows)
    for index, delta in enumerate(deltas):
        rows.append(dict(turn_index=offset + index, turn_type="tool_plan", gt_logp=logp,
                         action_key=f"{name}-action-{index}"))
        logp += delta
    rows.append(dict(turn_index=len(rows), turn_type="final_answer", gt_logp=logp, is_final=True))
    for row in rows:
        row.update(uid=uid, rollout_id=name, value=0.0, valid=True,
                   external_error=False, outcome_valid=True, outcome_reward=reward)
        row.setdefault("is_final", False)
    return rows


def indexed(*rollouts):
    return [dict(row, index=index) for index, row in
            enumerate(row for sequence in rollouts for row in sequence)]


class IgpoCreditTests(unittest.TestCase):
    def test_population_normalization_and_full_return_are_hand_calculable(self):
        rows = indexed(rollout("a", [1, 3], 1), rollout("b", [-1, 1], 0))
        result = IGPO.compute_igpo_advantages(rows)
        z = 2 / (math.sqrt(2) + 1e-6)
        out = 0.5 / (0.5 + 1e-6)
        for position, expected in ((1, z + out), (2, z + out),
                                   (5, -z - out), (6, -out)):
            self.assertAlmostEqual(result["advantage"][position], expected)
            self.assertTrue(result["valid"][position])
        self.assertEqual(result["raw_ig"][1:3], [1, 3])
        self.assertEqual(result["remaining_turns"][1:3], [2, 1])
        self.assertEqual(result["metrics"]["groups_valid"], 1)
        for position in (0, 3, 4, 7):
            self.assertTrue(math.isnan(result["advantage"][position]))
            self.assertTrue(math.isnan(result["raw_ig"][position]))

    def test_discount_and_component_weights_apply_once_and_not_as_average(self):
        rows = indexed(rollout("a", [1, 3], 1), rollout("b", [-1, 1], 0))
        config = dict(igpo_gamma=0.5, igpo_ig_weight=2.0, igpo_outcome_weight=3.0,
                      k=1, gamma=0.01, terminal_fill_enabled=True, terminal_fill_scale=999)
        result = IGPO.compute_igpo_advantages(rows, config)
        z = 2 / (math.sqrt(2) + 1e-6)
        out = 0.5 / (0.5 + 1e-6)
        self.assertAlmostEqual(result["ig_return"][1], z)
        self.assertAlmostEqual(result["terminal_return"][1], 0.75 * out)
        self.assertAlmostEqual(result["advantage"][1], z + 0.75 * out)
        self.assertAlmostEqual(result["ig_return"][2], 2 * z)
        self.assertAlmostEqual(result["terminal_return"][2], 1.5 * out)
        self.assertAlmostEqual(result["raw_ig_return"][1], 2.5)

    def test_true_zero_ig_remains_an_observation_and_a_turn_boundary(self):
        rows = indexed(rollout("a", [0, 2]), rollout("b", [-2, 0]))
        result = IGPO.compute_igpo_advantages(rows)
        self.assertEqual(result["raw_ig"][1], 0)
        self.assertEqual(result["normalized_ig"][1], 0)
        self.assertTrue(result["valid"][1])
        self.assertGreater(result["advantage"][1], 0)
        self.assertEqual(result["remaining_turns"][1], 2)
        self.assertTrue(result["valid"][6])
        self.assertEqual(result["advantage"][6], 0)

    def test_all_failed_outcomes_can_retain_ig_signal(self):
        rows = indexed(rollout("a", [1, 2], 0), rollout("b", [-1, -2], 0))
        result = IGPO.compute_igpo_advantages(rows)
        for position in (1, 2, 5, 6):
            self.assertEqual(result["outcome_z"][position], 0)
            self.assertEqual(result["terminal_return"][position], 0)
            self.assertTrue(result["valid"][position])
        self.assertGreater(result["advantage"][1], 0)
        self.assertLess(result["advantage"][5], 0)

    def test_relative_negative_can_have_positive_raw_ig(self):
        rows = indexed(rollout("a", [1]), rollout("b", [3]))
        result = IGPO.compute_igpo_advantages(rows)
        self.assertGreater(result["raw_ig"][1], 0)
        self.assertLess(result["advantage"][1], 0)
        self.assertGreater(result["advantage"][4], 0)

    def test_single_rollout_is_not_replaced_by_terminal_reward(self):
        rows = indexed(rollout("a", [-1, 2], 1))
        result = IGPO.compute_igpo_advantages(rows)
        self.assertFalse(any(result["valid"]))
        self.assertEqual(result["metrics"]["groups_insufficient_rollouts"], 1)
        self.assertEqual(result["outcome_z"][3], 0)

    def test_constant_ig_skips_tools_even_when_outcomes_differ(self):
        rows = indexed(rollout("a", [1], 1), rollout("b", [1], 0))
        result = IGPO.compute_igpo_advantages(rows)
        self.assertFalse(any(result["valid"]))
        self.assertEqual(result["metrics"]["groups_low_ig_std"], 1)
        self.assertTrue(all(math.isnan(result["advantage"][position]) for position in (1, 4)))

    def test_nearly_constant_ig_respects_the_configured_std_floor(self):
        rows = indexed(rollout("a", [0]), rollout("b", [0.0001]))
        blocked = IGPO.compute_igpo_advantages(rows)
        allowed = IGPO.compute_igpo_advantages(rows, {"igpo_ig_std_min": 1e-6})
        self.assertFalse(any(blocked["valid"]))
        self.assertTrue(allowed["valid"][1])

    def test_future_external_failure_blocks_prefix_but_not_recovered_suffix(self):
        first = rollout("a", [1, 1, 3], 1)
        first[2]["external_error"] = True
        first[2]["valid"] = False
        rows = indexed(first, rollout("b", [-1, -1, -3], 0))
        result = IGPO.compute_igpo_advantages(rows)
        self.assertEqual(result["raw_ig"][1], 1)
        self.assertTrue(math.isnan(result["normalized_ig"][1]))
        self.assertFalse(result["valid"][1])
        self.assertFalse(result["valid"][2])
        self.assertTrue(result["external_blocked"][1])
        self.assertTrue(result["external_blocked"][2])
        self.assertTrue(result["valid"][3])
        self.assertFalse(result["external_blocked"][3])

    def test_invalid_future_action_is_not_deleted_to_join_surrounding_states(self):
        first = rollout("a", [1, 1, 3], 1)
        first[2]["valid"] = False
        result = IGPO.compute_igpo_advantages(indexed(first, rollout("b", [-1, -1, -3])))
        self.assertEqual(result["blocked"][1:4], [True, True, False])
        self.assertFalse(any(result["external_blocked"]))
        self.assertTrue(result["valid"][3])

    def test_repeated_successful_action_blocks_dependent_prefix(self):
        first = rollout("a", [1, 1, 3], 1)
        first[2]["action_key"] = first[1]["action_key"]
        result = IGPO.compute_igpo_advantages(indexed(first, rollout("b", [-1, -1, -3])))
        self.assertEqual(result["blocked"][1:4], [True, True, False])
        self.assertTrue(math.isnan(result["raw_ig"][2]))

    def test_failed_attempt_does_not_make_successful_retry_duplicate(self):
        first = rollout("a", [1, 1, 3], 1)
        first[1].update(external_error=True, valid=False)
        first[2]["action_key"] = first[1]["action_key"]
        result = IGPO.compute_igpo_advantages(indexed(first, rollout("b", [-1, -1, -3])))
        self.assertFalse(result["valid"][1])
        self.assertTrue(result["valid"][2])
        self.assertTrue(result["valid"][3])

    def test_original_turn_gap_blocks_only_its_dependent_prefix(self):
        first = rollout("a", [1, 1, 3], 1)
        for row in first[2:]:
            row["turn_index"] += 1
        result = IGPO.compute_igpo_advantages(indexed(first, rollout("b", [-1, -1, -3])))
        self.assertFalse(result["valid"][1])
        self.assertTrue(result["valid"][2])
        self.assertTrue(result["valid"][3])
        self.assertEqual(result["remaining_turns"][1], 4)

    def test_identical_padding_copies_do_not_change_normalization_or_outcome_votes(self):
        rows = indexed(rollout("a", [1, 3], 1), rollout("b", [-1, 1], 0))
        baseline = IGPO.compute_igpo_advantages(rows)
        padded = rows + [dict(rows[1], index=100), dict(rows[3], index=101)]
        result = IGPO.compute_igpo_advantages(padded)
        for field in ("normalized_ig", "outcome_z", "advantage"):
            for position in (1, 2, 5, 6):
                self.assertEqual(result[field][position], baseline[field][position])
        self.assertFalse(any(result["valid"][-2:]))
        self.assertEqual(result["metrics"]["duplicate_rows"], 2)
        self.assertEqual(result["metrics"]["rollouts_trusted"], 2)

    def test_conflicting_copy_invalidates_the_transition_and_dependent_prefix(self):
        rows = indexed(rollout("a", [1, 1, 3], 1), rollout("b", [-1, -1, -3]))
        rows.append(dict(rows[2], index=100, response_key=(999,)))
        result = IGPO.compute_igpo_advantages(rows)
        self.assertFalse(result["valid"][1])
        self.assertFalse(result["valid"][2])
        self.assertFalse(result["valid"][-1])
        self.assertTrue(result["valid"][3])
        self.assertEqual(result["metrics"]["conflicting_turns"], 1)

    def test_conflicting_final_copy_is_not_a_trusted_outcome(self):
        rows = indexed(rollout("a", [1], 1), rollout("b", [-1], 0))
        rows.append(dict(rows[2], index=100, outcome_reward=0))
        result = IGPO.compute_igpo_advantages(rows)
        self.assertEqual(result["metrics"]["rollouts_trusted"], 1)
        self.assertFalse(any(result["valid"]))

    def test_outcome_population_has_one_vote_for_different_rollout_lengths(self):
        rows = indexed(rollout("a", [1, 2, 3], 1), rollout("b", [-1], 0))
        result = IGPO.compute_igpo_advantages(rows)
        expected = 0.5 / (0.5 + 1e-6)
        self.assertAlmostEqual(result["outcome_z"][1], expected)
        self.assertAlmostEqual(result["outcome_z"][6], -expected)

    def test_trusted_answer_only_rollout_counts_once_in_outcome_population(self):
        rows = indexed(rollout("a", [1], 1), rollout("b", [-1], 0),
                       rollout("c", [], 1, analysis=False))
        result = IGPO.compute_igpo_advantages(rows)
        mean = 2 / 3
        std = math.sqrt(2 / 9)
        self.assertAlmostEqual(result["outcome_z"][1], (1 - mean) / (std + 1e-6))
        self.assertAlmostEqual(result["outcome_z"][4], -mean / (std + 1e-6))
        self.assertEqual(result["metrics"]["rollouts_trusted"], 3)

    def test_untrusted_or_missing_final_never_uses_ig_only_fallback(self):
        for mutation in ("missing", "untrusted", "invalid", "wrong_type", "nonfinite"):
            with self.subTest(mutation=mutation):
                first = rollout("a", [1, 3], 1)
                if mutation == "missing":
                    first[-1]["is_final"] = False
                elif mutation == "untrusted":
                    first[-1]["outcome_valid"] = False
                elif mutation == "invalid":
                    first[-1]["valid"] = False
                elif mutation == "wrong_type":
                    first[-1]["turn_type"] = "tool_plan"
                else:
                    first[-1]["outcome_reward"] = math.nan
                result = IGPO.compute_igpo_advantages(indexed(first, rollout("b", [-1, 1])))
                self.assertFalse(any(result["valid"]))

    def test_multiple_real_final_flags_fail_closed(self):
        first = rollout("a", [1, 3], 1)
        first[1]["is_final"] = True
        result = IGPO.compute_igpo_advantages(indexed(first, rollout("b", [-1, 1])))
        self.assertFalse(any(result["valid"]))

    def test_post_final_output_cannot_enter_the_future_or_statistics(self):
        rows = indexed(rollout("a", [1, 3], 1), rollout("b", [-1, 1]))
        baseline = IGPO.compute_igpo_advantages(rows)
        rows.append(dict(rows[1], index=100, turn_index=4, gt_logp=-100,
                         external_error=True, action_key="suffix"))
        result = IGPO.compute_igpo_advantages(rows)
        for position in (1, 2, 5, 6):
            self.assertEqual(result["advantage"][position], baseline["advantage"][position])
        self.assertFalse(result["valid"][-1])
        self.assertFalse(result["blocked"][-1])

    def test_analysis_does_not_enter_ig_population_but_occupies_discount_distance(self):
        first = rollout("a", [1, 3], 1, analysis=False)
        second = rollout("b", [-1, 1], 0, analysis=False)
        for sequence in (first, second):
            sequence[1]["turn_index"] = 2
            sequence[2]["turn_index"] = 3
            sequence.insert(1, dict(sequence[0], turn_index=1, turn_type="analysis",
                                    gt_logp=sequence[1]["gt_logp"], action_key=None))
        result = IGPO.compute_igpo_advantages(indexed(first, second), {"igpo_gamma": 0.5})
        z = 2 / (math.sqrt(2) + 1e-6)
        out = 0.5 / (0.5 + 1e-6)
        self.assertAlmostEqual(result["advantage"][0], 0.25 * z + 0.125 * out)
        self.assertTrue(math.isnan(result["raw_ig"][1]))
        self.assertTrue(math.isnan(result["advantage"][1]))

    def test_input_order_does_not_change_the_original_turn_sequence(self):
        rows = indexed(rollout("a", [1, 3], 1), rollout("b", [-1, 1], 0))
        baseline = IGPO.compute_igpo_advantages(rows)
        random.Random(17).shuffle(rows)
        result = IGPO.compute_igpo_advantages(rows)
        for position, row in enumerate(rows):
            expected = baseline["advantage"][row["index"]]
            if math.isfinite(expected):
                self.assertEqual(result["advantage"][position], expected)
            else:
                self.assertTrue(math.isnan(result["advantage"][position]))

    def test_different_questions_never_share_statistics(self):
        first = indexed(rollout("a", [1], 1), rollout("b", [-1], 0))
        second = indexed(rollout("a", [2], 0, uid="other"),
                         rollout("b", [-2], 0, uid="other"))
        result = IGPO.compute_igpo_advantages(first + second)
        out = 0.5 / (0.5 + 1e-6)
        self.assertAlmostEqual(result["outcome_z"][1], out)
        self.assertEqual(result["outcome_z"][7], 0)
        self.assertEqual(result["metrics"]["groups_valid"], 2)

    def test_malformed_original_turn_does_not_become_a_baseline(self):
        for bad_turn in (True, math.nan, 1.5, -1, "bad"):
            with self.subTest(bad_turn=bad_turn):
                first = rollout("a", [1, 3], 1)
                first[1]["turn_index"] = bad_turn
                result = IGPO.compute_igpo_advantages(indexed(first, rollout("b", [-1, 1])))
                self.assertFalse(any(result["valid"]))

    def test_nonfinite_or_positive_gt_state_cannot_supply_ig(self):
        for bad_score in (math.nan, math.inf, 0.1, None):
            with self.subTest(bad_score=bad_score):
                first = rollout("a", [1, 3], 1)
                first[-1]["gt_logp"] = bad_score
                result = IGPO.compute_igpo_advantages(indexed(first, rollout("b", [-1, 1])))
                self.assertFalse(any(result["valid"]))

    def test_overflowing_weighted_return_fails_closed(self):
        first = rollout("a", [0, 0], 0)
        second = rollout("b", [0, 0], 0)
        for sequence, scores in ((first, [-1.7e308, -1.7e308, -8e307, -1e307]),
                                 (second, [-1e307, -1e307, -8e307, -1.7e308])):
            for row, score in zip(sequence, scores):
                row["gt_logp"] = score
        result = IGPO.compute_igpo_advantages(
            indexed(first, second), {"igpo_ig_weight": 1e308, "igpo_outcome_weight": 0})
        self.assertFalse(result["valid"][1])
        self.assertFalse(result["valid"][5])
        self.assertTrue(result["valid"][2])
        self.assertTrue(result["valid"][6])
        for field in ("ig_return", "terminal_return", "advantage"):
            self.assertFalse(any(math.isinf(value) for value in result[field]))

    def test_invalid_configuration_is_rejected(self):
        for config in ({"igpo_gamma": -1}, {"igpo_gamma": 0}, {"igpo_gamma": 2}, {"igpo_ig_weight": -1},
                       {"igpo_outcome_weight": math.inf}, {"igpo_norm_epsilon": 0},
                       {"igpo_ig_std_min": -1}, {"igpo_ig_std_min": 0},
                       {"igpo_min_rollouts": True}, {"igpo_min_rollouts": 1},
                       {"igpo_ig_weight": 0, "igpo_outcome_weight": 0}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    IGPO.compute_igpo_advantages([], config)


if __name__ == "__main__":
    unittest.main()
