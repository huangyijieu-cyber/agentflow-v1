"""CPU-only fixed-response scoring and TRACE boundary regression checks."""

import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


PATH = Path(__file__).resolve().parents[2] / "train-roma/alfworld_trace_probe_scoring.py"
SPEC = importlib.util.spec_from_file_location("alfworld_probe_scoring_test_subject", PATH)
scoring = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scoring
SPEC.loader.exec_module(scoring)


class FakeTokenizer:
    def __init__(self):
        self.contents = {}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [80, 81]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        content = messages[0]["content"]
        self.contents.setdefault(content, len(self.contents) + 10)
        return [7, self.contents[content]]

    def decode(self, ids, skip_special_tokens=False):
        return "chat-template:" + repr(ids)


class FakeScorer:
    def __init__(self, means=(-4.0, -2.0, -1.0, -3.0, -2.5)):
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.means = means

    def score(self, prompt_ids, target_ids):
        self.calls.append((list(prompt_ids), list(target_ids)))
        content = next(key for key, value in self.tokenizer.contents.items() if value == prompt_ids[-1])
        index = int(content.split("\n", 1)[0][1:])
        return [self.means[index]] * len(target_ids)


def saved(count=3):
    return {
        "snapshot_id": "saved-first-sampling",
        "trajectories": [{
            "trajectory_id": "traj_01", "goal": "clean a mug",
            "outcome": {"success": True, "reward": 1.0, "valid": True, "terminated": True, "truncated": False},
            "states": [{"state_index": i, "state_context": f"s{i}", "remaining_steps": 50 - i,
                        "is_post_state": i == count} for i in range(count + 1)],
            "turns": [{"turn_index": i, "action_prompt": f"action input {i}",
                       "action_response": f"<action>action {i}</action>", "external_error": False,
                       "done": i == count - 1, "finish_reason": "stop"}
                      for i in range(count)],
        }],
    }


def value_snapshot(fixtures):
    """Construct exact V paths without relying on terminal-fill implementation."""
    snapshot, means = {"snapshot_id": "controlled-values", "trajectories": []}, []
    for uid, identity, values, reward, trusted in fixtures:
        trajectory = saved(len(values) - 1)["trajectories"][0]
        trajectory.update(uid=uid, trajectory_id=identity)
        trajectory["outcome"].update(reward=reward, success=reward > .5, valid=trusted)
        for state, value in zip(trajectory["states"], values):
            state["state_context"] = f"s{len(means)}"
            means.append(-(4.1 * math.exp(-value) - .1))
        snapshot["trajectories"].append(trajectory)
    return snapshot, FakeScorer(means)


class ScoringTests(unittest.TestCase):
    def run_score(self, snapshot=None, probe=None, config=None, scorer=None):
        scorer = scorer or FakeScorer()
        with mock.patch.object(scoring, "_get_scorer", return_value=scorer):
            result = scoring.score_snapshots(
                snapshot or saved(), {"prompt": "predict outcome", "response": "Success.", **(probe or {})},
                {"backend": "vllm", "max_context_tokens": 32, **(config or {})},
            )
        return result, scorer

    def history_cache(self, *, legacy=False):
        snapshot = saved(6)
        trajectory = snapshot["trajectories"][0]
        spec = {"objects": {}, "initial_facts": [], "rules_context": "Applicable rules."}
        trajectory["goal_spec"] = spec
        module = scoring._goal_module()
        records = [{"action": f"act {i}", "observation": f"after {i}", "executed": True}
                   for i in range(6)]
        for state in trajectory["states"]:
            index = state["state_index"]
            state.update(facts=[], observation=f"visible {index}", state_view="full_state",
                         scoring_valid=True, goal_check_valid=True, goal_check=index == 6,
                         trace_recent_action_history=records[:index][-4:])
            state["state_context"] = f"Task: {trajectory['goal']}\n\n" + module.render_full_state(
                spec, [], observation=state["observation"], action_history=records[:index])
            state["scoring_context_version"] = (module.RECENT_ACTION_HISTORY_SCORING_VERSION if legacy
                                                 else module.ACTION_HISTORY_SCORING_VERSION)
            if not legacy:
                state.update(trace_full_action_history=records[:index], trace_history_length=4)
        return snapshot, records

    def test_direct_full_state_rescore_switches_history_windows_without_mutating_cache(self):
        snapshot, records = self.history_cache()
        original = json.dumps(snapshot, sort_keys=True)
        for window in (None, 0, 1, 6, 20, 4):
            with self.subTest(window=window):
                backend = SimpleNamespace(tokenizer=FakeTokenizer(), score=mock.Mock(return_value=[-2., -2.]))
                result, _ = self.run_score(snapshot, scorer=backend,
                                          probe={"state_view": "full_state", "history_length": window})
                terminal = result["trajectories"][0]["states"][-1]
                expected = records if window is None else records[-window:] if window else []
                self.assertEqual(terminal["trace_recent_action_history"], expected)
                self.assertEqual(terminal["trace_full_action_history"], records)
                if expected:
                    self.assertIn(json.dumps(expected, sort_keys=True), terminal["scoring_prompt"])
                else:
                    self.assertNotIn("Recorded environment action history:", terminal["scoring_prompt"])
                self.assertEqual(result["formula"]["history_length"], window)
                self.assertTrue(result["formula"]["history_window_applied"])
                self.assertEqual(json.dumps(snapshot, sort_keys=True), original)

    def test_legacy_overlapping_windows_recover_all_history_and_reject_conflicts(self):
        snapshot, records = self.history_cache(legacy=True)
        restored = scoring.prepare_history_contexts(snapshot, None)
        terminal = restored["trajectories"][0]["states"][-1]
        self.assertEqual(terminal["trace_full_action_history"], records)
        self.assertEqual(terminal["trace_recent_action_history"], records)
        # Deep-copy before corrupting a record that overlaps a previous state.
        corrupt = json.loads(json.dumps(snapshot))
        corrupt["trajectories"][0]["states"][4]["trace_recent_action_history"][0]["action"] = "different"
        with self.assertRaisesRegex(ValueError, "Conflicting.*enrich --replay"):
            scoring.prepare_history_contexts(corrupt, None)
        missing = json.loads(json.dumps(snapshot))
        missing["trajectories"][0]["states"] = missing["trajectories"][0]["states"][-1:]
        with self.assertRaisesRegex(ValueError, "not fully cached.*enrich --replay"):
            scoring.prepare_history_contexts(missing, None)
        short = scoring.prepare_history_contexts(missing, 2)
        self.assertEqual(short["trajectories"][0]["states"][0]["trace_recent_action_history"], records[-2:])

    def test_zero_history_needs_no_saved_records_and_saved_view_keeps_literal_context(self):
        snapshot, _ = self.history_cache()
        for state in snapshot["trajectories"][0]["states"]:
            for key in ("trace_full_action_history", "trace_recent_action_history", "trace_history_length",
                        "scoring_context_version"):
                state.pop(key, None)
        empty = scoring.prepare_history_contexts(snapshot, 0)
        self.assertTrue(all("Recorded environment action history:" not in state["state_context"]
                            for state in empty["trajectories"][0]["states"]))
        backend = SimpleNamespace(tokenizer=FakeTokenizer(), score=mock.Mock(return_value=[-2., -2.]))
        result, _ = self.run_score(snapshot, scorer=backend, probe={"state_view": "saved", "history_length": None})
        self.assertFalse(result["formula"]["history_window_applied"])
        for original, actual in zip(snapshot["trajectories"][0]["states"], result["trajectories"][0]["states"]):
            self.assertEqual(actual["state_context"], original["state_context"])

    def test_trace_nll_formula_response_only_and_last_action(self):
        result, backend = self.run_score()
        trajectory = result["trajectories"][0]
        values = [math.log(4.1) - math.log(-mean + 0.1) for mean in (-4.0, -2.0, -1.0, -3.0)]
        deltas = [values[i + 1] - values[i] for i in range(3)]
        # K=3 includes all three real transitions, including the terminal post-state.
        local = [(deltas[0] + .8 * deltas[1] + .8 ** 2 * deltas[2]) / (1 + .8 + .8 ** 2),
                 (deltas[1] + .8 * deltas[2]) / 1.8, deltas[-1]]
        scaled = max(0, *local)  # One trusted success maps to the question's positive extent.
        self.assertAlmostEqual(trajectory["turns"][0]["credit"], local[0] + scaled * .8 ** 2)
        self.assertAlmostEqual(trajectory["turns"][-1]["credit"], deltas[-1] + scaled)
        self.assertEqual(trajectory["turns"][-1]["label_source"], "trace")
        self.assertAlmostEqual(trajectory["turns"][-1]["delta"], deltas[-1])
        self.assertAlmostEqual(trajectory["turns"][-1]["diagnostic_post_delta"], deltas[-1])
        self.assertEqual(len(trajectory["turns"]), 3)
        self.assertEqual(len(trajectory["states"]), 4)
        self.assertEqual(trajectory["states"][0]["scoring_prompt"], "s0\n\npredict outcome")
        self.assertEqual(trajectory["states"][0]["mean_logp"], -4)
        self.assertEqual(trajectory["states"][0]["token_logps"], [-4, -4])
        self.assertEqual(backend.calls[0][1], [80, 81])
        self.assertEqual(trajectory["turns"][0]["action_response"], "<action>action 0</action>")
        # The environment's terminal success must not overwrite the model score.
        self.assertAlmostEqual(trajectory["states"][-1]["value"], values[-1])
        self.assertNotEqual(trajectory["states"][-1]["value"], 1)
        json.dumps(result, allow_nan=False)

    def test_k_one_uses_immediate_transition(self):
        result, _ = self.run_score(probe={"k": 1, "terminal_fill_enabled": False})
        for turn in result["trajectories"][0]["turns"]:
            self.assertAlmostEqual(turn["credit"], turn["delta"])
            self.assertEqual(turn["terminal_fill"], 0)

    def test_external_failure_blocks_every_window_containing_it(self):
        snapshot = saved()
        snapshot["trajectories"][0]["turns"][1]["external_error"] = True
        result, _ = self.run_score(snapshot, probe={"k": 2})
        turns = result["trajectories"][0]["turns"]
        self.assertIsNone(turns[0]["credit"])
        self.assertIsNone(turns[1]["credit"])
        self.assertIsNone(turns[1]["delta"])
        self.assertAlmostEqual(turns[2]["credit"], turns[2]["delta"])
        self.assertEqual(turns[2]["terminal_fill"], 0)
        self.assertEqual(result["summary"]["external_error_windows"], 2)

    def test_missing_state_does_not_bridge_gap(self):
        snapshot = saved(count=4)
        snapshot["trajectories"][0]["states"][1]["state_context"] = ""
        result, _ = self.run_score(snapshot, probe={"k": 1})
        turns = result["trajectories"][0]["turns"]
        self.assertIsNone(turns[0]["credit"])
        self.assertIsNone(turns[1]["credit"])
        self.assertIsNotNone(turns[2]["credit"])
        self.assertEqual(result["summary"]["invalid_scored_states"], 1)

    def test_missing_initial_state_invalidates_entire_value_baseline(self):
        snapshot = saved()
        snapshot["trajectories"][0]["states"][0]["state_context"] = None
        result, _ = self.run_score(snapshot)
        turns = result["trajectories"][0]["turns"]
        self.assertTrue(all(turn["credit"] is None for turn in turns))
        self.assertEqual(turns[0]["credit_error"], "invalid_initial_state_value")

    def test_external_post_state_marked_invalid_is_not_sent_to_backend(self):
        snapshot = saved()
        trajectory = snapshot["trajectories"][0]
        trajectory["states"][-1].update({"scoring_valid": False, "env_info": {"env_error": True}})
        trajectory["turns"][-1]["external_error"] = True
        result, backend = self.run_score(snapshot)
        scored = result["trajectories"][0]
        last = scored["states"][-1]
        self.assertEqual(len(backend.calls), 3)
        self.assertFalse(any(content.startswith("s3\n") for content in backend.tokenizer.contents))
        self.assertFalse(last["valid"])
        self.assertEqual(last["error"], "external_environment_error_state")
        self.assertIsNone(last["mean_logp"])
        self.assertIsNone(last["value"])
        self.assertTrue(all(turn["credit"] is None for turn in scored["turns"]))
        self.assertEqual(result["summary"]["invalid_scored_states"], 1)

    def test_discontinuous_original_state_numbers_do_not_join(self):
        snapshot = saved(count=4)
        snapshot["trajectories"][0]["states"][1]["state_index"] = 9
        result, _ = self.run_score(snapshot, probe={"k": 1})
        turns = result["trajectories"][0]["turns"]
        self.assertIsNone(turns[0]["credit"])
        self.assertIsNone(turns[0]["delta"])
        self.assertIsNone(turns[1]["credit"])
        self.assertIsNotNone(turns[2]["credit"])

    def test_same_probe_deduplicates_but_new_prompt_rescores(self):
        snapshot = saved()
        snapshot["trajectories"].append(json.loads(json.dumps(snapshot["trajectories"][0])))
        snapshot["trajectories"][1]["trajectory_id"] = "traj_02"
        backend = FakeScorer()
        result, _ = self.run_score(snapshot, scorer=backend)
        self.assertEqual(len(backend.calls), 4)
        self.assertEqual(result["summary"]["unique_scoring_requests"], 4)
        self.run_score(snapshot, probe={"prompt": "new outcome instruction"}, scorer=backend)
        self.assertEqual(len(backend.calls), 8)

    def test_overlong_input_and_target_are_rejected_not_truncated(self):
        result, backend = self.run_score(config={"max_context_tokens": 3})
        self.assertFalse(backend.calls)
        self.assertEqual(result["summary"]["valid_credit_turns"], 0)
        self.assertEqual(result["trajectories"][0]["states"][0]["error"], "context_too_long_no_truncation")
        result, backend = self.run_score(probe={"max_response_tokens": 1})
        self.assertFalse(backend.calls)
        self.assertEqual(result["trajectories"][0]["states"][0]["error"], "invalid_response_token_length")

    def test_response_is_fixed_per_goal_prompt_can_use_state_placeholders(self):
        result, _ = self.run_score(probe={"response": "Complete {goal}.",
                                              "prompt": "Goal: {goal}; steps: {remaining_steps}; turn: {turn_index}; JSON: {}"})
        states = result["trajectories"][0]["states"]
        self.assertEqual(states[0]["response"], "Complete clean a mug.")
        self.assertEqual(states[1]["response"], states[0]["response"])
        self.assertIn("steps: 49; turn: 1; JSON: {}", states[1]["scoring_prompt"])
        with self.assertRaisesRegex(ValueError, "stay fixed"):
            self.run_score(probe={"response": "Success within {remaining_steps}."})

    def test_nonfinite_or_positive_logps_fail_closed(self):
        for value in (math.nan, math.inf, .2):
            with self.subTest(value=value), self.assertRaises(scoring.ProbeScoringError):
                self.run_score(scorer=FakeScorer(means=[value] * 4))

    def test_invalid_trace_configuration_rejected(self):
        for probe in ({"k": 0}, {"epsilon": 0}, {"gamma": 1.2}):
            with self.subTest(probe=probe), self.assertRaises(ValueError):
                self.run_score(probe=probe)

    def full_state_saved(self):
        snapshot = saved()
        snapshot["sampling"] = {"history_length": 4}
        trajectory = snapshot["trajectories"][0]
        trajectory.update(goal_response="A cleaned mug is on a table.", goal_spec_hash="same-game")
        for state in trajectory["states"]:
            state.update(state_view="full_state", goal_check_valid=True, scoring_valid=True,
                         goal_check=state["state_index"] == 3,
                         trace_recent_action_history=[{"action": f"action {i}", "observation": f"observed {i}"}
                                                      for i in range(state["state_index"])][-4:],
                         scoring_context_version=scoring._recent_action_history_version())
        return snapshot

    def test_full_state_rejects_old_or_malformed_post_action_history_before_scoring(self):
        valid_record = {"action": "act", "observation": "result"}
        for change in ({"trace_recent_action_history": None}, {"trace_recent_action_history": "native pre-action text"},
                       {"trace_recent_action_history": [valid_record] * 5},
                       {"trace_recent_action_history": [{"action": "act"}]},
                       {"trace_recent_action_history": [{"action": 4, "observation": "result"}]},
                       {"scoring_context_version": "alfworld-actor-recent-history-v1"},
                       {"scoring_context_version": "old-no-history"}):
            with self.subTest(change=change):
                snapshot = self.full_state_saved()
                snapshot["trajectories"][0]["states"][0].update(change)
                backend = FakeScorer()
                with self.assertRaisesRegex(ValueError, "recent action history.*enrich --replay"):
                    self.run_score(snapshot, probe={"state_view": "full_state"}, scorer=backend)
                self.assertEqual(backend.calls, [])

    def test_saved_view_can_score_legacy_literal_full_state_context(self):
        snapshot = self.full_state_saved()
        for state in snapshot["trajectories"][0]["states"]:
            for key in ("trace_recent_action_history", "scoring_context_version"):
                state.pop(key)
        result, backend = self.run_score(snapshot, probe={"state_view": "saved"})
        self.assertEqual(len(backend.calls), 4)
        self.assertTrue(all(state["valid"] for state in result["trajectories"][0]["states"]))

    def test_invalid_external_tail_without_full_facts_does_not_block_prefix_scoring(self):
        snapshot = self.full_state_saved()
        tail = snapshot["trajectories"][0]["states"][-1]
        tail.update(state_view="legacy_observed", scoring_valid=False,
                    env_info={"env_error": True}, trace_recent_action_history=None,
                    scoring_context_version=None)
        result, backend = self.run_score(snapshot, probe={"state_view": "full_state"})
        states = result["trajectories"][0]["states"]
        self.assertEqual(len(backend.calls), 3)
        self.assertTrue(all(state["valid"] for state in states[:-1]))
        self.assertFalse(states[-1]["valid"])

    def test_deterministic_goal_fixed_at_every_turn_and_reused(self):
        result, backend = self.run_score(self.full_state_saved(),
                                       probe={"response": "{goal_response}", "state_view": "full_state"})
        trajectory = result["trajectories"][0]
        self.assertEqual({state["response"] for state in trajectory["states"]},
                         {"A cleaned mug is on a table."})
        self.assertEqual(trajectory["turns"][-1]["trace_turn_type"], "tool_plan")
        self.assertTrue(trajectory["turns"][-1]["outcome_valid"])
        positive_extent = max(0, *(turn["local_credit"] for turn in trajectory["turns"]))
        self.assertAlmostEqual(trajectory["turns"][0]["terminal_fill"], positive_extent * .8 ** 2)

    def test_invalid_terminal_poststate_keeps_action_type_and_blocks_credit_and_fill(self):
        snapshot = self.full_state_saved()
        snapshot["trajectories"][0]["states"][-1].update(goal_check_valid=False, scoring_valid=False)
        result, backend = self.run_score(snapshot, probe={"state_view": "full_state"})
        trajectory = result["trajectories"][0]
        self.assertEqual(len(backend.calls), 3)
        final = trajectory["turns"][-1]
        self.assertEqual(final["trace_turn_type"], "tool_plan")
        self.assertFalse(final["is_final"])
        self.assertTrue(final["is_terminal"])
        self.assertFalse(final["outcome_valid"])
        self.assertTrue(all(turn["terminal_fill"] is None for turn in trajectory["turns"]))
        self.assertTrue(all(turn["credit"] is None for turn in trajectory["turns"]))

    def test_truncated_environment_budget_is_terminal_failure_population(self):
        snapshot = self.full_state_saved()
        failed = json.loads(json.dumps(snapshot["trajectories"][0]))
        failed["trajectory_id"] = "traj_failed"
        failed["outcome"].update(success=False, reward=0.0, terminated=False, truncated=True)
        failed["states"][-1]["goal_check"] = False
        snapshot["trajectories"].append(failed)
        result, _ = self.run_score(snapshot, probe={"state_view": "full_state"})
        success, failed = [t["turns"] for t in result["trajectories"]]
        self.assertEqual(success[-1]["outcome_advantage"], .5)
        self.assertEqual(failed[-1]["outcome_advantage"], -.5)
        positive_extent = max(0, *(turn["local_credit"] for turn in success))
        negative_extent = min(0, *(turn["local_credit"] for turn in success))
        self.assertAlmostEqual(success[0]["terminal_fill"], positive_extent * .8 ** 2)
        self.assertAlmostEqual(failed[0]["terminal_fill"], negative_extent * .8 ** 2)
        self.assertEqual(failed[-1]["label_source"], "trace")
        self.assertAlmostEqual(success[-1]["terminal_fill"], positive_extent)
        self.assertAlmostEqual(failed[-1]["terminal_fill"], negative_extent)

    def test_model_length_stop_has_environment_boundary_without_trusted_terminal_fill(self):
        snapshot = self.full_state_saved()
        snapshot["trajectories"][0]["turns"][-1]["finish_reason"] = "length"
        result, _ = self.run_score(snapshot)
        final = result["trajectories"][0]["turns"][-1]
        self.assertEqual(final["trace_turn_type"], "tool_plan")
        self.assertFalse(final["is_final"])
        self.assertFalse(final["is_terminal"])
        self.assertTrue(final["environment_terminal"])
        self.assertFalse(final["outcome_valid"])
        self.assertEqual(final["terminal_fill"], 0)

    def test_single_action_uses_real_terminal_state_without_creating_a_final_turn(self):
        snapshot = saved(count=1)
        result, _ = self.run_score(snapshot)
        trajectory = result["trajectories"][0]
        self.assertEqual(len(trajectory["turns"]), 1)
        self.assertEqual(len(trajectory["states"]), 2)
        turn = trajectory["turns"][0]
        delta = math.log(4.1) - math.log(2.1)
        self.assertAlmostEqual(turn["delta"], delta)
        self.assertAlmostEqual(turn["local_credit"], delta)
        self.assertAlmostEqual(turn["terminal_fill"], delta)
        self.assertAlmostEqual(turn["credit"], 2 * delta)
        self.assertTrue(turn["is_terminal"])
        self.assertFalse(turn["is_final"])
        self.assertEqual(turn["trace_turn_type"], "tool_plan")
        self.assertEqual(turn["label_source"], "trace")
        self.assertEqual(result["summary"]["outcome_turns"], 0)
        self.assertEqual(result["summary"]["terminal_actions"], 1)

    def test_k_three_fills_last_three_actual_actions_including_terminal_action(self):
        result, _ = self.run_score(saved(count=4))
        turns = result["trajectories"][0]["turns"]
        self.assertEqual([t["terminal_reached"] for t in turns], [False, True, True, True])
        positive_extent = max(0, *(turn["local_credit"] for turn in turns))
        for turn, discount in zip(turns, (0, .8 ** 2, .8, 1)):
            self.assertAlmostEqual(turn["terminal_fill"], positive_extent * discount)
        self.assertTrue(all(t["label_source"] == "trace" for t in turns))

    def test_missing_terminal_poststate_never_fabricates_delta_or_credit(self):
        snapshot = saved()
        snapshot["trajectories"][0]["states"][-1]["state_context"] = ""
        result, backend = self.run_score(snapshot)
        turns = result["trajectories"][0]["turns"]
        self.assertEqual(len(backend.calls), 3)
        self.assertTrue(all(t["credit"] is None for t in turns))
        self.assertTrue(all(t["terminal_fill"] is None for t in turns))
        self.assertIsNone(turns[-1]["delta"])
        self.assertEqual(turns[-1]["label_source"], "trace")

    def test_dynamic_fill_matches_question_ranges_and_ignores_old_scale(self):
        fixtures = [("q", "a", [0, .2], 1., True), ("q", "b", [0, .1], 1., True),
                    ("q", "c", [0, .05], 1., True), ("q", "d", [0, -.08], 0., True)]
        for scale in (0, 2, 100):
            for normalize in (False, True):
                with self.subTest(scale=scale, normalize=normalize):
                    snapshot, backend = value_snapshot(fixtures)
                    original = json.dumps(snapshot, sort_keys=True)
                    result, _ = self.run_score(snapshot, scorer=backend,
                                              probe={"terminal_fill_scale": scale,
                                                     "norm_adv_by_std_in_grpo": normalize})
                    for trajectory, expected in zip(result["trajectories"], (.2, .2, .2, -.08)):
                        turn = trajectory["turns"][0]
                        self.assertAlmostEqual(turn["scaled_outcome_advantage"], expected)
                        self.assertAlmostEqual(turn["terminal_fill"], expected)
                    self.assertEqual(json.dumps(snapshot, sort_keys=True), original)
                    self.assertEqual(result["credit_mode"], "alfworld-environment-dynamic-terminal-fill-v1")
                    self.assertIn("terminal_fill_scale_ignored", result["formula"])

    def test_dynamic_ranges_include_untrusted_local_windows_and_isolate_questions(self):
        snapshot, backend = value_snapshot([
            ("q", "a", [0, .1], 1., True), ("q", "b", [0, -.08], 0., True),
            ("q", "prefix", [0, .8], 100., False),
            ("other", "c", [0, 2.], 1., True), ("other", "d", [0, -1.], 0., True)])
        result, _ = self.run_score(snapshot, scorer=backend)
        turns = [trajectory["turns"][0] for trajectory in result["trajectories"]]
        for turn, expected in zip(turns, (.8, -.08, 0, 2, -1)):
            self.assertAlmostEqual(turn["terminal_fill"], expected)
        self.assertIsNone(turns[2]["scaled_outcome_advantage"])
        self.assertTrue(turns[2]["range_eligible"])
        self.assertEqual([turns[0]["outcome_advantage"], turns[1]["outcome_advantage"]], [.5, -.5])

    def test_invalid_actions_do_not_set_dynamic_credit_range(self):
        for field, value in (("format_error", True), ("invalid_action", True),
                             ("external_error", True), ("finish_reason", "length"),
                             ("trace_valid", False), ("trace_dup", True),
                             ("is_drop_mask", True), ("response_token_ids", []),
                             ("action_response", "")):
            with self.subTest(field=field):
                snapshot, backend = value_snapshot([
                    ("q", "a", [0, .2], 1., True), ("q", "b", [0, -.08], 0., True),
                    ("q", "bad", [0, 1.5], 1., True)])
                snapshot["trajectories"][2]["turns"][0][field] = value
                result, _ = self.run_score(snapshot, scorer=backend)
                turns = [trajectory["turns"][0] for trajectory in result["trajectories"]]
                self.assertAlmostEqual(turns[0]["terminal_fill"], .2)
                self.assertAlmostEqual(turns[1]["terminal_fill"], -.08)
                self.assertFalse(turns[2]["range_eligible"])

    def test_same_context_action_duplicates_cannot_set_dynamic_range(self):
        snapshot, backend = value_snapshot([
            ("q", "a", [0, .2, 0, 1.], 1., True), ("q", "b", [0, -.1], 0., True)])
        repeated = snapshot["trajectories"][0]
        repeated["states"][2]["state_context"] = repeated["states"][0]["state_context"]
        repeated["turns"][2]["action_response"] = repeated["turns"][0]["action_response"]
        result, _ = self.run_score(snapshot, scorer=backend, probe={"k": 1})
        positive, negative = result["trajectories"]
        self.assertFalse(positive["turns"][2]["range_eligible"])
        self.assertAlmostEqual(positive["turns"][2]["local_credit"], 1.)
        self.assertAlmostEqual(positive["turns"][2]["terminal_fill"], .2)
        self.assertAlmostEqual(negative["turns"][0]["terminal_fill"], -.2)

    def test_missing_credit_sign_and_disabled_fill_add_zero(self):
        for enabled in (True, False):
            snapshot, backend = value_snapshot([
                ("q", "a", [0, .2], 1., True), ("q", "b", [0, .1], 0., True)])
            result, _ = self.run_score(snapshot, scorer=backend, probe={"terminal_fill_enabled": enabled})
            turns = [trajectory["turns"][0] for trajectory in result["trajectories"]]
            self.assertAlmostEqual(turns[0]["terminal_fill"], .2 if enabled else 0)
            self.assertEqual(turns[1]["terminal_fill"], 0)

    def test_terminal_actions_form_the_same_credit_pairs_as_other_actions(self):
        snapshot = saved(count=1)
        success = snapshot["trajectories"][0]
        failed = json.loads(json.dumps(success))
        failed["trajectory_id"] = "traj_failed"
        failed["outcome"].update(success=False, reward=0.0)
        failed["states"][-1]["state_context"] = "s2"
        failed["turns"][0]["action_response"] = "<action>wrong action</action>"
        snapshot["trajectories"].append(failed)
        result, _ = self.run_score(snapshot, scorer=FakeScorer(means=(-4., -2., -6.)))
        rows = []
        for index, trajectory in enumerate(result["trajectories"]):
            turn = trajectory["turns"][0]
            rows.append({"index": index, "uid": "same-question", "turn_index": turn["turn_index"],
                         "turn_type": turn["trace_turn_type"],
                         "value": turn["value"], "credit": turn["credit"], "label_source": turn["label_source"],
                         "length": 1, "old_per_token": -1., "valid": True,
                         "action_key": turn["action_response"], "is_final": turn["is_final"]})
        self.assertGreater(rows[0]["credit"], 0)
        self.assertLess(rows[1]["credit"], 0)
        self.assertEqual(scoring._trace_module().select_pairs(rows), [(0, 1)])

    def test_missing_or_mixed_goal_metadata_fails_before_requests(self):
        snapshot = self.full_state_saved()
        snapshot["trajectories"][0].pop("goal_spec_hash")
        backend = FakeScorer()
        with self.assertRaisesRegex(ValueError, "goal_spec_hash"):
            self.run_score(snapshot, probe={"response": "{goal_response}"}, scorer=backend)
        self.assertFalse(backend.calls)
        with self.assertRaisesRegex(ValueError, "enrich"):
            self.run_score(saved(), probe={"response": "{goal_response}"})

    def test_observations_cannot_be_labeled_full_state(self):
        backend = FakeScorer()
        with self.assertRaisesRegex(ValueError, "enrich --replay"):
            self.run_score(saved(), probe={"state_view": "full_state"}, scorer=backend)
        self.assertFalse(backend.calls)


class VLLMProtocolTests(unittest.TestCase):
    def backend(self):
        backend = scoring._VLLMScorer.__new__(scoring._VLLMScorer)
        backend.config = {}
        backend.model = "frozen-model"
        backend.base_url = "http://127.0.0.1:19998/v1"
        return backend

    def test_echo_uses_target_prompt_positions_not_generated_tokens(self):
        payload = {"choices": [{"logprobs": {"token_logprobs": [None, -99, -1, -3]}}],
                   "usage": {"prompt_tokens": 4}}
        with mock.patch.object(scoring.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as call:
            result = self.backend().score([10, 11], [20, 21])
        self.assertEqual(result, [-1, -3])
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(body["prompt"], [10, 11, 20, 21])
        self.assertTrue(body["echo"])
        self.assertEqual(body["max_tokens"], 0)
        self.assertEqual(body["temperature"], 1.0)

    def test_generated_only_logps_are_not_accepted(self):
        payload = {"choices": [{"logprobs": {"token_logprobs": [-.01]}}]}
        with mock.patch.object(scoring.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(scoring.ProbeScoringError, "Generated completion"):
                self.backend().score([10, 11], [20, 21])

    def test_server_token_count_mismatch_is_rejected(self):
        payload = {"choices": [{"logprobs": {"token_logprobs": [None, -9, -1, -3]}}],
                   "usage": {"prompt_tokens": 5}}
        with mock.patch.object(scoring.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(scoring.ProbeScoringError, "token count"):
                self.backend().score([10, 11], [20, 21])


class HFForwardTests(unittest.TestCase):
    def test_real_tensor_forward_uses_causal_shift_and_response_tokens_only(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional local-model tensor check requires torch")

        class TinyModel:
            def __call__(self, *, input_ids, attention_mask, use_cache):
                self.input_ids = input_ids.tolist()
                self.attention_mask = attention_mask.tolist()
                self.use_cache = use_cache
                return SimpleNamespace(logits=torch.tensor([[
                    [0., 0., 0., 0.],
                    [0., 0., 4., 0.],
                    [0., 2., 0., 0.],
                    [9., 0., 0., 0.],
                ]]))

        backend = scoring._HFScorer.__new__(scoring._HFScorer)
        backend.torch, backend.device, backend.model = torch, "cpu", TinyModel()
        actual = backend.score([0, 3], [2, 1])
        self.assertAlmostEqual(actual[0], 4 - math.log(math.exp(4) + 3), places=6)
        self.assertAlmostEqual(actual[1], 2 - math.log(math.exp(2) + 3), places=6)
        self.assertEqual(backend.model.input_ids, [[0, 3, 2, 1]])
        self.assertEqual(backend.model.attention_mask, [[1, 1, 1, 1]])
        self.assertIs(backend.model.use_cache, False)


if __name__ == "__main__":
    unittest.main()
