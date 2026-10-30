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


class ScoringTests(unittest.TestCase):
    def run_score(self, snapshot=None, probe=None, config=None, scorer=None):
        scorer = scorer or FakeScorer()
        with mock.patch.object(scoring, "_get_scorer", return_value=scorer):
            result = scoring.score_snapshots(
                snapshot or saved(), {"prompt": "predict outcome", "response": "Success.", **(probe or {})},
                {"backend": "vllm", "max_context_tokens": 32, **(config or {})},
            )
        return result, scorer

    def test_trace_nll_formula_response_only_and_last_action(self):
        result, backend = self.run_score()
        trajectory = result["trajectories"][0]
        values = [math.log(4.1) - math.log(-mean + 0.1) for mean in (-4.0, -2.0, -1.0, -3.0)]
        deltas = [values[i + 1] - values[i] for i in range(3)]
        # K=3 includes all three real transitions, including the terminal post-state.
        expected = (deltas[0] + .8 * deltas[1] + .8 ** 2 * deltas[2]) / (1 + .8 + .8 ** 2) + 2 * .8 ** 2
        self.assertAlmostEqual(trajectory["turns"][0]["credit"], expected)
        self.assertAlmostEqual(trajectory["turns"][-1]["credit"], deltas[-1] + 2)
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
        trajectory = snapshot["trajectories"][0]
        trajectory.update(goal_response="A cleaned mug is on a table.", goal_spec_hash="same-game")
        for state in trajectory["states"]:
            state.update(state_view="full_state", goal_check_valid=True, scoring_valid=True,
                         goal_check=state["state_index"] == 3)
        return snapshot

    def test_deterministic_goal_fixed_at_every_turn_and_reused(self):
        result, backend = self.run_score(self.full_state_saved(),
                                       probe={"response": "{goal_response}", "state_view": "full_state"})
        trajectory = result["trajectories"][0]
        self.assertEqual({state["response"] for state in trajectory["states"]},
                         {"A cleaned mug is on a table."})
        self.assertEqual(trajectory["turns"][-1]["trace_turn_type"], "tool_plan")
        self.assertTrue(trajectory["turns"][-1]["outcome_valid"])
        self.assertEqual(trajectory["turns"][0]["terminal_fill"], 2 * .8 ** 2)

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
        self.assertAlmostEqual(success[0]["terminal_fill"], .8 ** 2)
        self.assertAlmostEqual(failed[0]["terminal_fill"], -.8 ** 2)
        self.assertEqual(failed[-1]["label_source"], "trace")
        self.assertAlmostEqual(success[-1]["terminal_fill"], 1)
        self.assertAlmostEqual(failed[-1]["terminal_fill"], -1)

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
        self.assertAlmostEqual(turn["terminal_fill"], 2)
        self.assertAlmostEqual(turn["credit"], delta + 2)
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
        for turn, expected in zip(turns, (0, 2 * .8 ** 2, 2 * .8, 2)):
            self.assertAlmostEqual(turn["terminal_fill"], expected)
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
            rows.append({"index": index, "uid": "same-question", "turn_type": turn["trace_turn_type"],
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
