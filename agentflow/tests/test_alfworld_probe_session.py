"""CPU tests for immutable capture, resume, rescore, and exported diagnostics."""
import copy
import importlib.util
import io
import json
from email.message import Message
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "train-roma"
sys.path.insert(0, str(SCRIPTS))
import alfworld_trace_probe as probe
import alfworld_trace_probe_scoring as scoring


class FakeEnv:
    def __init__(self, config):
        self.config = config
        self.task = "Clean a cup and place it on the table."
        self.trace_state_context = "stage=0"
        self.prev_text_obs = "stage=0"
        self.prev_admissible_command = [["act"]]
        self.index = 0

    def reset(self, **kwargs):
        self.index = 0
        self.trace_state_context = "stage=0"
        self.prev_text_obs = "stage=0"
        return "action prompt 0", {"anchor": "stage=0", "admissible_commands": [["act"]],
                                   "extra.gamefile": ["same-game"], "won": [False]}

    def step(self, response):
        self.index += 1
        self.trace_state_context = f"stage={self.index}"
        self.prev_text_obs = self.trace_state_context
        done = self.index == 2
        return f"action prompt {self.index}", float(done), done, {
            "anchor": self.trace_state_context, "admissible_commands": [["act"]],
            "won": [done], "terminated": done, "truncated": False,
            "trace_env_action": "act", "env_error": False,
        }

    def close(self):
        pass


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("USER:" + messages[0]["content"] + "\nASSISTANT:")

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


class FakeScorer:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = 0

    def score(self, prompt_ids, target_ids):
        self.calls += 1
        content = self.tokenizer.decode(prompt_ids)
        nll = 4 if "stage=0" in content else 2 if "stage=1" in content else 1
        if "strict" in content and "stage=1" in content:
            nll = 8
        return [-float(nll)] * len(target_ids)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.experiment = probe.Experiment(self.temp.name, {
            "sampling": {"max_steps": 3},
            "probe": {"prompt": "forecast {remaining_steps}", "response": "success", "k": 1, "state_view": "saved", "terminal_fill_enabled": False},
        })
        self.selection = {"path": "/dataset.parquet", "row_index": 0,
                          "question": "train_all_alfworld_0", "source_row": {"id": 1},
                          "reset": {"data_type": "train", "task_type": "all", "question_id": 0}}

    def tearDown(self):
        self.temp.cleanup()

    def sample(self, policy=None, env_factory=FakeEnv):
        if policy is None:
            policy = lambda *args: {"response": "<action>act</action>", "finish_reason": "stop"}
        with mock.patch.object(probe, "select_row", return_value=self.selection), mock.patch.object(
                probe, "resolve_dataset", return_value=Path(self.selection["path"])):
            return self.experiment.sample(policy=policy, env_factory=env_factory)

    def test_first_capture_four_rollouts_and_reuse_without_policy_or_env(self):
        policy = mock.Mock(return_value={"response": "<action>act</action>", "finish_reason": "stop"})
        snapshot = self.sample(policy)
        self.assertEqual(policy.call_count, 8)
        self.assertTrue(snapshot["complete"])
        self.assertEqual(len(snapshot["trajectories"]), 4)
        self.assertTrue(all(len(t["states"]) == len(t["turns"]) + 1 for t in snapshot["trajectories"]))
        self.assertTrue(all(t["outcome"]["success"] for t in snapshot["trajectories"]))
        bad = mock.Mock(side_effect=AssertionError("must not sample"))
        self.assertEqual(self.sample(bad, bad), snapshot)
        bad.assert_not_called()

    def test_resume_replays_saved_actions_and_only_generates_missing_suffix(self):
        initial_calls = []
        def interrupted(prompt, config, trajectory, turn):
            initial_calls.append((trajectory, turn))
            if (trajectory, turn) == (1, 1):
                raise RuntimeError("connection interrupted")
            return {"response": "<action>act</action>"}
        with self.assertRaisesRegex(RuntimeError, "connection interrupted"):
            self.sample(interrupted)
        partial = probe.read_json(Path(self.temp.name) / "trajectories.json")
        self.assertFalse(partial["complete"])
        self.assertEqual(len(partial["trajectories"][1]["turns"]), 1)
        resumed_calls = []
        def resume(prompt, config, trajectory, turn):
            resumed_calls.append((trajectory, turn))
            return {"response": "<action>act</action>"}
        result = self.sample(resume)
        self.assertEqual(resumed_calls, [(1, 1), (2, 0), (2, 1), (3, 0), (3, 1)])
        self.assertTrue(result["complete"])
        self.assertEqual(result["trajectories"][1]["turns"][0], partial["trajectories"][1]["turns"][0])

    def test_prompt_edit_rescores_fixed_cache_and_archives_full_inputs(self):
        self.sample()
        snapshot_path = Path(self.temp.name) / "trajectories.json"
        original = snapshot_path.read_bytes()
        fake = FakeScorer()
        with mock.patch.object(scoring, "_get_scorer", return_value=fake):
            first = self.experiment.score(scorer=scoring.score_snapshots)
            self.experiment.save_probe({"prompt": "strict forecast {remaining_steps}"})
            second = self.experiment.score(scorer=scoring.score_snapshots)
        self.assertEqual(snapshot_path.read_bytes(), original)
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(len(list((Path(self.temp.name) / "runs").glob("*.json"))), 2)
        self.assertGreater(first["trajectories"][0]["turns"][0]["credit"], 0)
        self.assertLess(second["trajectories"][0]["turns"][0]["credit"], 0)
        self.assertIn("strict forecast", second["trajectories"][0]["turns"][0]["scoring_prompt"])
        self.assertEqual(first["trajectories"][0]["turns"][0]["action_prompt"], "action prompt 0")
        self.assertEqual(fake.calls, 6)  # Three unique states per run, shared across four trajectories.
        for filename in ("latest.json", "latest.csv", "report.html"):
            self.assertTrue((Path(self.temp.name) / filename).stat().st_size > 0)

    def test_score_never_samples_when_cache_missing_or_incomplete(self):
        scorer = mock.Mock(side_effect=AssertionError("must not score incomplete data"))
        with self.assertRaisesRegex(RuntimeError, "initial sampling"):
            self.experiment.score(scorer=scorer)
        probe.atomic_json(Path(self.temp.name) / "trajectories.json", {"complete": False})
        with self.assertRaisesRegex(RuntimeError, "initial sampling"):
            self.experiment.score(scorer=scorer)
        scorer.assert_not_called()

    def test_no_new_sampling_job_after_complete(self):
        self.sample()
        with self.assertRaisesRegex(ValueError, "already saved"):
            self.experiment.start("sample")

    def test_different_reset_state_across_rollouts_is_rejected(self):
        count = []
        class Changed(FakeEnv):
            def reset(self, **kwargs):
                result = super().reset(**kwargs)
                count.append(1)
                if len(count) > 1:
                    self.trace_state_context = "a different initial environment"
                return result
        policy = mock.Mock(return_value={"response": "<action>act</action>"})
        with self.assertRaisesRegex(RuntimeError, "mix initial states"):
            self.sample(policy, Changed)
        self.assertEqual(policy.call_count, 2)

    def test_state_dependent_response_is_rejected_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "fixed across turns"):
            self.experiment.save_probe({"response": "success at {turn_index}"})

    def goal_tools(self):
        spec = {"goal_response": "A clean cup is on the table.", "goal_spec_hash": "same-goal",
                "goal_source": "traj_data.json", "goal_task_type": "pick_clean_then_place_in_recep"}
        return SimpleNamespace(
            normalize_gamefile=lambda value: Path(value[0] if isinstance(value, list) else value),
            load_goal_spec=lambda path: spec,
            normalize_facts=lambda facts: sorted(facts),
            evaluate_goal=lambda spec, facts: "s2" in facts,
            render_full_state=lambda spec, facts, **kwargs: "full facts " + " ".join(facts)), spec

    def test_offline_goal_enrichment_preserves_raw_cache_and_checks_every_state(self):
        self.sample()
        path = Path(self.temp.name) / "trajectories.json"
        snapshot = probe.read_json(path)
        for trajectory in snapshot["trajectories"]:
            for state in trajectory["states"]:
                state["facts"] = [f"s{state['state_index']}"]
        probe.atomic_json(path, snapshot)
        original = path.read_bytes()
        tools, _ = self.goal_tools()
        with mock.patch.object(probe, "_goal_module", return_value=tools):
            converted = self.experiment.enrich()
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(converted["trajectories"][0]["goal_response"], "A clean cup is on the table.")
        self.assertTrue(all(state["goal_check_valid"] for t in converted["trajectories"] for state in t["states"]))
        self.assertEqual(self.experiment.state()["snapshot"]["trajectories"][0]["state_view"], "full_state")
        metadata = probe.read_json(Path(self.temp.name) / "goal_metadata.json")
        self.assertEqual(metadata["source_snapshot_sha256"], __import__("hashlib").sha256(original).hexdigest())
        bad = copy.deepcopy(snapshot)
        bad["trajectories"][0]["states"][1]["env_info"]["won"] = [True]
        with mock.patch.object(probe, "_goal_module", return_value=tools):
            invalid = probe.enrich_snapshot(bad)
        self.assertFalse(invalid["trajectories"][0]["states"][1]["goal_check_valid"])
        self.assertFalse(invalid["trajectories"][0]["states"][1]["scoring_valid"])

    def test_old_cache_replay_recovers_full_facts_without_sampling(self):
        self.sample()
        path = Path(self.temp.name) / "trajectories.json"
        original = path.read_bytes()
        tools, spec = self.goal_tools()
        replayed = []
        class ReplayEnv(FakeEnv):
            def details(self, result):
                prompt, info = result[0], result[-1]
                self.trace_facts = [f"s{self.index}"]
                self.trace_goal_spec = spec
                self.trace_state_context = "full facts " + self.trace_facts[0]
                info.update(trace_goal_check=self.index == 2, trace_goal_check_valid=True,
                            trace_scoring_valid=True)
                return result
            def reset(self, **kwargs):
                self.assert_game = kwargs.get("gamefile")
                return self.details(super().reset(**kwargs))
            def step(self, response):
                replayed.append(response)
                return self.details(super().step(response))
        with mock.patch.object(probe, "_goal_module", return_value=tools):
            with self.assertRaisesRegex(ValueError, "enrich --replay"):
                self.experiment.enrich()
            converted = self.experiment.enrich(replay=True, env_factory=ReplayEnv)
        self.assertEqual(len(replayed), 8)
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(all(state["state_view"] == "full_state" for t in converted["trajectories"] for state in t["states"]))
        self.experiment.save_probe({"response": "{goal_response}", "state_view": "full_state"})
        fake = FakeScorer()
        with mock.patch.object(scoring, "_get_scorer", return_value=fake), mock.patch.object(
                probe, "make_environment", side_effect=AssertionError("scoring must not load an environment")):
            result = self.experiment.score(scorer=scoring.score_snapshots)
        self.assertTrue(result["trajectories"][0]["turns"][-1]["outcome_valid"])
        self.assertEqual(path.read_bytes(), original)

    def test_goal_sidecar_cannot_be_reused_for_a_changed_cache(self):
        self.sample()
        path = Path(self.temp.name) / "trajectories.json"
        source = probe.read_json(path)
        payload = {"source_snapshot_sha256": "wrong-cache", "trajectories": []}
        with self.assertRaisesRegex(ValueError, "different trajectory cache"):
            probe._merge_enrichment(source, payload, "actual-cache")

    def test_ui_exposes_qa_fill_fixed_gt_and_replay(self):
        ui = (SCRIPTS / "alfworld_trace_probe.html").read_text()
        for text in ("{goal_response}", "/api/enrich", "terminalFill", "terminalScale", "Local credit", "Effective credit"):
            self.assertIn(text, ui)
        self.assertNotIn("不添加终局优势回填", ui)

    def test_multisample_enrichment_isolated_and_lazy_snapshot_version_changes(self):
        snapshot = self.sample()
        raw_path = Path(self.temp.name) / "trajectories.json"
        original = raw_path.read_bytes()
        workspace = probe.SampleWorkspace(self.experiment)
        with mock.patch.object(probe, "select_row", return_value=self.selection), mock.patch.object(
                probe.Experiment, "start") as start:
            created = workspace.create_sample({"row_index": 1, "trajectories": 4, "workers": 2})
        start.assert_called_once_with("sample")
        selected = workspace.current
        copy_snapshot = copy.deepcopy(snapshot)
        copy_snapshot["dataset"]["row_index"] = 1
        for trajectory in copy_snapshot["trajectories"]:
            for state in trajectory["states"]:
                state["facts"] = [f"s{state['state_index']}"]
        selected_path = selected.directory / "trajectories.json"
        probe.atomic_json(selected_path, copy_snapshot)
        selected_raw = selected_path.read_bytes()
        before = workspace.state()
        self.assertIsNone(before["snapshot"])
        self.assertFalse(before["goal_enrichment"])
        tools, _ = self.goal_tools()
        with mock.patch.object(probe, "_goal_module", return_value=tools):
            selected.enrich()
        after = workspace.state()
        self.assertNotEqual(before["snapshot_version"], after["snapshot_version"])
        self.assertTrue(after["goal_enrichment"])
        self.assertEqual(after["active_sample_id"], created["sample_id"])
        self.assertEqual(workspace.snapshot()["trajectories"][0]["goal_spec_hash"], "same-goal")
        self.assertEqual(selected_path.read_bytes(), selected_raw)
        self.assertEqual(raw_path.read_bytes(), original)
        workspace.select_sample("legacy")
        self.assertNotIn("goal_response", workspace.snapshot()["trajectories"][0])
        self.assertFalse(workspace.state()["goal_enrichment"])
        workspace.select_sample(created["sample_id"])
        self.assertEqual(workspace.snapshot()["trajectories"][0]["goal_response"], "A clean cup is on the table.")

    def test_server_enrich_and_snapshot_routes_target_the_active_sample(self):
        snapshot = self.sample()
        workspace = probe.SampleWorkspace(self.experiment)
        with mock.patch.object(probe, "select_row", return_value=self.selection), mock.patch.object(
                probe.Experiment, "start"):
            workspace.create_sample({"row_index": 1})
        selected = workspace.current
        selected_snapshot = copy.deepcopy(snapshot)
        for trajectory in selected_snapshot["trajectories"]:
            for state in trajectory["states"]:
                state["facts"] = [f"s{state['state_index']}"]
        probe.atomic_json(selected.directory / "trajectories.json", selected_snapshot)
        tools, _ = self.goal_tools()
        with mock.patch.object(probe, "_goal_module", return_value=tools):
            selected.enrich()
        server = mock.Mock(server_port=8765)
        with mock.patch.object(probe, "SampleWorkspace", return_value=workspace), mock.patch.object(
                probe, "ThreadingHTTPServer", return_value=server) as constructor:
            probe.serve(self.experiment, "127.0.0.1", 8765)
        handler_cls = constructor.call_args.args[1]
        handler = object.__new__(handler_cls)
        handler.send = mock.Mock()
        handler.headers = Message()
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = "2"
        handler.rfile = io.BytesIO(b"{}")
        handler.path = "/api/enrich"
        with mock.patch.object(selected, "start") as active_start, mock.patch.object(
                self.experiment, "start") as legacy_start:
            handler.do_POST()
        active_start.assert_called_once_with("enrich", None)
        legacy_start.assert_not_called()
        handler.send.assert_called_with({"ok": True})
        handler.path = "/api/snapshot"
        handler.do_GET()
        self.assertEqual(handler.send.call_args.args[0]["trajectories"][0]["goal_spec_hash"], "same-goal")

    def test_ui_keeps_multisample_recovery_and_lazy_loading(self):
        ui = (SCRIPTS / "alfworld_trace_probe.html").read_text()
        for text in ("/api/new-sample", "/api/select-sample", "/api/recover-snapshot", "/api/snapshot",
                     "snapshot_version", "snapshotCacheKey", "enrichButton", "repairButton"):
            self.assertIn(text, ui)

    def test_error_post_state_does_not_reuse_stale_context(self):
        env = FakeEnv({})
        state = probe.state_snapshot(env, "bad", {
            "anchor": "[FORMAT_ERROR]", "observation_text": "Episode ended due to invalid format.",
            "format_error": True,
        }, 1, self.experiment.config["sampling"], done=True)
        self.assertNotEqual(state["state_context"], "stage=0")
        self.assertIn("invalid format", state["state_context"])

    def test_external_failure_is_saved_without_success_fabrication(self):
        class Broken(FakeEnv):
            def step(self, response):
                raise RuntimeError("simulator crashed")
        result = self.sample(env_factory=Broken)
        for trajectory in result["trajectories"]:
            self.assertEqual(trajectory["status"], "external_error")
            self.assertFalse(trajectory["outcome"]["valid"])
            self.assertIsNone(trajectory["outcome"]["success"])
            self.assertFalse(trajectory["states"][-1]["scoring_valid"])

    def test_report_escapes_model_text_and_csv_preserves_multiline(self):
        result = {"trajectories": [{"trajectory_id": "t", "outcome": {}, "states": [],
                  "turns": [{"turn_index": 0, "action_response": "<script>bad()</script>\nsecond line",
                             "credit": None}]}]}
        report = probe.export_report(result)
        self.assertNotIn("<script>bad()", report)
        self.assertIn("&lt;script&gt;", report)
        rows = list(__import__("csv").DictReader(__import__("io").StringIO(probe.export_csv(result))))
        self.assertEqual(rows[0]["action_response"], "<script>bad()</script>\nsecond line")

    def test_select_row_uses_question_zero_based_id(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("optional pyarrow is unavailable")
        path = Path(self.temp.name) / "sample.parquet"
        pq.write_table(pa.Table.from_pylist([{"id": 1, "question": "train_all_alfworld_0", "result": ""}]), path)
        selected = probe.select_row(str(path), 0)
        self.assertEqual(selected["reset"]["question_id"], 0)
        self.assertEqual(selected["source_row"]["id"], 1)


if __name__ == "__main__":
    unittest.main()
