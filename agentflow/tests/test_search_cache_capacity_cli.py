"""Selection, budget and reporting regressions, without Internet or services."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TRAIN_ROMA = Path(__file__).resolve().parents[2] / "train-roma"
sys.path.insert(0, str(TRAIN_ROMA))
spec = importlib.util.spec_from_file_location("capacity_cli", TRAIN_ROMA / "test_search_cache_capacity.py")
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def stage(rpm=120, concurrency=2, passed=True, duration=60):
    return {"offered_rpm": rpm, "target_concurrency": concurrency, "peak_inflight": concurrency,
            "sent": 120, "succeeded": 120, "success_rate": 1, "dispatch_seconds": duration,
            "latency_p95": .1, "latency_limit": 20, "rate_limited": False,
            "target_reached": True, "aborted_reason": "", "error_codes": {},
            "achieved_rpm": rpm, "assessment": {"passed": passed, "reasons": [] if passed else ["rate_limited"]}}


class CapacityCliTests(unittest.TestCase):
    def test_grid_validation_and_sorting(self):
        self.assertEqual(capacity.number_grid("4,1,2,2", integer=True), [1, 2, 4])
        for invalid in ("nan", "inf", "0", "-1", ""):
            with self.assertRaises(Exception):
                capacity.number_grid(invalid)

    def test_query_and_url_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text(json.dumps({"data": {"question": "Q"}, "url": "https://site.test/p"}) + "\n" +
                            json.dumps({"data": {"question": "Q"}, "url": "https://en.wikipedia.org/wiki/A"}), encoding="utf-8")
            self.assertEqual(capacity.load_queries(path, "data.question"), ["Q"])
            self.assertEqual(capacity.load_url_groups(queries_path=str(path)), {"site.test": ["https://site.test/p"]})

    def test_unachieved_load_and_truncated_window_do_not_pass(self):
        row = stage()
        row.update(target_reached=False, dispatch_seconds=2)
        verdict = capacity.assess_stage(row, duration=60, min_samples=8)
        self.assertFalse(verdict["passed"])
        self.assertEqual(set(verdict["reasons"]), {"incomplete_window", "target_not_reached"})

    def test_largest_pass_is_not_absolute_upper_bound(self):
        result = capacity.select_rate_result([stage(rpm=60), stage(rpm=120)], [60, 120])
        self.assertEqual(result["highest_stable_rpm"], 120)
        self.assertEqual(result["status"], "tested_ceiling_passed_upper_bound_not_found")

    def test_failed_low_concurrency_is_not_rpm_upper_bound(self):
        result = capacity.select_rate_result([stage(rpm=120, concurrency=1, passed=False),
                                             stage(rpm=120, concurrency=2)], [60, 120])
        self.assertIsNone(result["first_failed_higher_rpm"])
        self.assertEqual(result["smallest_passing_concurrency_cap"], 2)

    def test_failure_boundary_retains_stable_lower_value(self):
        result = capacity.select_rate_result([stage(rpm=120), stage(rpm=180, passed=False)], [60, 120, 180, 240])
        self.assertEqual(result["highest_stable_rpm"], 120)
        self.assertEqual(result["first_failed_higher_rpm"], 180)
        self.assertEqual(result["status"], "failure_boundary_observed")

    def test_budget_counts_actual_calls_and_blocks_excess(self):
        control = capacity.RunControl(10, 1)
        calls = []
        call = control.wrap(lambda index: calls.append(index) or capacity.ProbeOutcome(True, 200))
        self.assertTrue(call(0).ok)
        self.assertEqual(call(1).error_code, "benchmark_request_budget")
        self.assertEqual(calls, [0])
        self.assertEqual(control.used_requests, 1)

    def test_capacity_limited_rate_sweep_increases_concurrency(self):
        args = capacity.build_parser().parse_args(["--rpm-steps", "60", "--concurrency-steps", "1,2"])
        first = stage(rpm=60, concurrency=1, passed=False)
        first.update(target_reached=False)
        second = stage(rpm=60, concurrency=2)
        probe = type("Probe", (), {"call": lambda self, i: capacity.ProbeOutcome(True)})()
        with patch.object(capacity, "probe_stage", side_effect=[first, second]), patch.object(capacity, "say"):
            result = capacity.rate_sweep(probe, "test", args, capacity.RunControl(300, 300))
        self.assertEqual(result["limit"]["smallest_passing_concurrency_cap"], 2)

    def test_concurrency_burst_must_actually_exercise_cap(self):
        args = capacity.build_parser().parse_args(["--concurrency-steps", "4", "--burst-rounds", "1"])
        row = stage(concurrency=4)
        row.update(sent=4, peak_inflight=1, concurrency_reached=False)
        with patch.object(capacity, "probe_stage", return_value=row), \
             patch.object(capacity, "rest_for_window", return_value=True), patch.object(capacity, "say"):
            result = capacity.burst_sweep(lambda index: None, "test", args, capacity.RunControl(300, 300), 120)
        self.assertIsNone(result["highest_stable_burst_concurrency"])
        self.assertEqual(result["status"], "concurrency_not_exercised")

    def test_joint_failure_reduces_all_rates_then_confirms(self):
        args = capacity.build_parser().parse_args(["--rpm-steps", "60,120", "--confirmation-rounds", "2"])
        outcomes = []
        lock = capacity.threading.Lock()
        def probe(call, **kwargs):
            row = stage(rpm=kwargs["rpm"], duration=120)
            if kwargs["rpm"] == 120:
                row.update(success_rate=.9)
            with lock:
                outcomes.append(kwargs["rpm"])
            return row
        plans = {name: {"rpm": 120, "concurrency": 2, "call": lambda index: None} for name in ("wiki", "brave")}
        with patch.object(capacity, "probe_stage", side_effect=probe), \
             patch.object(capacity, "rest_for_window", return_value=True), patch.object(capacity, "say"):
            result = capacity.mixed_confirmation(plans, args, capacity.RunControl(1000, 1000))
        self.assertEqual({v["rpm"] for v in result["confirmed"].values()}, {60})
        self.assertEqual(sorted(outcomes), [60, 60, 120, 120])

    def test_cli_local_report_contains_no_network_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            local = {"boundaries": {"queue": {"status": "verified"}}, "recommendations": {"scope": "synthetic"}}
            with patch.object(capacity, "run_local_suite", return_value=local), patch.object(capacity, "say"):
                code = capacity.main(["--mode", "local", "--output-dir", str(output)])
            self.assertEqual(code, 0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["local"], local)
            self.assertNotIn("confirmed_network_settings", report)
            self.assertTrue((output / "summary.md").is_file())

    def _run_fake_network(self, *, short=False, missing_brave=False, max_hosts=3):
        # Exercise the complete orchestration without sleeping, opening a
        # listener, loading credentials or contacting any real endpoint.
        class FakeProbe:
            def __init__(self, kind, queries, urls, **kwargs):
                if missing_brave and kind == "brave":
                    raise ValueError("missing key")
                self.label = "web:" + capacity.urlsplit(urls[0]).hostname if kind == "web" else kind
                self.calls = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def call(self, index):
                self.calls += 1
                return capacity.ProbeOutcome(True, 200, error_code=self.label)
            def metrics(self):
                return {"upstream_attempts": self.calls}
        def fake_stage(call, **kwargs):
            label = call(0).error_code
            row = stage(rpm=kwargs["rpm"], concurrency=kwargs["concurrency"], duration=kwargs["duration"])
            if label == "web:slow.test" and kwargs["rpm"] >= 60 and kwargs["concurrency"] < 2:
                row["target_reached"] = False
            return row
        with tempfile.TemporaryDirectory() as directory:
            urls = Path(directory) / "urls.txt"
            urls.write_text("https://fast.test/a\nhttps://slow.test/b\n", encoding="utf-8")
            command = ["--mode", "upstream", "--tools", "wiki,brave,web", "--rpm-steps", "30,60",
                       "--concurrency-steps", "1,2", "--skip-bursts", "--urls-file", str(urls)]
            command += ["--max-web-hosts", str(max_hosts)]
            if short:
                command += ["--duration", "1", "--confirmation-duration", "1", "--min-samples", "1"]
            args = capacity.build_parser().parse_args(command)
            report = {"unmeasured": [], "arguments": vars(args)}
            checkpoints = []
            with patch.object(capacity, "UpstreamProbe", FakeProbe), patch.object(capacity, "probe_stage", side_effect=fake_stage), \
                 patch.object(capacity, "rest_for_window", return_value=True), patch.object(capacity, "say"):
                capacity.run_upstream(args, capacity.RunControl(10000, 10000), report,
                                      lambda: checkpoints.append(json.dumps(report)))
            self.assertGreater(len(checkpoints), 5)
            return report

    def test_complete_orchestration_confirms_common_web_caps_after_downgrade(self):
        report = self._run_fake_network()
        settings = report["confirmed_network_settings"]
        self.assertEqual(settings["SEARCH_WEB_RPM"], 30)
        self.assertEqual(settings["SEARCH_WEB_CONCURRENCY"], 1)
        self.assertTrue(report["coverage"]["all_requested_buckets_confirmed"])
        self.assertEqual(len(report["joint_confirmation"]["rounds"]), 2)

    def test_short_windows_never_export_sustained_network_configuration(self):
        report = self._run_fake_network(short=True)
        self.assertEqual(report["confirmed_network_settings"], {})
        self.assertTrue(report["smoke_network_settings"])

    def test_missing_tool_is_not_hidden_by_other_confirmed_settings(self):
        report = self._run_fake_network(missing_brave=True)
        self.assertNotIn("SEARCH_BRAVE_RPM", report["confirmed_network_settings"])
        self.assertFalse(report["coverage"]["all_requested_buckets_confirmed"])
        self.assertIn("brave", report["coverage"]["missing_buckets"])

    def test_truncated_host_selection_is_not_reported_as_complete_coverage(self):
        report = self._run_fake_network(max_hosts=1)
        self.assertFalse(report["coverage"]["all_requested_buckets_confirmed"])
        self.assertEqual(report["coverage"]["tested_web_hostnames"], ["fast.test"])
        self.assertIn("web:slow.test", report["coverage"]["missing_buckets"])

    def test_completed_rate_stages_are_checkpointed_before_interrupt(self):
        args = capacity.build_parser().parse_args(["--rpm-steps", "60,120", "--concurrency-steps", "1"])
        probe = type("Probe", (), {"call": lambda self, i: capacity.ProbeOutcome(True)})()
        snapshots = []
        with patch.object(capacity, "probe_stage", side_effect=[stage(rpm=60, concurrency=1), KeyboardInterrupt]), \
             patch.object(capacity, "say"):
            with self.assertRaises(KeyboardInterrupt):
                capacity.rate_sweep(probe, "test", args, capacity.RunControl(1000, 1000),
                                    progress=lambda rows, grid: snapshots.append(list(rows)))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0][0]["offered_rpm"], 60)


if __name__ == "__main__":
    unittest.main()
