"""Capacity diagnostics must measure real local HTTP/SQLite without deployment state."""
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("search_cache_capacity_local", REPO / "train-roma/search_cache_capacity_local.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class CapacityLocalTests(unittest.TestCase):
    def test_old_admission_and_connection_boundaries_are_not_applicable(self):
        for result in (probe.probe_admission_boundary(workers=1, queue_size=0),
                       probe.probe_connection_boundary(workers=1, queue_size=0)):
            self.assertEqual(result["status"], "not_applicable", result)
            self.assertEqual(result["legacy_ignored"], {"workers": 1, "queue_size": 0})
            self.assertIsNone(result.get("expected_capacity"))
            self.assertIsNone(result.get("expected_connection_cap"))

    def test_actual_http_workload_distinguishes_warm_cold_and_shared_requests(self):
        common = dict(workers=1, burst_size=12, duration=0, latency_limit=3,
                      payload_bytes=1024, producer_delay=0.1)
        warm = probe.probe_worker_stage(mode="warm_cache", **common)
        cold = probe.probe_worker_stage(mode="cold_unique", **common)
        shared = probe.probe_worker_stage(mode="same_key_singleflight", **common)
        for result in (warm, cold, shared):
            self.assertEqual(result["status"], "stable", result)
            self.assertEqual(result["requests"], 12)
            self.assertEqual(result["successes"], 12)
            self.assertEqual(result["errors"], 0)
            self.assertGreater(result["throughput_rps"], 0)
            self.assertGreaterEqual(result["max_seconds"], result["p95_seconds"])
        self.assertEqual(warm["producer_count"], 0)
        self.assertGreater(warm["peak_executing_workers"], 0)
        self.assertEqual(warm["metrics_delta"]["cache_hits"], 12)
        self.assertEqual(cold["producer_count"], 12)
        self.assertGreater(cold["peak_producer_active"], 1)
        self.assertEqual(cold["queue_wait_p95_seconds"], 0.0)
        self.assertEqual(cold["legacy_ignored"]["workers"], 1)
        self.assertEqual(shared["producer_count"], 1)
        self.assertGreater(shared["metrics_delta"].get("coalesced_waiters", 0), 0)

    def test_deadline_bounds_independent_producers_without_pretending_to_kill_threads(self):
        result = probe.probe_deadline_behavior()
        self.assertEqual(result["status"], "verified", result)
        self.assertIsNone(result["queued_task_cancelled"])
        self.assertTrue(result["running_task_continued_after_caller_deadline"])
        self.assertEqual(result["provider_executions_before_release"], 2)
        self.assertEqual(result["cache_entries_after_expired_producer"], 0)
        self.assertLess(result["elapsed_wait_seconds"], 1)

    def test_loopback_limiter_respects_shared_and_per_hostname_buckets_despite_proxy_env(self):
        # This URL must never actually be contacted by an offline diagnostic.
        with patch.dict(os.environ, {"HTTP_PROXY": "http://proxy.invalid:9",
                                     "HTTPS_PROXY": "http://proxy.invalid:9", "NO_PROXY": ""}):
            result = probe.probe_budget_rules()
        self.assertEqual(result["scope"], "limiter_mechanism_only_not_production_quota")
        wiki, brave, web = result["scenarios"]
        for item in result["scenarios"]:
            self.assertEqual(item["status"], "verified", item)
            self.assertEqual(item["offered_requests"], 8)
            if item["service_limit_applies"]:
                self.assertTrue(all(value <= 2 for value in item["peak_concurrency_by_group"].values()))
        self.assertEqual(wiki["bucket_names"], [])
        self.assertTrue(wiki["legacy_wiki_limits_ignored"])
        self.assertTrue(all(value["unlimited"] for value in wiki["routing"].values()))
        self.assertEqual(brave["bucket_names"], ["brave"])
        self.assertEqual(set(web["bucket_names"]), {"web:127.0.0.1", "web:localhost"})
        self.assertGreater(web["overall_peak_concurrency"], 2)

    def test_suite_report_is_json_and_does_not_claim_real_world_upper_limits(self):
        result = probe.run_local_suite(workers=[2], burst_size=4, duration=0, latency_limit=3,
                                       payload_bytes=256, producer_delay=0.01, include_connections=False)
        json.dumps(result, allow_nan=False)
        self.assertFalse(result["actual_production_service_touched"])
        self.assertIsNone(result["recommendations"]["largest_tested_stable_workers"])
        self.assertEqual(result["settings"]["legacy_ignored_worker_candidates"], [2])
        self.assertIsNone(result["recommendations"]["queue_size_upper_limit"])
        self.assertIsNone(result["recommendations"]["production_rpm_upper_limit"])

    def test_ignored_worker_candidates_do_not_repeat_identical_workloads(self):
        completed = {"mode": "synthetic", "workers": 1, "status": "stable", "requests": 4}
        with patch.object(probe, "probe_worker_stage", return_value=completed) as stage, \
             patch.object(probe, "probe_admission_boundary", return_value={}), \
             patch.object(probe, "probe_deadline_behavior", return_value={}), \
             patch.object(probe, "probe_budget_rules", return_value={}):
            suite = probe.run_local_suite(workers=[1, 2, 8], include_connections=False)
        self.assertEqual(stage.call_count, 3)
        self.assertEqual(suite["settings"]["legacy_ignored_worker_candidates"], [1, 2, 8])
        self.assertIsNone(suite["recommendations"]["largest_tested_stable_workers"])

    def test_invalid_probe_inputs_fail_before_starting_threads(self):
        for kwargs in (dict(workers=[]), dict(workers=[0]), dict(workers=[True]),
                       dict(workers=[2], burst_size=0), dict(workers=[2], duration=float("nan")),
                       dict(workers=[2], latency_limit=0), dict(workers=[2], payload_bytes=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                probe.run_local_suite(**kwargs)

    def test_failed_prewarm_keeps_worker_identity_and_is_not_recommended(self):
        failure = {"status": 503, "error_code": "queue_full", "latency_seconds": 0.01,
                   "queue_wait_seconds": None}
        with patch.object(probe, "_http_call", return_value=failure):
            result = probe.probe_worker_stage(workers=4, burst_size=12, duration=0, latency_limit=3,
                                              payload_bytes=256, producer_delay=0, mode="warm_cache")
        self.assertEqual(result["workers"], 4)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["offered_client_concurrency"], 12)
        json.dumps(result, allow_nan=False)
        with patch.object(probe, "probe_worker_stage", return_value=result), \
             patch.object(probe, "probe_admission_boundary", return_value={}), \
             patch.object(probe, "probe_deadline_behavior", return_value={}), \
             patch.object(probe, "probe_budget_rules", return_value={}):
            suite = probe.run_local_suite(workers=[4], include_connections=False)
        self.assertIsNone(suite["recommendations"]["largest_tested_stable_workers"])

    def test_boundary_resource_failures_preserve_completed_stage_results(self):
        completed = {"mode": "synthetic", "workers": 4, "status": "stable", "requests": 12}
        with patch.object(probe, "probe_worker_stage", return_value=completed), \
             patch.object(probe, "probe_admission_boundary", side_effect=OSError("No descriptors")), \
             patch.object(probe, "probe_deadline_behavior", side_effect=RuntimeError("No threads")), \
             patch.object(probe, "probe_connection_boundary", side_effect=OSError("No descriptors")), \
             patch.object(probe, "probe_budget_rules", return_value={}):
            suite = probe.run_local_suite(workers=[4])
        self.assertEqual(len(suite["stages"]), 3)
        self.assertTrue(all(item["status"] == "inconclusive" for item in suite["boundaries"].values()))
        self.assertIsNone(suite["recommendations"]["largest_tested_stable_workers"])
        json.dumps(suite, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
