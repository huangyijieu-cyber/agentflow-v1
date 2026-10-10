"""Offline persistence/concurrency/real local HTTP tests for the search service."""
import json
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.search_service.core import CACHE_TTL_SECONDS, GatewayFailure, SearchService, ServiceConfig, SQLiteCache, json_dumps


class FakeProviders:
    def __init__(self):
        self.calls = Counter()
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self.fail = False

    def prepare(self, tool, params):
        if tool not in {"wikipedia", "brave", "fetch"} or not isinstance(params, dict) or not isinstance(params.get("query"), str):
            raise GatewayFailure("Invalid request", code="invalid_request", status_code=400)
        return {"query": params["query"]}

    def execute(self, tool, params, service):
        def produce():
            with self.lock:
                self.calls[params["query"]] += 1
            self.started.set()
            if not self.release.wait(timeout=2):
                raise AssertionError("Test producer was not released")
            if self.fail:
                raise GatewayFailure("Temporary test failure", retryable=True)
            return {"results": [{"title": params["query"], "nested": [3, 2, 1]},
                                {"title": "second", "a": 1, "b": 2}], "_meta": {"partial": False}}
        return service.cached("fake:" + tool, params, produce, 3600)


class SQLiteCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cache.sqlite3"
        self.cache = SQLiteCache(self.path, max_bytes=100000, max_ttl=30)

    def test_restart_keeps_complete_ordered_values(self):
        value = [{"z": 1, "a": 2}, {"title": "second", "items": [4, 3, 2]}]
        self.cache.put("wiki", "one", json_dumps(value), 10)
        restart = SQLiteCache(self.path, max_bytes=100000, max_ttl=30)
        restored = json.loads(restart.get("wiki", "one"))
        self.assertEqual(restored, value)
        self.assertEqual(list(restored[0]), ["z", "a"])

    def test_hits_renew_capped_ttl_and_expired_entries_are_not_revived(self):
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            self.cache.put("wiki", "one", '"old"', 10000)
        with patch("agentflow.search_service.core.time.time", return_value=1029):
            self.assertEqual(self.cache.get("wiki", "one"), '"old"')
        # The hit at 1029 extends expiry from 1030 to 1059.
        with patch("agentflow.search_service.core.time.time", return_value=1031):
            self.assertEqual(self.cache.get("wiki", "one"), '"old"')
        with patch("agentflow.search_service.core.time.time", return_value=1061):
            self.assertIsNone(self.cache.get("wiki", "one"))
        self.assertEqual(self.cache.stats()["entries"], 0)

    def test_stats_do_not_renew_ttl_or_refresh_other_entries(self):
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            self.cache.put("wiki", "hit", '"hit"', 20)
            self.cache.put("wiki", "idle", '"idle"', 20)
        with patch("agentflow.search_service.core.time.time", return_value=1010):
            self.assertEqual(self.cache.stats()["entries"], 2)
            self.assertEqual(self.cache.get("wiki", "hit"), '"hit"')
        restart = SQLiteCache(self.path, max_bytes=100000, max_ttl=30)
        with patch("agentflow.search_service.core.time.time", return_value=1021):
            self.assertIsNone(restart.get("wiki", "idle"))
            self.assertEqual(restart.get("wiki", "hit"), '"hit"')

    def test_healthy_legacy_schema_gets_seven_day_sliding_expiry(self):
        path = Path(self.directory.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("""CREATE TABLE entries (
                namespace TEXT, key TEXT, value TEXT, expires REAL,
                accessed REAL, size INTEGER, PRIMARY KEY(namespace, key))""")
            conn.execute("INSERT INTO entries VALUES ('wiki','old','\"value\"',2000,1000,7)")
        cache = SQLiteCache(path, max_bytes=100000, max_ttl=CACHE_TTL_SECONDS)
        with patch("agentflow.search_service.core.time.time", return_value=1500):
            self.assertEqual(cache.get("wiki", "old"), '"value"')
        with cache.connect() as conn:
            expires, ttl = conn.execute("SELECT expires, ttl FROM entries").fetchone()
        self.assertEqual((expires, ttl), (1500 + CACHE_TTL_SECONDS, CACHE_TTL_SECONDS))
        cache.put("wiki", "new", '"new"', CACHE_TTL_SECONDS)

    def test_capacity_evicts_oldest_without_partial_writes(self):
        cache = SQLiteCache(self.path, max_bytes=12, max_ttl=100)
        with patch("agentflow.search_service.core.time.time", return_value=100):
            cache.put("a", "old", '"123456"', 100)
        with patch("agentflow.search_service.core.time.time", return_value=101):
            cache.put("a", "new", '"abcdef"', 100)
        with patch("agentflow.search_service.core.time.time", return_value=102):
            self.assertIsNone(cache.get("a", "old"))
            self.assertEqual(cache.get("a", "new"), '"abcdef"')
        self.assertFalse(cache.put("a", "huge", '"' + "x" * 20 + '"', 10))

    def test_concurrent_store_and_lookup_have_no_partial_json(self):
        def write(i):
            data = json_dumps({"index": i, "list": list(range(50))})
            self.cache.put("a", str(i), data, 20)
            return json.loads(self.cache.get("a", str(i)))["index"]
        with ThreadPoolExecutor(max_workers=16) as executor:
            self.assertEqual(list(executor.map(write, range(100))), list(range(100)))
        self.assertEqual(self.cache.stats()["entries"], 100)

    def test_invalid_ttl_does_not_overwrite_success(self):
        self.cache.put("a", "one", '"success"', 20)
        for ttl in (float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                self.cache.put("a", "one", '"error"', ttl)
        self.assertFalse(self.cache.put("a", "one", '"error"', 0))
        self.assertEqual(self.cache.get("a", "one"), '"success"')


class SearchServiceCoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.providers = FakeProviders()
        self.service = SearchService(ServiceConfig(cache_dir=self.directory.name,
            db_path=str(Path(self.directory.name) / "cache.sqlite3"), token="test-token", workers=16), self.providers)
        self.addCleanup(self.service.close)

    def wait_for_waiters(self, count):
        until = time.monotonic() + 1
        while time.monotonic() < until:
            if self.service.metrics.snapshot().get("coalesced_waiters", 0) >= count:
                return
            time.sleep(0.005)
        self.fail("Concurrent test requests did not all enter the shared flight")

    def test_identical_cold_requests_use_one_producer_and_independent_objects(self):
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(self.service.handle, "wikipedia", {"query": "same"}) for _ in range(16)]
            self.assertTrue(self.providers.started.wait(1))
            self.wait_for_waiters(15)
            self.providers.release.set()
            results = [future.result() for future in futures]
        self.assertEqual(self.providers.calls["same"], 1)
        results[0]["results"][0]["nested"].append(99)
        self.assertEqual(results[1]["results"][0]["nested"], [3, 2, 1])
        next_result = self.service.handle("wikipedia", {"query": "same"})
        self.assertEqual(next_result["meta"]["cache_hits"], 1)
        self.assertNotIn("_meta", next_result)
        self.assertFalse(next_result["meta"]["partial"])

    def test_failures_are_shared_but_not_persisted(self):
        self.providers.fail = True
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.service.handle, "wikipedia", {"query": "same"}) for _ in range(8)]
            self.assertTrue(self.providers.started.wait(1))
            self.wait_for_waiters(7)
            self.providers.release.set()
            for future in futures:
                with self.assertRaises(GatewayFailure):
                    future.result()
        self.assertEqual(self.providers.calls["same"], 1)
        self.assertEqual(self.service.cache.stats()["entries"], 0)
        self.providers.fail = False
        self.service.handle("wikipedia", {"query": "same"})
        self.assertEqual(self.providers.calls["same"], 2)

    def test_batch_deduplicates_and_restores_all_ordered_entries(self):
        items = [{"tool": "wikipedia", "params": {"query": query}} for query in ["B", "A", "B"]]
        response = self.service.batch(items + [{"tool": "bad", "params": {}}])
        self.assertEqual([r["data"]["results"][0]["title"] for r in response["results"][:3]], ["B", "A", "B"])
        self.assertEqual(self.providers.calls, {"A": 1, "B": 1})
        self.assertFalse(response["results"][3]["ok"])
        self.assertEqual(response["results"][3]["error"]["code"], "invalid_request")
        response["results"][0]["data"]["results"].clear()
        self.assertEqual(len(response["results"][2]["data"]["results"]), 2)

    def test_nonfinite_producer_result_is_not_cached(self):
        with self.assertRaises(ValueError):
            self.service.cached("invalid", {}, lambda: {"value": float("nan")}, 100)
        self.assertEqual(self.service.cache.stats()["entries"], 0)

    def test_callable_zero_ttl_does_not_cache_negative_result(self):
        self.assertEqual(self.service.cached("empty", {}, lambda: [], lambda value: 0 if not value else 10), [])
        self.assertEqual(self.service.cache.stats()["entries"], 0)

    def test_same_database_cannot_have_multiple_independent_limiters(self):
        second_dir = Path(self.directory.name) / "second-dir"
        with self.assertRaisesRegex(RuntimeError, "already owns"):
            SearchService(ServiceConfig(cache_dir=str(second_dir), token="test-token", db_path=self.service.cache.path), self.providers)

    def test_queue_is_bounded_and_waiting_deadline_is_enforced(self):
        self.service.close()
        self.service = SearchService(ServiceConfig(cache_dir=self.directory.name,
            db_path=str(Path(self.directory.name) / "cache.sqlite3"), token="test-token", workers=1, queue_size=1), self.providers)
        self.addCleanup(self.service.close)
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(self.service.handle, "wikipedia", {"query": "blocked"})
            self.assertTrue(self.providers.started.wait(1))
            second = executor.submit(self.service.handle, "wikipedia", {"query": "queued"}, deadline=time.monotonic() + 0.15)
            time.sleep(0.03)
            with self.assertRaises(GatewayFailure) as error:
                self.service.handle("wikipedia", {"query": "rejected"})
            self.assertEqual(error.exception.code, "queue_full")
            with self.assertRaises(GatewayFailure) as error:
                second.result()
            self.assertEqual(error.exception.code, "deadline_exceeded")
            self.providers.release.set()
            first.result()
        self.assertNotIn("queued", self.providers.calls)


class _LocalUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            pass

    def log_message(self, *args):
        pass

    def do_GET(self):
        with self.server.lock:
            self.server.paths.append((self.path, time.monotonic(), dict(self.headers)))
            self.server.counts[self.path] += 1
            count = self.server.counts[self.path]
        if self.path == "/429" or self.path == "/recover" and count == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/permanent":
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.server.server_port}/ok")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/slow":
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            for _ in range(100):
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
                time.sleep(0.05)
            return
        if self.path == "/hold":
            with self.server.lock:
                self.server.active += 1
                self.server.maximum_active = max(self.server.maximum_active, self.server.active)
            time.sleep(0.04)
            with self.server.lock:
                self.server.active -= 1
        body = b"x" * (200 if self.path == "/large" else 10)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class UpstreamBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _LocalUpstream)
        self.upstream.daemon_threads = True
        self.upstream.paths, self.upstream.counts, self.upstream.lock = [], Counter(), threading.Lock()
        self.upstream.active, self.upstream.maximum_active = 0, 0
        self.thread = threading.Thread(target=self.upstream.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.upstream.server_close)
        self.addCleanup(self.upstream.shutdown)
        self.base_url = f"http://127.0.0.1:{self.upstream.server_port}"
        config = ServiceConfig(cache_dir=self.directory.name,
                               db_path=str(Path(self.directory.name) / "cache.sqlite3"),
                               token="test", wiki_rpm=100000, web_rpm=100000,
                               max_retries=1, request_timeout=2)
        self.service = SearchService(config, FakeProviders())
        self.addCleanup(self.service.close)

    def test_retry_is_an_actual_counted_attempt_and_recovery_preserves_body(self):
        response = self.service.request("wikipedia", "GET", self.base_url + "/recover")
        self.assertEqual(response.content, b"x" * 10)
        self.assertEqual(self.upstream.counts["/recover"], 2)
        self.assertEqual(self.service.metrics.snapshot()["upstream_attempts/wikipedia"], 2)
        self.assertGreaterEqual(self.upstream.paths[1][1] - self.upstream.paths[0][1], 0.49)

    def test_last_429_cools_down_other_requests_not_only_its_retry(self):
        self.service.config.max_retries = 0
        with self.assertRaises(GatewayFailure) as error:
            self.service.request("wikipedia", "GET", self.base_url + "/429")
        self.assertEqual(error.exception.upstream_status, 429)
        self.service.request("wikipedia", "GET", self.base_url + "/ok")
        self.assertGreaterEqual(self.upstream.paths[1][1] - self.upstream.paths[0][1], 0.49)

    def test_permanent_failure_does_not_retry(self):
        with self.assertRaises(GatewayFailure) as error:
            self.service.request("wikipedia", "GET", self.base_url + "/permanent")
        self.assertEqual(error.exception.upstream_status, 403)
        self.assertEqual(self.upstream.counts["/permanent"], 1)

    def test_redirects_count_as_separate_attempts_and_strip_cross_host_auth(self):
        response = self.service.request("web:127.0.0.1", "GET", self.base_url + "/redirect",
                                        headers={"Authorization": "Bearer secret", "Cookie": "secret=1"})
        self.assertEqual(response.content, b"x" * 10)
        self.assertEqual(self.service.metrics.snapshot()["upstream_attempts"], 2)
        headers = self.upstream.paths[1][2]
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Cookie", headers)
        self.assertIn("upstream_attempts/web:localhost", self.service.metrics.snapshot())

    def test_slow_trickle_cannot_hold_a_worker_past_overall_deadline(self):
        self.service._local.deadline = time.monotonic() + 0.25
        start = time.monotonic()
        with self.assertRaises(GatewayFailure) as error:
            self.service.request("wikipedia", "GET", self.base_url + "/slow")
        self.assertEqual(error.exception.code, "deadline_exceeded")
        self.assertLess(time.monotonic() - start, 0.8)

    def test_large_upstream_response_is_rejected(self):
        self.service.config.max_upstream_bytes = 100
        with self.assertRaises(GatewayFailure) as error:
            self.service.request("wikipedia", "GET", self.base_url + "/large")
        self.assertEqual(error.exception.code, "upstream_too_large")

    def test_each_attempt_start_obeys_rate_spacing(self):
        self.service.config.wiki_rpm = 600
        self.service.request("wikipedia", "GET", self.base_url + "/ok")
        self.service.request("wikipedia", "GET", self.base_url + "/ok")
        self.assertGreaterEqual(self.upstream.paths[1][1] - self.upstream.paths[0][1], 0.09)

    def test_concurrent_upstream_requests_are_bounded(self):
        self.service.config.wiki_concurrency = 2
        with ThreadPoolExecutor(max_workers=8) as executor:
            replies = list(executor.map(lambda _: self.service.request("wikipedia", "GET", self.base_url + "/hold"), range(8)))
        self.assertEqual(len(replies), 8)
        self.assertLessEqual(self.upstream.maximum_active, 2)
        self.assertEqual(self.upstream.maximum_active, 2)

    def test_web_fetch_of_wiki_uses_the_same_wikipedia_budget(self):
        _, wiki_budget = self.service._budget("wikipedia", "https://en.wikipedia.org/w/api.php")
        name, web_budget = self.service._budget("web:en.wikipedia.org", "https://en.wikipedia.org/wiki/Moon")
        self.assertEqual(name, "wikipedia")
        self.assertIs(web_budget, wiki_budget)

    def test_api_level_cooldown_applies_to_other_requests(self):
        self.service.notify_cooldown("wikipedia", self.base_url, 0.08)
        started = time.monotonic()
        self.service.request("wikipedia", "GET", self.base_url + "/ok")
        self.assertGreaterEqual(time.monotonic() - started, 0.07)
        self.assertEqual(self.service.metrics.snapshot()["api_cooldowns/wikipedia"], 1)
        for invalid in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                self.service.notify_cooldown("wikipedia", self.base_url, invalid)


if __name__ == "__main__":
    unittest.main()
