"""Offline persistence/concurrency/real local HTTP tests for the search service."""
import json
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        self.addCleanup(self.cache.close)

    def test_restart_keeps_complete_ordered_values(self):
        value = [{"z": 1, "a": 2}, {"title": "second", "items": [4, 3, 2]}]
        self.cache.put("wiki", "one", json_dumps(value), 10)
        restart = SQLiteCache(self.path, max_bytes=100000, max_ttl=30)
        self.addCleanup(restart.close)
        restored = json.loads(restart.get("wiki", "one"))
        self.assertEqual(restored, value)
        self.assertEqual(list(restored[0]), ["z", "a"])

    def test_hits_keep_original_capped_expiry_and_do_not_revive_expired_entries(self):
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            self.cache.put("wiki", "one", '"old"', 10000)
        with patch("agentflow.search_service.core.time.time", return_value=1029):
            self.assertEqual(self.cache.get("wiki", "one"), '"old"')
        with self.cache.connect() as conn:
            self.assertEqual(conn.execute("SELECT expires, accessed FROM entries").fetchone(), (1030, 1029))
        # Even a hit one second before expiry does not extend the original TTL.
        with patch("agentflow.search_service.core.time.time", return_value=1030):
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
        self.addCleanup(restart.close)
        with patch("agentflow.search_service.core.time.time", return_value=1021):
            self.assertIsNone(restart.get("wiki", "idle"))
            self.assertIsNone(restart.get("wiki", "hit"))

    def test_healthy_legacy_schema_preserves_existing_expiry(self):
        path = Path(self.directory.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.execute("""CREATE TABLE entries (
                namespace TEXT, key TEXT, value TEXT, expires REAL,
                accessed REAL, size INTEGER, PRIMARY KEY(namespace, key))""")
            conn.execute("INSERT INTO entries VALUES ('wiki','old','\"value\"',2000,1000,7)")
        cache = SQLiteCache(path, max_bytes=100000, max_ttl=CACHE_TTL_SECONDS)
        self.addCleanup(cache.close)
        with patch("agentflow.search_service.core.time.time", return_value=1500):
            self.assertEqual(cache.get("wiki", "old"), '"value"')
        with cache.connect() as conn:
            expires, ttl = conn.execute("SELECT expires, ttl FROM entries").fetchone()
        self.assertEqual((expires, ttl), (2000, CACHE_TTL_SECONDS))
        cache.put("wiki", "new", '"new"', CACHE_TTL_SECONDS)

    def test_hits_update_lru_without_extending_expiry(self):
        cache = SQLiteCache(self.path, max_bytes=16, max_ttl=100)
        self.addCleanup(cache.close)
        with patch("agentflow.search_service.core.time.time", return_value=100):
            cache.put("a", "old", '"123456"', 100)
        with patch("agentflow.search_service.core.time.time", return_value=101):
            cache.put("a", "idle", '"abcdef"', 100)
        with patch("agentflow.search_service.core.time.time", return_value=102):
            self.assertEqual(cache.get("a", "old"), '"123456"')
        with patch("agentflow.search_service.core.time.time", return_value=103):
            cache.put("a", "new", '"uvwxyz"', 100)
            self.assertIsNone(cache.get("a", "idle"))
            self.assertEqual(cache.get("a", "old"), '"123456"')
        with cache.connect() as conn:
            self.assertEqual(conn.execute("SELECT expires FROM entries WHERE key='old'").fetchone()[0], 200)

    def test_capacity_evicts_oldest_without_partial_writes(self):
        cache = SQLiteCache(self.path, max_bytes=12, max_ttl=100)
        self.addCleanup(cache.close)
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
        self.assertTrue(self.service.flush_cache())
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

    def test_legacy_worker_and_queue_settings_do_not_serialize_unique_misses(self):
        self.service.close()
        self.service = SearchService(ServiceConfig(cache_dir=self.directory.name,
            db_path=str(Path(self.directory.name) / "cache.sqlite3"), token="test-token", workers=1, queue_size=0), self.providers)
        self.addCleanup(self.service.close)
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.service.handle, "wikipedia", {"query": q}) for q in ("first", "second")]
            try:
                until = time.monotonic() + 1
                while len(self.providers.calls) < 2 and time.monotonic() < until:
                    time.sleep(0.005)
                self.assertEqual(self.providers.calls, {"first": 1, "second": 1})
            finally:
                self.providers.release.set()
            self.assertEqual(len([future.result() for future in futures]), 2)
        self.assertEqual(self.service.metrics.snapshot().get("queue_rejections", 0), 0)

    def test_cache_hit_does_not_wait_for_slow_unique_miss(self):
        self.service.close()
        self.service = SearchService(ServiceConfig(cache_dir=self.directory.name,
            db_path=str(Path(self.directory.name) / "cache.sqlite3"), workers=1, queue_size=0), self.providers)
        self.addCleanup(self.service.close)
        self.service.handle("wikipedia", {"query": "warm"})
        self.assertTrue(self.service.flush_cache())
        self.providers.started.clear()
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=1) as executor:
            slow = executor.submit(self.service.handle, "wikipedia", {"query": "slow"})
            try:
                self.assertTrue(self.providers.started.wait(1))
                start = time.monotonic()
                warm = self.service.handle("wikipedia", {"query": "warm"}, deadline=start + 0.3)
                self.assertLess(time.monotonic() - start, 0.25)
                self.assertEqual(warm["meta"]["cache_hits"], 1)
                self.assertFalse(slow.done())
            finally:
                self.providers.release.set()
            slow.result()

    def test_duplicate_waiters_do_not_block_new_query_or_cached_result(self):
        self.service.close()
        self.service = SearchService(ServiceConfig(cache_dir=self.directory.name,
            db_path=str(Path(self.directory.name) / "cache.sqlite3"), workers=1, queue_size=0), self.providers)
        self.addCleanup(self.service.close)
        self.service.handle("wikipedia", {"query": "warm"})
        self.assertTrue(self.service.flush_cache())
        self.providers.started.clear()
        self.providers.release.clear()
        original = self.providers.execute

        def execute(tool, params, service):
            if params["query"] != "independent":
                return original(tool, params, service)
            def produce():
                self.providers.calls["independent"] += 1
                return {"results": [{"title": "independent"}]}
            return service.cached("fake:" + tool, params, produce, 3600)

        with patch.object(self.providers, "execute", side_effect=execute), ThreadPoolExecutor(max_workers=25) as executor:
            requests = [executor.submit(self.service.handle, "wikipedia", {"query": "slow"}) for _ in range(25)]
            try:
                self.assertTrue(self.providers.started.wait(1))
                self.wait_for_waiters(24)
                start = time.monotonic()
                for query in ("independent", "warm"):
                    result = self.service.handle("wikipedia", {"query": query}, deadline=time.monotonic() + 0.3)
                    self.assertEqual(result["results"][0]["title"], query)
                self.assertLess(time.monotonic() - start, 0.5)
                self.assertTrue(all(not future.done() for future in requests))
            finally:
                self.providers.release.set()
            [future.result() for future in requests]
        self.assertEqual(self.providers.calls, {"warm": 1, "slow": 1, "independent": 1})

    def test_short_waiter_deadline_does_not_cancel_owner_or_cache_write(self):
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=1) as executor:
            owner = executor.submit(self.service.handle, "wikipedia", {"query": "shared"})
            try:
                self.assertTrue(self.providers.started.wait(1))
                with self.assertRaises(GatewayFailure) as caught:
                    self.service.handle("wikipedia", {"query": "shared"}, deadline=time.monotonic() + 0.025)
                self.assertEqual(caught.exception.code, "deadline_exceeded")
                self.assertFalse(owner.done())
            finally:
                self.providers.release.set()
            owner.result()
        self.assertTrue(self.service.flush_cache())
        result = self.service.handle("wikipedia", {"query": "shared"})
        self.assertEqual(result["meta"]["cache_hits"], 1)
        self.assertEqual(self.providers.calls["shared"], 1)

    def test_batch_unique_misses_execute_concurrently_despite_legacy_worker_limit(self):
        self.service.close()
        self.service = SearchService(ServiceConfig(cache_dir=self.directory.name,
            db_path=str(Path(self.directory.name) / "cache.sqlite3"), workers=1, queue_size=0), self.providers)
        self.addCleanup(self.service.close)
        self.providers.release.clear()
        items = [{"tool": "wikipedia", "params": {"query": q}} for q in ("B", "A", "B")]
        with ThreadPoolExecutor(max_workers=1) as executor:
            batch = executor.submit(self.service.batch, items)
            try:
                until = time.monotonic() + 1
                while len(self.providers.calls) < 2 and time.monotonic() < until:
                    time.sleep(0.005)
                self.assertEqual(self.providers.calls, {"A": 1, "B": 1})
            finally:
                self.providers.release.set()
            results = batch.result()["results"]
        self.assertEqual([item["data"]["results"][0]["title"] for item in results], ["B", "A", "B"])

    def test_more_than_32_unique_cold_keys_can_all_start_before_any_finishes(self):
        gate = threading.Event()
        all_started = threading.Event()
        started_keys = set()
        lock = threading.Lock()

        def execute(tool, params, service):
            def produce():
                with lock:
                    started_keys.add(params["query"])
                    if len(started_keys) == 40:
                        all_started.set()
                if not gate.wait(4):
                    raise AssertionError("Unique producers were not released")
                return {"results": [{"title": params["query"]}]}
            return service.cached("parallel", params, produce, 3600)

        with patch.object(self.providers, "execute", side_effect=execute), ThreadPoolExecutor(max_workers=40) as executor:
            futures = [executor.submit(self.service.handle, "wikipedia", {"query": str(i)}) for i in range(40)]
            try:
                self.assertTrue(all_started.wait(3), f"Only {len(started_keys)} producers started")
                self.assertTrue(all(not future.done() for future in futures))
            finally:
                gate.set()
            results = [future.result() for future in futures]
        self.assertEqual(len(results), 40)
        self.assertEqual(self.service.metrics.snapshot()["producer_executions"], 40)

    def test_owner_deadline_does_not_remove_flight_before_producer_finishes(self):
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=2) as executor:
            owner = executor.submit(self.service.handle, "wikipedia", {"query": "slow"},
                                    deadline=time.monotonic() + 0.05)
            try:
                self.assertTrue(self.providers.started.wait(1))
                with self.assertRaises(GatewayFailure) as caught:
                    owner.result()
                self.assertEqual(caught.exception.code, "deadline_exceeded")
                waiter = executor.submit(self.service.handle, "wikipedia", {"query": "slow"})
                self.wait_for_waiters(1)
                self.assertEqual(self.providers.calls["slow"], 1)
            finally:
                self.providers.release.set()
            self.assertEqual(waiter.result()["results"][0]["title"], "slow")
        self.assertEqual(self.providers.calls["slow"], 1)
        self.assertTrue(self.service.flush_cache())
        self.assertEqual(self.service.cache.stats()["entries"], 1)

    def test_producer_deadline_wakes_waiters_and_keeps_tombstone_until_actual_exit(self):
        self.service.config.request_timeout = 0.15
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=2) as executor:
            owner = executor.submit(self.service.handle, "wikipedia", {"query": "stuck"},
                                    deadline=time.monotonic() + 3)
            try:
                self.assertTrue(self.providers.started.wait(1))
                waiter = executor.submit(self.service.handle, "wikipedia", {"query": "stuck"},
                                         deadline=time.monotonic() + 3)
                self.wait_for_waiters(1)
                for future in (owner, waiter):
                    with self.assertRaises(GatewayFailure) as caught:
                        future.result(timeout=0.8)
                    self.assertEqual(caught.exception.code, "deadline_exceeded")
                    self.assertIn("producer", caught.exception.message)
                self.assertEqual(len(self.service._flights), 1)
                started = time.monotonic()
                with self.assertRaises(GatewayFailure) as caught:
                    self.service.handle("wikipedia", {"query": "stuck"}, deadline=started + 3)
                self.assertEqual(caught.exception.code, "shared_execution_expired")
                self.assertEqual(caught.exception.diagnostics["deadline_scope"], "inherited")
                self.assertLess(time.monotonic() - started, 0.1)
                self.assertEqual(self.providers.calls["stuck"], 1)
                self.assertEqual(self.service.metrics.snapshot()["producer_deadlines_exceeded"], 1)
            finally:
                self.providers.release.set()
        until = time.monotonic() + 1
        while self.service._flights and time.monotonic() < until:
            time.sleep(0.005)
        self.assertFalse(self.service._flights)
        self.assertEqual(self.service.cache.stats()["entries"], 0)
        self.service.config.request_timeout = 2
        result = self.service.handle("wikipedia", {"query": "stuck"})
        self.assertEqual(result["results"][0]["title"], "stuck")
        self.assertEqual(self.providers.calls["stuck"], 2)

    def test_expired_owner_without_waiters_still_allows_cache_fill(self):
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=1) as executor:
            owner = executor.submit(self.service.handle, "wikipedia", {"query": "fill"},
                                    deadline=time.monotonic() + 0.05)
            try:
                self.assertTrue(self.providers.started.wait(1))
                with self.assertRaises(GatewayFailure):
                    owner.result(timeout=0.5)
                self.assertEqual(len(self.service._flights), 1)
            finally:
                self.providers.release.set()
        until = time.monotonic() + 1
        while self.service._flights and time.monotonic() < until:
            time.sleep(0.005)
        self.assertEqual(self.service.handle("wikipedia", {"query": "fill"})["meta"]["cache_hits"], 1)
        self.assertEqual(self.providers.calls["fill"], 1)

    def test_blocked_transport_abort_does_not_delay_waiters_or_close(self):
        self.service.config.request_timeout = 0.15
        started, release, abort_started, abort_release = (threading.Event() for _ in range(4))
        factory = MagicMock()
        session = factory.return_value.__enter__.return_value

        def request(*args, **kwargs):
            started.set()
            release.wait(2)
            response = requests.Response()
            response.status_code = 200
            response._content = b"ok"
            response._content_consumed = True
            return response

        def close():
            abort_started.set()
            abort_release.wait(2)

        def execute(tool, params, service):
            def produce():
                service.request("brave", "GET", "https://example.test/search")
                return {"results": []}
            return service.cached("blocked-network", params, produce, 100)

        session.request.side_effect = request
        session.close.side_effect = close
        with patch.object(self.service, "_session_factory", factory), patch.object(self.providers, "execute", side_effect=execute), ThreadPoolExecutor(max_workers=1) as executor:
            owner = executor.submit(self.service.handle, "brave", {"query": "blocked"},
                                    deadline=time.monotonic() + 3)
            try:
                self.assertTrue(started.wait(1))
                with self.assertRaises(GatewayFailure) as caught:
                    owner.result(timeout=0.8)
                self.assertEqual(caught.exception.code, "deadline_exceeded")
                self.assertTrue(abort_started.wait(0.5))
                before = time.monotonic()
                self.service.close()
                self.assertLess(time.monotonic() - before, 0.5)
            finally:
                release.set()
                abort_release.set()

    def test_service_does_not_serialize_independent_cache_reads(self):
        import hashlib
        for query in ("slow-hit", "fast-hit"):
            self.service.handle("wikipedia", {"query": query})
        self.assertTrue(self.service.flush_cache())
        slow_key = hashlib.sha256(json_dumps({"query": "slow-hit"}, canonical=True).encode()).hexdigest()
        entered, release = threading.Event(), threading.Event()
        original = self.service.cache.get

        def get(namespace, key):
            if key == slow_key:
                entered.set()
                release.wait(2)
            return original(namespace, key)

        with patch.object(self.service.cache, "get", side_effect=get), ThreadPoolExecutor(max_workers=2) as executor:
            slow = executor.submit(self.service.handle, "wikipedia", {"query": "slow-hit"})
            try:
                self.assertTrue(entered.wait(1))
                fast = executor.submit(self.service.handle, "wikipedia", {"query": "fast-hit"})
                self.assertEqual(fast.result(timeout=0.5)["meta"]["cache_hits"], 1)
                self.assertFalse(slow.done())
            finally:
                release.set()
            slow.result()

    def test_close_releases_waiters_without_allowing_late_producer_to_write(self):
        self.providers.release.clear()
        with ThreadPoolExecutor(max_workers=2) as executor:
            calls = [executor.submit(self.service.handle, "wikipedia", {"query": "slow"}) for _ in range(2)]
            try:
                self.assertTrue(self.providers.started.wait(1))
                self.wait_for_waiters(1)
                started = time.monotonic()
                self.service.close()
                self.assertLess(time.monotonic() - started, 0.5)
                for call in calls:
                    with self.assertRaises(GatewayFailure) as caught:
                        call.result(timeout=0.5)
                    self.assertEqual(caught.exception.code, "service_closed")
                replacement = SearchService(self.service.config, FakeProviders())
                self.addCleanup(replacement.close)
                with patch.object(self.service.cache, "put", side_effect=AssertionError("closed producer must not write")) as put:
                    self.providers.release.set()
                    until = time.monotonic() + 1
                    while self.service._flights and time.monotonic() < until:
                        time.sleep(0.005)
                    self.assertEqual(len(self.service._flights), 0)
                    put.assert_not_called()
                self.assertEqual(replacement.cache.stats()["entries"], 0)
            finally:
                self.providers.release.set()


class NetworkFailureTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.factory = MagicMock()
        self.session = self.factory.return_value.__enter__.return_value
        self.service = SearchService(ServiceConfig(
            cache_dir=directory.name, db_path=str(Path(directory.name) / "cache.sqlite3"),
            wiki_rpm=100000, request_timeout=5, max_retries=3,
        ), FakeProviders(), session_factory=self.factory)
        self.addCleanup(self.service.close)

    def test_proxy_auth_failure_returns_original_status_without_retry_or_secrets(self):
        self.session.request.side_effect = requests.exceptions.ProxyError(
            "Cannot connect to http://secret-token@proxy.test:80; Tunnel connection failed: 407 Proxy Authentication Required"
        )
        with patch.object(self.service, "_sleep") as sleep, self.assertRaises(GatewayFailure) as caught:
            self.service.request("wikipedia", "GET", "https://en.wikipedia.org/page")
        error = caught.exception
        self.assertEqual((error.code, error.upstream_status, error.attempts), ("upstream_proxy_error", 407, 1))
        self.assertFalse(error.retryable)
        self.assertNotIn("secret-token", json.dumps(error.as_dict()))
        self.assertEqual(self.session.request.call_count, 1)
        sleep.assert_not_called()

    def test_transient_proxy_failure_keeps_its_type_after_bounded_retries(self):
        self.session.request.side_effect = requests.exceptions.ProxyError("Tunnel connection failed: 502 Bad Gateway")
        with patch.object(self.service, "_sleep"), self.assertRaises(GatewayFailure) as caught:
            self.service.request("wikipedia", "GET", "https://en.wikipedia.org/page")
        self.assertEqual((caught.exception.code, caught.exception.upstream_status, caught.exception.attempts),
                         ("upstream_proxy_error", 502, 4))
        self.assertEqual(self.session.request.call_count, 4)
        self.assertEqual(self.service.metrics.snapshot()["upstream_errors/upstream_proxy_error"], 4)

    def test_known_proxy_error_is_not_replaced_when_retry_budget_is_exhausted(self):
        self.service._local.deadline = time.monotonic() + 0.1
        self.session.request.side_effect = requests.exceptions.ProxyError("Tunnel connection failed: 504 Gateway Time-out")
        with patch.object(self.service, "_sleep") as sleep, self.assertRaises(GatewayFailure) as caught:
            self.service.request("wikipedia", "GET", "https://en.wikipedia.org/page")
        self.assertEqual((caught.exception.code, caught.exception.upstream_status), ("upstream_proxy_error", 504))
        self.assertEqual(self.session.request.call_count, 1)
        sleep.assert_not_called()

    def test_http407_response_is_a_nonretryable_proxy_error(self):
        response = MagicMock(status_code=407, headers={})
        self.session.request.return_value = response
        with self.assertRaises(GatewayFailure) as caught:
            self.service.request("wikipedia", "GET", "https://en.wikipedia.org/page")
        self.assertEqual((caught.exception.code, caught.exception.upstream_status), ("upstream_proxy_error", 407))
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(self.session.request.call_count, 1)
        response.close.assert_called()

    def test_long_web_and_yibu_retry_after_returns_429_without_sleep(self):
        for upstream in ("brave", "yibu", "web:example.test"):
            for retry_after in ("31", "300"):
                with self.subTest(upstream=upstream, retry_after=retry_after):
                    self.session.request.reset_mock()
                    response = MagicMock(status_code=429, headers={"Retry-After": retry_after})
                    self.session.request.return_value = response
                    with patch.object(self.service, "_sleep") as sleep, self.assertRaises(GatewayFailure) as caught:
                        self.service.request(upstream, "GET", "https://example.test/search")
                    self.assertEqual(caught.exception.upstream_status, 429)
                    self.assertEqual(caught.exception.attempts, 1)
                    self.assertEqual(self.session.request.call_count, 1)
                    sleep.assert_not_called()

    def test_short_retry_after_only_backs_off_current_web_and_yibu_request(self):
        for upstream in ("brave", "web:example.test"):
            with self.subTest(upstream=upstream):
                self.session.request.reset_mock()
                limited = MagicMock(status_code=429, headers={"Retry-After": "0"})
                recovered = MagicMock(status_code=200, headers={})
                recovered.iter_content.return_value = [b"ok"]
                self.session.request.side_effect = [limited, recovered]
                with patch.object(self.service, "_sleep") as sleep:
                    self.service.request(upstream, "GET", "https://example.test/search")
                sleep.assert_called_once_with(0.5)
                self.assertEqual(self.session.request.call_count, 2)
                self.assertEqual(self.service._budget(upstream, "https://example.test/search")[1], None)

    def test_http_timeout_is_distinct_from_other_connection_errors(self):
        for exception, code, status in ((requests.ReadTimeout("read"), "upstream_timeout", 504),
                                        (requests.ConnectionError("connect"), "upstream_network_error", 502)):
            with self.subTest(code=code):
                self.service.config.max_retries = 0
                self.session.request.side_effect = exception
                with self.assertRaises(GatewayFailure) as caught:
                    self.service.request("wikipedia", "GET", "https://en.wikipedia.org/page")
                self.assertEqual((caught.exception.code, caught.exception.status_code), (code, status))

    def test_streaming_read_timeout_wrapped_by_requests_is_still_a_timeout(self):
        from urllib3.exceptions import ReadTimeoutError
        self.service.config.max_retries = 0
        response = MagicMock(status_code=200, headers={})
        response.iter_content.side_effect = requests.ConnectionError(
            ReadTimeoutError(None, "/private?token=do-not-log", "read timed out"))
        self.session.request.return_value = response
        with self.assertRaises(GatewayFailure) as caught:
            self.service.request("wikipedia", "GET", "https://en.wikipedia.org/page")
        self.assertEqual(caught.exception.code, "upstream_timeout")
        self.assertNotIn("do-not-log", str(caught.exception))
        self.assertEqual(self.service.metrics.snapshot()["upstream_errors/upstream_timeout"], 1)

    def test_tls_failure_is_explicit_and_not_retried(self):
        self.session.request.side_effect = requests.exceptions.SSLError("certificate secret")
        with self.assertRaises(GatewayFailure) as caught:
            self.service.request("wikipedia", "GET", "https://en.wikipedia.org/page")
        self.assertEqual(caught.exception.code, "upstream_tls_error")
        self.assertFalse(caught.exception.retryable)
        self.assertEqual(self.session.request.call_count, 1)
        self.assertNotIn("secret", str(caught.exception))

    def test_wikipedia_hosts_fetched_as_web_bypass_global_budgets_too(self):
        response = MagicMock(status_code=200, headers={})
        response.iter_content.return_value = [b"ok"]
        self.session.request.return_value = response
        self.service.config.wiki_rpm = 1
        self.service.config.wiki_concurrency = 1
        for host in ("wikipedia.org", "en.wikipedia.org"):
            self.assertEqual(self.service._budget("web:" + host, "https://" + host), ("wikipedia", None))
            result = self.service.request("web:" + host, "GET", "https://" + host + "/wiki/Moon")
            self.assertIs(result, response)


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

    def test_wikipedia_429_does_not_cool_down_independent_requests(self):
        self.service.config.max_retries = 0
        with self.assertRaises(GatewayFailure) as error:
            self.service.request("wikipedia", "GET", self.base_url + "/429")
        self.assertEqual(error.exception.upstream_status, 429)
        started = time.monotonic()
        self.service.request("wikipedia", "GET", self.base_url + "/ok")
        self.assertLess(time.monotonic() - started, 0.25)

    def test_brave_429_does_not_cool_down_independent_requests(self):
        self.service.config.max_retries = 0
        with self.assertRaises(GatewayFailure) as error:
            self.service.request("brave", "GET", self.base_url + "/429")
        self.assertEqual(error.exception.upstream_status, 429)
        started = time.monotonic()
        self.service.request("brave", "GET", self.base_url + "/ok")
        self.assertLess(time.monotonic() - started, 0.25)

    def test_permanent_failure_does_not_retry(self):
        with self.assertRaises(GatewayFailure) as error:
            self.service.request("wikipedia", "GET", self.base_url + "/permanent")
        self.assertEqual(error.exception.upstream_status, 403)
        self.assertEqual(self.upstream.counts["/permanent"], 1)

    def test_blocked_domain_never_opens_a_session_or_attempts_http(self):
        self.service.config.blocked_domains = ("127.0.0.1",)
        with patch.object(self.service, "_session_factory") as session_factory:
            for _ in range(2):
                with self.assertRaises(GatewayFailure) as error:
                    self.service.request("web:127.0.0.1", "GET", self.base_url + "/ok")
                self.assertEqual(error.exception.code, "domain_blocked")
                self.assertEqual(error.exception.status_code, 410)
                self.assertFalse(error.exception.retryable)
                self.assertIn("网站死亡", str(error.exception))
            session_factory.assert_not_called()
        self.assertEqual(self.upstream.paths, [])
        self.assertEqual(self.service.metrics.snapshot().get("upstream_attempts", 0), 0)
        self.assertEqual(self.service.metrics.snapshot()["blocked_domain_requests"], 2)

    def test_blacklist_matches_subdomains_case_idna_and_domain_boundaries(self):
        self.service.config.blocked_domains = ServiceConfig(
            blocked_domains=("Example.ORG.", "例子.测试")
        ).blocked_domains
        for url in ("https://EXAMPLE.org/page", "https://deep.sub.example.org./page",
                    "https://例子.测试/page", "https://xn--fsqu00a.xn--0zwm56d/page"):
            with self.assertRaises(GatewayFailure) as error:
                self.service.check_url_allowed(url)
            self.assertEqual(error.exception.code, "domain_blocked")
        for url in ("https://notexample.org", "https://example.org.other.test",
                    "https://allowed.test/path?url=https://example.org"):
            self.service.check_url_allowed(url)

    def test_redirect_to_blocked_domain_stops_without_requesting_or_retrying_target(self):
        self.service.config.blocked_domains = ("localhost",)
        with self.assertRaises(GatewayFailure) as error:
            self.service.request("web:127.0.0.1", "GET", self.base_url + "/redirect")
        self.assertEqual(error.exception.code, "domain_blocked")
        self.assertFalse(error.exception.retryable)
        self.assertEqual(self.upstream.counts, {"/redirect": 1})
        self.assertEqual(self.service.metrics.snapshot()["upstream_attempts"], 1)
        # The blocked redirect also releases its initial upstream slot.
        self.service.request("web:127.0.0.1", "GET", self.base_url + "/ok")

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

    def test_web_attempt_start_ignores_legacy_per_host_rate_spacing(self):
        self.service.config.web_rpm = 1
        self.service.request("web:127.0.0.1", "GET", self.base_url + "/ok")
        started = time.monotonic()
        self.service.request("web:127.0.0.1", "GET", self.base_url + "/ok")
        self.assertLess(time.monotonic() - started, 0.25)

    def test_web_concurrent_upstream_requests_ignore_legacy_per_host_budget(self):
        self.service.config.web_concurrency = 1
        with ThreadPoolExecutor(max_workers=8) as executor:
            replies = list(executor.map(lambda _: self.service.request("web:127.0.0.1", "GET", self.base_url + "/hold"), range(8)))
        self.assertEqual(len(replies), 8)
        self.assertGreater(self.upstream.maximum_active, 1)

    def test_wikipedia_ignores_legacy_global_rpm_and_concurrency(self):
        self.service.config.wiki_rpm = 1
        self.service.config.wiki_concurrency = 1
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            replies = list(executor.map(lambda _: self.service.request("wikipedia", "GET", self.base_url + "/hold"), range(4)))
        self.assertEqual(len(replies), 4)
        self.assertGreater(self.upstream.maximum_active, 1)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_wikipedia_api_level_cooldown_does_not_delay_other_requests(self):
        self.service.notify_cooldown("wikipedia", self.base_url, 1)
        started = time.monotonic()
        self.service.request("wikipedia", "GET", self.base_url + "/ok")
        self.assertLess(time.monotonic() - started, 0.25)
        for invalid in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                self.service.notify_cooldown("wikipedia", self.base_url, invalid)

    def test_brave_api_level_cooldown_does_not_delay_other_requests(self):
        self.service.notify_cooldown("brave", self.base_url, 60)
        started = time.monotonic()
        self.service.request("brave", "GET", self.base_url + "/ok")
        self.assertLess(time.monotonic() - started, 0.25)
        self.assertNotIn("api_cooldowns/brave", self.service.metrics.snapshot())


if __name__ == "__main__":
    unittest.main()
