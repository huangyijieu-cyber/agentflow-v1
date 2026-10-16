"""Permanent shared cache compatibility, without real network traffic."""
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.search_service.core import (
    CACHE_TTL_SECONDS, SearchService, ServiceConfig, SQLiteCache, json_dumps,
)


class PermanentProviders:
    namespace = "permanent-regression"

    def __init__(self):
        self.calls = 0

    def prepare(self, tool, params):
        return {"query": params["query"]}

    def execute(self, tool, params, service):
        def produce():
            self.calls += 1
            return {"results": [{"title": params["query"], "text": "retained"}]}

        return service.cached(self.namespace, params, produce, CACHE_TTL_SECONDS)


class PermanentCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cache.sqlite3"
        self.now = 1000.0
        clock = patch("agentflow.search_service.core.time.time", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.providers = PermanentProviders()
        self.service = self.make_service()

    def make_service(self):
        # Existing deployment/config objects may still carry a seven-day cap.
        service = SearchService(ServiceConfig(
            cache_dir=self.directory.name, db_path=str(self.path), token="test-token",
            cache_max_ttl=604800,
        ), self.providers)
        self.addCleanup(service.close)
        return service

    def request(self):
        return self.service.handle("wikipedia", {"query": "same"})

    def key(self):
        return hashlib.sha256(json_dumps({"query": "same"}, canonical=True).encode()).hexdigest()

    def raw_rows(self):
        with closing(sqlite3.connect(self.path)) as conn:
            return conn.execute("SELECT namespace,key,value,expires,accessed,size,ttl "
                                "FROM entries ORDER BY namespace,key").fetchall()

    def test_months_later_cache_hit_never_calls_upstream_again(self):
        self.assertIsNone(CACHE_TTL_SECONDS)
        first = self.request()
        self.assertTrue(self.service.flush_cache())
        stored = self.raw_rows()[0]
        self.assertEqual((stored[3], stored[6]), (0, 0))
        self.now += 180 * 86400
        later = self.request()
        self.assertEqual(later["results"], first["results"])
        self.assertEqual(later["meta"].get("cache_hits"), 1)
        self.assertFalse(later["meta"].get("cache_stale", False))
        self.assertEqual(self.providers.calls, 1)

    def test_restart_months_later_reuses_same_database_value(self):
        first = self.request()
        self.assertTrue(self.service.flush_cache())
        self.service.close()
        self.now += 365 * 86400
        self.service = self.make_service()
        self.assertEqual(self.request()["results"], first["results"])
        self.assertEqual(self.providers.calls, 1)
        self.assertEqual(self.service.metrics_snapshot()["cache"]["entries"], 1)

    def test_existing_expired_records_are_hits_without_startup_rewrite(self):
        value = json_dumps({"results": [{"title": "same", "text": "old-database"}]})
        with self.service.cache.connect() as conn:
            conn.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?)", (
                self.providers.namespace, self.key(), value, 50, 25,
                len(value.encode()), 604800,
            ))
        self.service.close()
        before = self.raw_rows()
        self.now = 1000 + 365 * 86400
        self.service = self.make_service()
        self.assertEqual(self.raw_rows(), before, "Opening the DB must not rewrite every old row")
        result = self.request()
        self.assertEqual(result["results"][0]["text"], "old-database")
        self.assertEqual(result["meta"].get("cache_hits"), 1)
        self.assertEqual(self.providers.calls, 0)
        self.assertEqual(self.raw_rows(), before, "Reads should defer LRU metadata writes")
        self.service.close()
        after = self.raw_rows()[0]
        self.assertEqual((after[2], after[3], after[6]), (value, 50, 604800))
        self.assertEqual(after[4], self.now)

    def test_old_ttl_environment_cannot_reenable_expiry(self):
        config = ServiceConfig.from_env({"SEARCH_CACHE_MAX_TTL": "604800"})
        cache = SQLiteCache(Path(self.directory.name) / "old-env.sqlite3",
                            max_bytes=10000, max_ttl=config.cache_max_ttl)
        self.addCleanup(cache.close)
        cache.put("test", "value", '"saved"', 1)
        self.now += 365 * 86400
        self.assertEqual(cache.get("test", "value"), '"saved"')
        self.assertEqual(cache.stats(), {"entries": 1, "bytes": 7})

    def test_none_and_positive_ttl_both_store_permanently_despite_old_constructor_cap(self):
        cache = self.service.cache
        self.assertTrue(cache.put("test", "none", '"none"', None))
        self.assertTrue(cache.put("test", "positive", '"positive"', 0.001))
        self.now += 365 * 86400
        self.assertEqual(cache.get("test", "none"), '"none"')
        self.assertEqual(cache.get("test", "positive"), '"positive"')
        self.assertTrue(all((row[3], row[6]) == (0, 0) for row in self.raw_rows()))

    def test_unrelated_write_keeps_expired_legacy_row(self):
        with self.service.cache.connect() as conn:
            conn.execute("INSERT INTO entries VALUES ('legacy','old','\"old\"',1,2,5,604800)")
        before = self.raw_rows()[0]
        self.now += 365 * 86400
        self.service.cache.put("other", "new", "{}", None)
        self.assertEqual(self.raw_rows()[0], before)
        self.assertEqual(self.service.cache.get("legacy", "old"), '"old"')
        self.assertEqual(self.service.cache.stats(), {"entries": 2, "bytes": 7})

    def test_expired_legacy_hit_updates_lru_and_capacity_can_still_evict(self):
        cache = SQLiteCache(Path(self.directory.name) / "capacity.sqlite3", max_bytes=16, max_ttl=1)
        self.addCleanup(cache.close)
        with cache.connect() as conn:
            conn.execute("INSERT INTO entries VALUES ('a','recent','\"123456\"',1,10,8,604800)")
            conn.execute("INSERT INTO entries VALUES ('a','idle','\"abcdef\"',1,20,8,604800)")
        self.assertEqual(cache.get("a", "recent"), '"123456"')
        self.now += 1
        self.assertTrue(cache.put("a", "new", '"uvwxyz"', None))
        self.assertEqual(cache.get("a", "recent"), '"123456"')
        self.assertIsNone(cache.get("a", "idle"))
        self.assertEqual(cache.get("a", "new"), '"uvwxyz"')
        self.assertEqual(cache.stats(), {"entries": 2, "bytes": 16})

    def test_explicit_zero_and_negative_ttl_skip_cache_without_overwriting_good_value(self):
        cache = self.service.cache
        cache.put("test", "good", '"good"', None)
        for ttl in (0, -1):
            with self.subTest(ttl=ttl):
                self.assertFalse(cache.put("test", "good", '"bad"', ttl))
                self.assertFalse(cache.put("test", "new", '"bad"', ttl))
        self.assertEqual(cache.get("test", "good"), '"good"')
        self.assertIsNone(cache.get("test", "new"))
        for ttl in (True, False, float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=ttl), self.assertRaises(ValueError):
                cache.put("test", "good", '"bad"', ttl)
        self.assertEqual(cache.get("test", "good"), '"good"')


if __name__ == "__main__":
    unittest.main()
