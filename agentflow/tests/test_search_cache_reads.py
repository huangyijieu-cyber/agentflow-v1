"""Reusable WAL reads, deferred LRU metadata, and cache lifecycle regressions."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.search_service.core import SQLiteCache


class SQLiteReadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cache.sqlite3"
        self.cache = SQLiteCache(self.path, max_bytes=100000, max_ttl=604800)
        self.addCleanup(self.cache.close)

    def raw_row(self, key):
        # An independent connection must not trigger the compatibility write
        # interface's explicit metadata flush.
        with closing(sqlite3.connect(self.path)) as conn:
            return conn.execute("SELECT value,expires,accessed FROM entries WHERE key=?", (key,)).fetchone()

    def test_hits_and_stats_reuse_reader_without_writes_or_expiry_extension(self):
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            self.cache.put("a", "one", '"value"', 604800)
        with patch.object(self.cache, "_open_connection", wraps=self.cache._open_connection) as opened:
            with patch("agentflow.search_service.core.time.time", return_value=1100):
                for _ in range(50):
                    self.assertEqual(self.cache.get("a", "one"), '"value"')
                    self.assertEqual(self.cache.stats()["entries"], 1)
            self.assertEqual(opened.call_count, 1)
        self.assertEqual(self.raw_row("one"), ('"value"', 605800, 1000))
        self.assertEqual(self.cache._pending_touches, {("a", "one"): 1100})
        self.cache.close()
        self.assertEqual(self.raw_row("one"), ('"value"', 605800, 1100))

    def test_expired_read_is_a_miss_without_deleting_or_touching(self):
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            self.cache.put("a", "old", '"old"', 10)
        with patch("agentflow.search_service.core.time.time", return_value=1011):
            self.assertIsNone(self.cache.get("a", "old"))
            self.assertEqual(self.cache.stats()["entries"], 0)
        self.assertEqual(self.raw_row("old"), ('"old"', 1010, 1000))
        self.assertEqual(self.cache._pending_touches, {})

    def test_wal_hit_returns_while_writer_has_uncommitted_transaction(self):
        self.cache.put("a", "one", '"committed"', 100)
        entered, release = threading.Event(), threading.Event()

        def writer():
            with self.cache.connect() as conn:
                conn.execute("UPDATE entries SET value='\"pending\"' WHERE key='one'")
                entered.set()
                if not release.wait(2):
                    raise AssertionError("Writer was not released")

        with ThreadPoolExecutor(max_workers=2) as executor:
            writing = executor.submit(writer)
            try:
                self.assertTrue(entered.wait(1))
                reading = executor.submit(self.cache.get, "a", "one")
                self.assertEqual(reading.result(timeout=0.5), '"committed"')
                self.assertFalse(writing.done())
            finally:
                release.set()
            writing.result()
        self.assertEqual(self.cache.get("a", "one"), '"pending"')

    def test_slow_reader_does_not_serialize_other_readers(self):
        self.cache.put("a", "slow", '"slow"', 100)
        self.cache.put("a", "fast", '"fast"', 100)
        entered, release = threading.Event(), threading.Event()
        original = self.cache._open_connection

        def open_reader():
            conn = original()
            def trace(sql):
                if sql.startswith("SELECT value") and "'slow'" in sql:
                    entered.set()
                    release.wait(2)
            conn.set_trace_callback(trace)
            return conn

        with patch.object(self.cache, "_open_connection", side_effect=open_reader), ThreadPoolExecutor(max_workers=2) as executor:
            slow = executor.submit(self.cache.get, "a", "slow")
            try:
                self.assertTrue(entered.wait(1))
                fast = executor.submit(self.cache.get, "a", "fast")
                self.assertEqual(fast.result(timeout=0.5), '"fast"')
                self.assertFalse(slow.done())
            finally:
                release.set()
            self.assertEqual(slow.result(), '"slow"')

    def test_short_lived_threads_share_a_bounded_reader_pool(self):
        self.cache.put("a", "one", '"value"', 100)
        with patch.object(self.cache, "_open_connection", wraps=self.cache._open_connection) as opened:
            for _ in range(3):
                with ThreadPoolExecutor(max_workers=32) as executor:
                    self.assertEqual(list(executor.map(lambda _: self.cache.get("a", "one"), range(96))),
                                     ['"value"'] * 96)
            self.assertLessEqual(opened.call_count, self.cache._MAX_READERS)
            self.assertEqual(self.cache._active_readers, 0)
            self.assertEqual(self.cache._reader_count, len(self.cache._idle_readers))

    def test_deferred_hits_are_applied_before_lru_eviction(self):
        self.cache.max_bytes = 16
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            self.cache.put("a", "old", '"123456"', 100)
        with patch("agentflow.search_service.core.time.time", return_value=1001):
            self.cache.put("a", "idle", '"abcdef"', 100)
        with patch("agentflow.search_service.core.time.time", return_value=1002):
            self.assertEqual(self.cache.get("a", "old"), '"123456"')
            self.assertEqual(self.raw_row("old")[2], 1000)
        with patch("agentflow.search_service.core.time.time", return_value=1003):
            self.cache.put("a", "new", '"uvwxyz"', 100)
            self.assertIsNone(self.cache.get("a", "idle"))
            self.assertEqual(self.cache.get("a", "old"), '"123456"')
        self.assertEqual(self.raw_row("old")[1], 1100)

    def test_precommit_deadline_failure_rolls_back_value_and_preserves_pending_touch(self):
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            self.cache.put("a", "old", '"old"', 100)
        with patch("agentflow.search_service.core.time.time", return_value=1001):
            self.cache.get("a", "old")

        def expired():
            raise TimeoutError("Request expired before commit")

        with self.assertRaisesRegex(TimeoutError, "before commit"):
            self.cache.put("a", "late", '"late"', 100, before_commit=expired)
        self.assertIsNone(self.raw_row("late"))
        self.assertEqual(self.raw_row("old")[2], 1000)
        self.cache.close()
        self.assertEqual(self.raw_row("old")[2], 1001)

    def test_close_releases_connections_and_rejects_all_new_access(self):
        self.cache.put("a", "one", '"value"', 100)
        self.cache.get("a", "one")
        reader = self.cache._idle_readers[0]
        writer = self.cache._writer
        self.cache.close()
        self.cache.close()
        for conn in (reader, writer):
            with self.assertRaises(sqlite3.ProgrammingError):
                conn.execute("SELECT 1")
        for operation in (lambda: self.cache.get("a", "one"), self.cache.stats,
                          lambda: self.cache.put("a", "two", '"two"', 100)):
            with self.assertRaisesRegex(RuntimeError, "closed"):
                operation()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with self.cache.connect():
                pass


if __name__ == "__main__":
    unittest.main()
