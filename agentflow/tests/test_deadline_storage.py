"""Cache waits, transactional capacity accounting and bounded LRU maintenance."""
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
from agentflow.search_service.core import GatewayFailure, SQLiteCache


class DeadlineStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cache.sqlite3"
        self.cache = SQLiteCache(self.path, max_bytes=100000, max_ttl=604800)
        self.addCleanup(self.cache.close)

    def totals(self):
        with closing(sqlite3.connect(self.path)) as conn:
            actual = conn.execute("SELECT COUNT(*),COALESCE(SUM(size),0) FROM entries").fetchone()
            tracked = conn.execute("SELECT entries,bytes FROM cache_totals WHERE id=1").fetchone()
        self.assertEqual(tracked, actual)
        return tracked

    def assert_deadline(self, phase, function):
        phases = []
        started = time.monotonic()
        with self.assertRaises(GatewayFailure) as caught:
            with self.cache.operation(deadline=started + 0.06, phase_callback=phases.append):
                function()
        self.assertEqual(caught.exception.code, "deadline_exceeded")
        self.assertEqual(caught.exception.diagnostics["deadline_reason"], phase)
        self.assertIn(phase, phases)
        self.assertLess(time.monotonic() - started, 0.6)

    def test_put_uses_incremental_bytes_not_full_table_sum(self):
        statements = []
        self.cache._writer.set_trace_callback(statements.append)
        self.cache.put("a", "one", '"one"', 100)
        self.cache.put("a", "one", '"longer"', 100)
        self.assertEqual(self.totals(), (1, 8))
        self.assertFalse(any("SUM(" in statement.upper() for statement in statements))

    def test_manual_writes_and_rollback_keep_totals_transactional(self):
        self.cache.put("a", "one", '"one"', 100)
        with self.cache.connect() as conn:
            conn.execute("UPDATE entries SET size=20 WHERE key='one'")
        self.assertEqual(self.totals(), (1, 20))
        with self.cache.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO entries VALUES ('a','one','{}',9999999999,1,2,100)")
            conn.execute("INSERT INTO entries VALUES ('a','two','[]',9999999999,1,2,100)")
        self.assertEqual(self.totals(), (2, 4))
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with self.cache.connect() as conn:
                conn.execute("DELETE FROM entries")
                raise RuntimeError("rollback")
        self.assertEqual(self.totals(), (2, 4))
        with self.cache.connect() as conn:
            conn.execute("DELETE FROM entries WHERE key='two'")
        self.assertEqual(self.totals(), (1, 2))

    def test_legacy_database_initializes_totals_without_touching_expiry(self):
        legacy = Path(self.directory.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(legacy)) as conn, conn:
            conn.execute("CREATE TABLE entries(namespace TEXT,key TEXT,value TEXT,expires REAL,"
                         "accessed REAL,size INTEGER,PRIMARY KEY(namespace,key))")
            conn.execute("INSERT INTO entries VALUES ('a','old','{}',1500,1000,2)")
        cache = SQLiteCache(legacy, max_bytes=10000, max_ttl=604800)
        self.addCleanup(cache.close)
        with cache.connect() as conn:
            self.assertEqual(conn.execute("SELECT entries,bytes FROM cache_totals").fetchone(), (1, 2))
            self.assertEqual(conn.execute("SELECT expires FROM entries").fetchone()[0], 1500)

    def test_reader_pool_wait_obeys_deadline_and_does_not_leak_lease(self):
        self.cache._MAX_READERS = 1
        entered, release = threading.Event(), threading.Event()

        def hold_reader():
            with self.cache._read_connection():
                entered.set()
                release.wait(2)

        with ThreadPoolExecutor(max_workers=1) as executor:
            held = executor.submit(hold_reader)
            try:
                self.assertTrue(entered.wait(1))
                self.assert_deadline("cache_read_wait", lambda: self.cache.get("a", "one"))
                self.assertEqual(self.cache._active_readers, 1)
            finally:
                release.set()
            held.result()
        self.assertIsNone(self.cache.get("a", "one"))
        self.assertEqual(self.cache._active_readers, 0)

    def test_writer_lock_wait_obeys_deadline(self):
        entered, release = threading.Event(), threading.Event()

        def hold_writer():
            with self.cache.connect():
                entered.set()
                release.wait(2)

        with ThreadPoolExecutor(max_workers=1) as executor:
            held = executor.submit(hold_writer)
            try:
                self.assertTrue(entered.wait(1))
                self.assert_deadline("cache_write_wait", lambda: self.cache.put("a", "late", "{}", 100))
            finally:
                release.set()
            held.result()
        self.assertIsNone(self.cache.get("a", "late"))
        self.cache.put("a", "ok", "{}", 100)
        self.assertEqual(self.totals(), (1, 2))

    def test_external_sqlite_write_lock_obeys_deadline(self):
        with closing(sqlite3.connect(self.path)) as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            try:
                self.assert_deadline("cache_write", lambda: self.cache.put("a", "late", "{}", 100))
            finally:
                blocker.rollback()
        self.assertIsNone(self.cache.get("a", "late"))
        self.cache.put("a", "ok", "{}", 100)

    def test_slow_sqlite_query_preserves_deadline_reason_and_releases_reader(self):
        self.cache.put("a", "one", "{}", 100)
        original = self.cache._open_connection

        def open_slow():
            conn = original()
            conn.set_trace_callback(lambda sql: time.sleep(0.09) if sql.startswith("SELECT value") else None)
            return conn

        with patch.object(self.cache, "_open_connection", side_effect=open_slow):
            self.assert_deadline("cache_read", lambda: self.cache.get("a", "one"))
        self.assertEqual(self.cache._active_readers, 0)

    def test_operation_restores_outer_deadline_and_does_not_leak_to_later_calls(self):
        outer, inner = [], []
        with self.cache.operation(deadline=time.monotonic() - 1, phase_callback=outer.append):
            with self.cache.operation(phase_callback=inner.append):
                self.assertIsNone(self.cache.get("a", "one"))
            with self.assertRaises(GatewayFailure):
                self.cache.get("a", "one")
        self.assertEqual(inner, ["cache_read_wait", "cache_read"])
        self.assertEqual(outer, ["cache_read_wait"])
        self.assertIsNone(self.cache.get("a", "one"))

    def test_ordinary_writes_flush_only_a_bounded_touch_batch(self):
        self.cache._TOUCH_BATCH_SIZE = 2
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            for number in range(5):
                self.cache.put("a", str(number), "{}", 100)
        with patch("agentflow.search_service.core.time.time", return_value=1001):
            for number in range(5):
                self.cache.get("a", str(number))
        with patch("agentflow.search_service.core.time.time", return_value=1002):
            self.cache.put("a", "new", "{}", 100)
        self.assertEqual(len(self.cache._pending_touches), 3)

    def test_eviction_applies_all_observed_hits_even_beyond_flush_batch(self):
        self.cache._TOUCH_BATCH_SIZE = 2
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            self.cache.put("a", "idle", "{}", 100)
        with patch("agentflow.search_service.core.time.time", return_value=1001):
            for number in range(4):
                self.cache.put("a", str(number), "{}", 100)
        with patch("agentflow.search_service.core.time.time", return_value=1002):
            for number in range(4):
                self.cache.get("a", str(number))
        self.cache.max_bytes = 10
        with patch("agentflow.search_service.core.time.time", return_value=1003):
            self.cache.put("a", "new", "{}", 100)
            self.assertIsNone(self.cache.get("a", "idle"))
            for number in range(4):
                self.assertEqual(self.cache.get("a", str(number)), "{}")
        self.assertEqual(self.totals(), (5, 10))

    def test_eviction_rollback_restores_all_batches_of_pending_touches(self):
        self.cache._TOUCH_BATCH_SIZE = 2
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            for number in range(4):
                self.cache.put("a", str(number), "{}", 100)
        with patch("agentflow.search_service.core.time.time", return_value=1001):
            for number in range(4):
                self.cache.get("a", str(number))
        self.cache.max_bytes = 8

        def reject_commit():
            raise RuntimeError("rollback")

        with patch("agentflow.search_service.core.time.time", return_value=1002):
            with self.assertRaisesRegex(RuntimeError, "rollback"):
                self.cache.put("a", "new", "{}", 100, before_commit=reject_commit)
        self.assertEqual(len(self.cache._pending_touches), 4)
        self.assertEqual(self.totals(), (4, 8))
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertEqual(conn.execute("SELECT DISTINCT accessed FROM entries").fetchall(), [(1000.0,)])

    def test_expiry_cleanup_updates_incremental_totals(self):
        with patch("agentflow.search_service.core.time.time", return_value=1000):
            self.cache.put("a", "old", '"old"', 1)
        with patch("agentflow.search_service.core.time.time", return_value=1002):
            self.cache.put("a", "new", "{}", 100)
        self.assertEqual(self.totals(), (1, 2))


if __name__ == "__main__":
    unittest.main()
