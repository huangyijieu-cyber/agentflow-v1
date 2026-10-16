"""Persistence and request coalescing for one service process.

No planner, embedding, or training dependencies are imported here. The service
owns all network attempts (including retries); tool clients must not retry by
calling an external provider directly when this service fails.
"""
from __future__ import annotations

import email.utils
import fcntl
import hashlib
import json
import logging
import math
import os
import random
import re
import socket
import sqlite3
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

import requests


# Compatibility export: None means records have no age-based expiry.
CACHE_TTL_SECONDS = None
DEFAULT_CACHE_DB_PATH = "/var/tmp/agentflow-search-cache/search_cache.sqlite3"
LOG = logging.getLogger("agentflow.search_service")
_YIBU_TZ = timezone(timedelta(hours=8))


def _blocked_domain(value):
    if not isinstance(value, str):
        raise ValueError("SEARCH_BLOCKED_DOMAINS must contain domain names")
    domain = value.strip().rstrip(".").lower()
    if domain.startswith("*."):
        domain = domain[2:]
    try:
        domain = domain.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("invalid domain in SEARCH_BLOCKED_DOMAINS") from exc
    if not domain or len(domain) > 253 or any(
        not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
        for label in domain.split(".")
    ):
        raise ValueError("SEARCH_BLOCKED_DOMAINS requires domains without URLs, ports or paths")
    return domain


class GatewayFailure(RuntimeError):
    """A bounded, structured failure; its message must not contain credentials."""

    def __init__(self, message: str, *, code: str = "upstream_error",
                 status_code: int = 502, retryable: bool = False,
                 upstream: str | None = None, upstream_status: int | None = None,
                 retry_after: float | None = None, attempts: int | None = None,
                 diagnostics: dict | None = None, yibu_usage_events=None):
        self.message = " ".join(str(message).split())[:500]
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.upstream = upstream
        self.upstream_status = upstream_status
        self.retry_after = retry_after
        self.attempts = attempts
        self.diagnostics = dict(diagnostics or {})
        # Response telemetry is deliberately separate from safe error diagnostics.
        # Queries/bodies must never leak into deadline or exception messages.
        self.yibu_usage_events = list(yibu_usage_events or [])
        super().__init__(self.message)

    def as_dict(self) -> dict:
        result = {"code": self.code, "message": self.message, "retryable": self.retryable}
        for name in ("upstream", "upstream_status", "retry_after", "attempts"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        result.update(self.diagnostics)
        return result

    def with_diagnostics(self, **details):
        return GatewayFailure(self.message, code=self.code, status_code=self.status_code,
                              retryable=self.retryable, upstream=self.upstream,
                              upstream_status=self.upstream_status, retry_after=self.retry_after,
                              attempts=self.attempts, diagnostics={**self.diagnostics, **details},
                              yibu_usage_events=self.yibu_usage_events)

    def with_yibu_usage(self, events):
        copy = self.with_diagnostics()
        copy.yibu_usage_events = list({event["event_id"]: event for event in
                                     [*self.yibu_usage_events, *events]}.values())
        return copy


def json_dumps(value: Any, *, canonical: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=canonical, separators=(",", ":"))


def json_clone(value: Any) -> Any:
    return json.loads(json_dumps(value))


@dataclass
class ServiceConfig:
    cache_dir: str = "/home/ma-user/work/code-rl/cache"
    token: str = ""
    # Legacy constructor fields: no shared pool/queue or upstream limiter uses them.
    workers: int = 32
    queue_size: int = 128
    request_timeout: float = 180.0
    max_batch: int = 128
    max_body_bytes: int = 8 * 1024 * 1024
    max_upstream_bytes: int = 20 * 1024 * 1024
    cache_max_bytes: int = 2 * 1024 * 1024 * 1024
    # Legacy argument, ignored: cache lifetime is controlled only by LRU capacity.
    cache_max_ttl: float = 0.0
    # Retained only for callers constructing old configs; all are ignored.
    wiki_rpm: float = 120.0
    wiki_concurrency: int = 32
    brave_rpm: float = 240.0
    brave_concurrency: int = 16
    web_rpm: float = 120.0
    web_concurrency: int = 4
    max_retries: int = 3
    connect_timeout: float = 5.0
    read_timeout: float = 20.0
    db_path: str | None = DEFAULT_CACHE_DB_PATH
    blocked_domains: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.blocked_domains, (tuple, list)):
            raise ValueError("blocked_domains must be a list or tuple of domain names")
        self.blocked_domains = tuple(dict.fromkeys(_blocked_domain(value) for value in self.blocked_domains))
        for name in ("max_batch", "max_body_bytes", "max_upstream_bytes",
                     "cache_max_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_retries",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("request_timeout", "connect_timeout", "read_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        defaults = cls()
        mapping = {
            "request_timeout": "SEARCH_SERVICE_REQUEST_TIMEOUT", "max_batch": "SEARCH_SERVICE_MAX_BATCH",
            "max_body_bytes": "SEARCH_SERVICE_MAX_BODY_BYTES", "max_upstream_bytes": "SEARCH_SERVICE_MAX_UPSTREAM_BYTES",
            "cache_max_bytes": "SEARCH_CACHE_MAX_BYTES",
            "max_retries": "SEARCH_SERVICE_MAX_RETRIES", "connect_timeout": "SEARCH_SERVICE_CONNECT_TIMEOUT",
            "read_timeout": "SEARCH_SERVICE_READ_TIMEOUT",
        }
        values = {}
        for field, name in mapping.items():
            default = getattr(defaults, field)
            try:
                values[field] = type(default)(env.get(name, default))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid {name}") from exc
        values["cache_dir"] = env.get("SEARCH_CACHE_DIR", defaults.cache_dir)
        values["token"] = env.get("SEARCH_SERVICE_TOKEN") or env.get("SEARCH_CACHE_TOKEN", "")
        values["db_path"] = env.get("SEARCH_CACHE_DB_PATH") or defaults.db_path
        values["blocked_domains"] = tuple(filter(None, re.split(r"[,\s]+", env.get("SEARCH_BLOCKED_DOMAINS", ""))))
        return cls(**values)


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._values = Counter()

    def add(self, key, count=1):
        with self._lock:
            self._values[key] += count

    def snapshot(self):
        with self._lock:
            return dict(self._values)


class _Trace:
    """Bounded, credential-free timings for one caller or one shared producer."""

    def __init__(self, deadline, *, tool=None, request_id=None, source="service", started=None):
        self.started = time.monotonic() if started is None else started
        self.deadline = deadline
        self.tool, self.request_id, self.source = tool, request_id, source
        self.phase, self.phase_started = "request_setup", self.started
        self.durations = Counter()
        self.namespace = None
        self.upstream = self.upstream_status = self.attempts = None
        self.last_error = None
        self.recovered = False
        self.shared = None
        self._lock = threading.Lock()
        self._reported = False
        self._yibu_usage = {}

    def add_yibu_usage(self, events):
        with self._lock:
            self._yibu_usage.update((event["event_id"], event) for event in events)

    def yibu_usage(self):
        with self._lock:
            return list(self._yibu_usage.values())

    def set_phase(self, phase):
        with self._lock:
            if phase != "shared_wait":
                self.shared = None
            if phase != self.phase:
                now = time.monotonic()
                self.durations[self.phase] += max(0.0, now - self.phase_started)
                self.phase, self.phase_started = phase, now

    def set_shared(self, shared):
        with self._lock:
            self.shared = shared

    def shared_trace(self):
        with self._lock:
            return self.shared

    def observe(self, upstream, status, attempts, failure=None):
        with self._lock:
            self.upstream, self.upstream_status, self.attempts = upstream, status, attempts
            if failure is not None:
                self.last_error = self._safe_error(failure)
                self.recovered = False
            elif status is not None and 200 <= status < 300:
                self.recovered = self.last_error is not None

    def failure(self, error):
        with self._lock:
            self.last_error = self._safe_error(error)
            self.recovered = False
            if error.upstream is not None:
                self.upstream = error.upstream
            if error.attempts is not None:
                self.attempts = error.attempts

    @staticmethod
    def _safe_error(error):
        return {name: value for name in ("code", "upstream", "upstream_status", "attempts", "retry_after")
                if (value := getattr(error, name, None)) is not None}

    def claim_outcome(self):
        with self._lock:
            if self._reported:
                return False
            self._reported = True
            return True

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            durations = dict(self.durations)
            durations[self.phase] = durations.get(self.phase, 0.0) + max(0.0, now - self.phase_started)
            result = {
                "phase": self.phase, "deadline_source": self.source,
                "elapsed_seconds": round(max(0.0, now - self.started), 6),
                "budget_seconds": round(max(0.0, self.deadline - self.started), 6),
                "phase_elapsed_seconds": round(max(0.0, now - self.phase_started), 6),
                "phase_durations": {key: round(value, 6) for key, value in durations.items()},
            }
            if self.namespace is not None:
                result["cache_namespace"] = self.namespace
            if self.last_error is not None:
                result["last_upstream_error"] = dict(self.last_error)
                result["upstream_recovered"] = self.recovered
            return result


class SQLiteCache:
    """Reuse WAL readers and bound request-side waits by their caller budget."""

    _MAX_READERS = 16
    _TOUCH_BATCH_SIZE = 128

    def __init__(self, path, *, max_bytes, max_ttl=None):
        self.path = str(path)
        self.max_bytes = max_bytes
        # Retain the old constructor/attribute without allowing an old env or
        # caller to re-enable age-based expiration.
        self.max_ttl = None
        self._writer_lock = threading.RLock()
        self._read_condition = threading.Condition()
        self._idle_readers = []
        self._reader_count = 0
        self._active_readers = 0
        self._closed = False
        self._touch_lock = threading.Lock()
        self._pending_touches = {}
        self._operation_local = threading.local()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = self._open_connection()
        try:
            self._initialize()
        except BaseException:
            self._writer.close()
            raise

    def _open_connection(self):
        conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=5000")
        # REPLACE must fire the delete trigger too, including writes made via
        # connect(), otherwise accounting would double-count replacements.
        conn.execute("PRAGMA recursive_triggers=ON")
        return conn

    def _initialize(self):
        # WAL mode must be set before opening an explicit write transaction.
        conn = self._writer
        conn.execute("PRAGMA journal_mode=WAL")
        with conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS entries (
                namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                expires REAL NOT NULL, accessed REAL NOT NULL, size INTEGER NOT NULL,
                ttl REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(namespace, key))""")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(entries)")}
            if "ttl" not in columns:
                # Keep legacy records intact; lookup ignores their old expiry.
                conn.execute("ALTER TABLE entries ADD COLUMN ttl REAL NOT NULL DEFAULT 0")
            conn.execute("CREATE INDEX IF NOT EXISTS entries_expiry ON entries(expires)")
            conn.execute("CREATE INDEX IF NOT EXISTS entries_access ON entries(accessed)")
            # Bootstrap legacy databases once. Triggers keep totals transactional
            # even for callers using the compatibility connect() interface.
            conn.execute("""CREATE TABLE IF NOT EXISTS cache_totals (
                id INTEGER PRIMARY KEY CHECK(id=1), entries INTEGER NOT NULL,
                bytes INTEGER NOT NULL)""")
            if conn.execute("SELECT 1 FROM cache_totals WHERE id=1").fetchone() is None:
                conn.execute("INSERT INTO cache_totals SELECT 1,COUNT(*),COALESCE(SUM(size),0) FROM entries")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS cache_totals_insert AFTER INSERT ON entries
                BEGIN UPDATE cache_totals SET entries=entries+1,bytes=bytes+NEW.size WHERE id=1; END""")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS cache_totals_update AFTER UPDATE OF size ON entries
                BEGIN UPDATE cache_totals SET bytes=bytes+NEW.size-OLD.size WHERE id=1; END""")
            conn.execute("""CREATE TRIGGER IF NOT EXISTS cache_totals_delete AFTER DELETE ON entries
                BEGIN UPDATE cache_totals SET entries=entries-1,bytes=bytes-OLD.size WHERE id=1; END""")

    @contextmanager
    def operation(self, deadline=None, phase_callback=None):
        """Attach a monotonic deadline and phase observer to this thread only."""
        previous = getattr(self._operation_local, "state", None)
        self._operation_local.state = {"deadline": deadline, "callback": phase_callback, "phase": None}
        try:
            yield
        finally:
            self._operation_local.state = previous

    def _remaining(self, phase):
        state = getattr(self._operation_local, "state", None)
        if state is None:
            return None
        if state["phase"] != phase:
            state["phase"] = phase
            if state["callback"] is not None:
                state["callback"](phase)
        deadline = state["deadline"]
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GatewayFailure("Search deadline expired during cache access", code="deadline_exceeded",
                                 status_code=504, retryable=True,
                                 diagnostics={"phase": phase, "deadline_reason": phase})
        return remaining

    @contextmanager
    def _sqlite_operation(self, conn, phase):
        remaining = self._remaining(phase)
        if remaining is not None:
            conn.execute("PRAGMA busy_timeout=%d" % max(1, min(5000, math.ceil(remaining * 1000))))
            deadline = self._operation_local.state["deadline"]
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        try:
            yield
            self._remaining(phase)
        except sqlite3.OperationalError:
            # SQLite lock waits/progress interruption must retain deadline
            # classification instead of surfacing as a generic database error.
            self._remaining(phase)
            raise
        finally:
            if remaining is not None:
                conn.set_progress_handler(None, 0)
                conn.execute("PRAGMA busy_timeout=5000")

    def _check_open(self):
        with self._read_condition:
            if self._closed:
                raise RuntimeError("Search cache is closed")

    @contextmanager
    def _read_connection(self):
        # Lease, rather than thread-local storage: producer threads are short
        # lived and must not retain one connection for every past request.
        with self._read_condition:
            while not self._closed and not self._idle_readers and self._reader_count >= self._MAX_READERS:
                self._read_condition.wait(timeout=self._remaining("cache_read_wait"))
            if self._closed:
                raise RuntimeError("Search cache is closed")
            self._remaining("cache_read_wait")
            conn = self._idle_readers.pop() if self._idle_readers else None
            if conn is None:
                self._reader_count += 1
            self._active_readers += 1
        try:
            if conn is None:
                conn = self._open_connection()
            with self._sqlite_operation(conn, "cache_read"):
                yield conn
        finally:
            with self._read_condition:
                self._active_readers -= 1
                if conn is None or self._closed:
                    self._reader_count -= 1
                    if conn is not None:
                        conn.close()
                else:
                    self._idle_readers.append(conn)
                self._read_condition.notify_all()

    def _take_touches(self, limit=0):
        if limit == 0:
            limit = self._TOUCH_BATCH_SIZE
        with self._touch_lock:
            if limit is None or len(self._pending_touches) <= limit:
                touches, self._pending_touches = self._pending_touches, {}
            else:
                touches = {}
                for _ in range(limit):
                    key = next(iter(self._pending_touches))
                    touches[key] = self._pending_touches.pop(key)
        return touches

    def _restore_touches(self, touches):
        with self._touch_lock:
            for key, accessed in touches.items():
                self._pending_touches[key] = max(accessed, self._pending_touches.get(key, accessed))

    @staticmethod
    def _apply_touches(conn, touches):
        if touches:
            # Access time is monotonic per entry; a concurrent old read must
            # not move the access time of a replacement value backwards.
            conn.executemany("UPDATE entries SET accessed=MAX(accessed,?) "
                             "WHERE namespace=? AND key=?",
                             ((accessed, namespace, key)
                              for (namespace, key), accessed in touches.items()))

    @contextmanager
    def connect(self, *, before_commit=None):
        """Compatibility write interface; flush pending LRU touches atomically."""
        remaining = self._remaining("cache_write_wait")
        acquired = self._writer_lock.acquire() if remaining is None else self._writer_lock.acquire(timeout=remaining)
        if not acquired:
            self._remaining("cache_write_wait")
            raise RuntimeError("Search cache writer lock was not acquired")
        try:
            self._check_open()
            touches = self._take_touches()
            try:
                with self._sqlite_operation(self._writer, "cache_write"):
                    with self._writer as conn:
                        conn.execute("BEGIN IMMEDIATE")
                        self._apply_touches(conn, touches)
                        yield conn
                        if before_commit is not None:
                            before_commit()
                        self._remaining("cache_write")
            except BaseException:
                self._restore_touches(touches)
                raise
        finally:
            self._writer_lock.release()

    def get(self, namespace, key):
        with self._read_connection() as conn:
            now = time.time()
            row = conn.execute("SELECT value FROM entries WHERE namespace=? AND key=?",
                               (namespace, key)).fetchone()
            if row is None:
                return None
            with self._touch_lock:
                pair = (namespace, key)
                self._pending_touches[pair] = max(now, self._pending_touches.get(pair, now))
        return row[0]

    def put(self, namespace, key, value_json, ttl=None, *, before_commit=None):
        self._check_open()
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, (float, int)) or not math.isfinite(ttl)):
            raise ValueError("cache ttl must be finite")
        # Nonpositive TTL still means "do not store" for old callable policies.
        # Positive legacy TTLs are accepted but no longer expire the record.
        if ttl is not None and ttl <= 0:
            return False
        size = len(value_json.encode("utf-8"))
        if size > self.max_bytes:
            return False
        eviction_touches = {}
        try:
            with self.connect(before_commit=before_commit) as conn:
                now = time.time()
                conn.execute("""INSERT INTO entries (namespace, key, value, expires, accessed, size, ttl)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, key) DO UPDATE SET value=excluded.value,
                    expires=excluded.expires, accessed=excluded.accessed, size=excluded.size,
                    ttl=excluded.ttl""",
                    (namespace, key, value_json, 0, now, size, 0))
                total = conn.execute("SELECT bytes FROM cache_totals WHERE id=1").fetchone()[0]
                if total > self.max_bytes:
                    # Ordinary writes flush a bounded batch. Before eviction,
                    # apply remaining observed hits so exact LRU is preserved.
                    eviction_touches = self._take_touches(limit=None)
                    items = list(eviction_touches.items())
                    for offset in range(0, len(items), self._TOUCH_BATCH_SIZE):
                        self._remaining("cache_write")
                        self._apply_touches(conn, dict(items[offset:offset + self._TOUCH_BATCH_SIZE]))
                    while total > self.max_bytes:
                        self._remaining("cache_write")
                        rows = conn.execute("SELECT namespace,key,size FROM entries "
                                            "ORDER BY accessed,namespace,key LIMIT 128").fetchall()
                        if not rows:
                            break
                        for old_namespace, old_key, old_size in rows:
                            if total <= self.max_bytes:
                                break
                            conn.execute("DELETE FROM entries WHERE namespace=? AND key=?", (old_namespace, old_key))
                            total -= old_size
        except BaseException:
            self._restore_touches(eviction_touches)
            raise
        return True

    def stats(self):
        with self._read_connection() as conn:
            row = conn.execute("SELECT COUNT(*),COALESCE(SUM(size),0) FROM entries").fetchone()
        return {"entries": row[0], "bytes": row[1]}

    def close(self):
        with self._read_condition:
            if self._closed:
                return
            self._closed = True
            self._read_condition.notify_all()
            while self._active_readers:
                self._read_condition.wait()
            idle, self._idle_readers = self._idle_readers, []
            self._reader_count -= len(idle)
        for conn in idle:
            conn.close()
        with self._writer_lock:
            touches = self._take_touches(limit=None)
            try:
                with self._writer as conn:
                    if touches:
                        conn.execute("BEGIN IMMEDIATE")
                        self._apply_touches(conn, touches)
            finally:
                self._writer.close()


class _Flight:
    """One actual producer, whose completion and budget outlive individual callers."""

    def __init__(self, deadline, trace=None):
        self.deadline = deadline
        self.trace = trace or _Trace(deadline)
        self.future = Future()
        self.finished = threading.Event()
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._aborters = {}
        self.timer = None

    def complete(self, *, value=None, error=None):
        with self._lock:
            if self.future.done():
                return False
            if error is None:
                self.future.set_result(value)
            else:
                self.cancelled.set()
                self.future.set_exception(error)
            if self.timer is not None:
                self.timer.cancel()
            return True

    def cancel(self, error):
        if not self.complete(error=error):
            return False
        # Wake waiters before best-effort transport cleanup. Closing a mocked,
        # third-party or OS-blocked transport must never block service.close().
        with self._lock:
            aborters = list(self._aborters.values())
        for abort in aborters:
            threading.Thread(target=self._abort, args=([abort],),
                             name="search-abort", daemon=True).start()
        return True

    @staticmethod
    def _abort(aborters):
        for abort in aborters:
            try:
                abort()
            except Exception:
                pass

    def register_abort(self, abort):
        token = object()
        with self._lock:
            if not self.cancelled.is_set():
                self._aborters[token] = abort
                return token
        # The deadline may have elapsed while a request was returning headers.
        threading.Thread(target=self._abort, args=([abort],), daemon=True).start()
        return token

    def unregister_abort(self, token):
        with self._lock:
            self._aborters.pop(token, None)


def _retry_after(header):
    if not header:
        return None
    try:
        value = float(header)
        if math.isfinite(value):
            return max(0.0, value)
    except (ValueError, TypeError):
        pass
    try:
        date = email.utils.parsedate_to_datetime(header)
        return max(0.0, date.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None


def _proxy_http_status(error):
    # Inspect internally, but never expose a raw exception that may contain
    # proxy credentials, signed URLs, or request headers.
    match = re.search(r"(?:Tunnel connection failed:\s*|HTTP(?:/\d(?:\.\d)?)?\s+)([45]\d{2})\b",
                      str(error), re.IGNORECASE)
    return int(match.group(1)) if match else None


def _is_request_timeout(error):
    """requests wraps streaming ReadTimeoutError inside ConnectionError."""
    pending, seen = [error], set()
    while pending and len(seen) < 16:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, requests.Timeout) or type(current).__name__ in {
            "ReadTimeoutError", "ConnectTimeoutError", "TimeoutError",
        }:
            return True
        pending.extend(value for value in (
            getattr(current, "__cause__", None), getattr(current, "__context__", None),
            *getattr(current, "args", ()),
        ) if isinstance(value, BaseException))
    return False


def _abort_response(response):
    """Interrupt a slow response at its overall deadline, including slow trickles.

    requests' read timeout limits silence between bytes, rather than total time.
    urllib3's active socket must be shut down to wake a thread blocked in read().
    The fallback close also handles mocked/non-socket Responses.
    """
    try:
        connection = getattr(response.raw, "_connection", None)
        active_socket = getattr(connection, "sock", None)
        if active_socket is None:
            fp = getattr(getattr(response.raw, "_fp", None), "fp", None)
            active_socket = getattr(getattr(fp, "raw", None), "_sock", None)
        if active_socket is not None:
            active_socket.shutdown(socket.SHUT_RDWR)
    except (OSError, AttributeError):
        pass
    try:
        response.close()
    except (OSError, AttributeError):
        pass


class SearchService:
    def __init__(self, config: ServiceConfig, providers=None, *, session_factory=requests.Session):
        self.config = config
        self.metrics = Metrics()
        self._local = threading.local()
        self._flights = {}
        self._flight_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._session_factory = session_factory
        self._closed = False
        cache_dir = Path(config.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        # The state/config directory may be an object-store mount (e.g. s3fs).
        # SQLite and its WAL/SHM files must live on a local filesystem.
        db_path = Path(config.db_path or DEFAULT_CACHE_DB_PATH).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Lock the actual DB identity, including --db-path outside cache_dir.
        self._process_lock = db_path.with_name(db_path.name + ".service.lock").open("a+")
        try:
            fcntl.flock(self._process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._process_lock.close()
            raise RuntimeError("Another search service already owns this cache directory") from None
        try:
            self.cache = SQLiteCache(db_path,
                                     max_bytes=config.cache_max_bytes, max_ttl=config.cache_max_ttl)
            if providers is None:
                from . import providers as default_providers
                providers = default_providers
            self.providers = providers
        except BaseException:
            if hasattr(self, "cache"):
                self.cache.close()
            self._process_lock.close()
            raise

    def close(self):
        # Drain already-published material, without waiting for blocked network
        # producers. Cache IO never runs under the service-wide state lock.
        self.flush_cache(timeout=min(5.0, self.config.request_timeout))
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            with self._flight_lock:
                pending = list(self._flights.values())
            for flight in pending:
                flight.cancel(self._closed_failure())
        try:
            self.cache.close()
        finally:
            fcntl.flock(self._process_lock, fcntl.LOCK_UN)
            self._process_lock.close()

    def flush_cache(self, timeout=5.0):
        """Wait for already-returned results to finish their best-effort writes."""
        until = time.monotonic() + timeout
        with self._flight_lock:
            pending = [flight for flight in self._flights.values()
                       if flight.future.done() and not flight.future.cancelled()
                       and flight.future.exception() is None]
        return all(flight.finished.wait(max(0.0, until - time.monotonic())) for flight in pending)

    @staticmethod
    def _closed_failure():
        return GatewayFailure("Search service is closed", code="service_closed",
                              status_code=503, retryable=True)

    def deadline(self):
        return getattr(self._local, "deadline", time.monotonic() + self.config.request_timeout)

    def remaining(self):
        return max(0.0, self.deadline() - time.monotonic())

    def set_phase(self, phase):
        trace = getattr(self._local, "trace", None)
        if trace is not None:
            trace.set_phase(phase)

    def _deadline_failure(self, message="Search request deadline exceeded", *, trace=None,
                          shared=None, scope="caller", reason=None, code="deadline_exceeded"):
        trace = trace or getattr(self._local, "trace", None)
        details = trace.snapshot() if trace is not None else {
            "phase": "request_setup", "elapsed_seconds": 0.0,
            "budget_seconds": self.config.request_timeout,
        }
        cause_phase = details["phase"]
        if shared is not None:
            seen = set()
            while id(shared) not in seen:
                seen.add(id(shared))
                nested = shared.shared_trace()
                if nested is None:
                    break
                shared = nested
            shared_details = shared.snapshot()
            details["caller_phase"] = details["phase"]
            for field in ("phase", "elapsed_seconds", "budget_seconds", "phase_durations"):
                details["shared_" + field] = shared_details[field]
            for field in ("last_upstream_error", "upstream_recovered", "cache_namespace"):
                if field in shared_details:
                    details[field] = shared_details[field]
            cause_phase = shared_details["phase"]
        details.update(deadline_reason=reason or cause_phase or "request_budget_exhausted",
                       deadline_scope=scope)
        return GatewayFailure(message, code=code, status_code=504, retryable=True,
                              diagnostics=details)

    def _decorate_failure(self, error, trace):
        if error.code not in {"deadline_exceeded", "shared_execution_expired"}:
            return error
        details = trace.snapshot()
        # Preserve the producing operation's timings and cause as shared details,
        # but report the duration/budget of this particular logical caller.
        diagnostics = {**details, **error.diagnostics}
        diagnostics.update(elapsed_seconds=details["elapsed_seconds"],
                           budget_seconds=details["budget_seconds"],
                           phase=details["phase"],
                           phase_elapsed_seconds=details["phase_elapsed_seconds"],
                           phase_durations=details["phase_durations"],
                           deadline_source=details["deadline_source"])
        diagnostics.setdefault("phase", details["phase"])
        diagnostics.setdefault("deadline_reason", diagnostics["phase"])
        diagnostics.setdefault("deadline_scope", "caller")
        return error.with_diagnostics(**diagnostics)

    def _record_failure(self, error, trace):
        if not trace.claim_outcome():
            return
        self.metrics.add("requests_failed")
        self.metrics.add("requests_failed/" + error.code)
        if error.code in {"deadline_exceeded", "shared_execution_expired"}:
            details = error.diagnostics
            reason = details.get("deadline_reason", "request_budget_exhausted")
            phase = details.get("shared_phase", details.get("phase", "request_setup"))
            scope = details.get("deadline_scope", "caller")
            self.metrics.add("deadline_events")
            self.metrics.add("deadline_reasons/" + reason)
            self.metrics.add("deadline_phases/" + phase)
            self.metrics.add("deadline_scopes/" + scope)
            self.metrics.add("deadline_tools/" + (trace.tool or "unknown"))
            last_error = details.get("last_upstream_error", {})
            status = last_error.get("upstream_status")
            if status is not None and not details.get("upstream_recovered", False):
                self.metrics.add("deadline_upstream_status/" + str(status))
            if last_error.get("code") and not details.get("upstream_recovered", False):
                self.metrics.add("deadline_upstream_errors/" + last_error["code"])
            LOG.warning("search_deadline %s", json_dumps({
                "event": "search_deadline", "request_id": trace.request_id,
                "tool": trace.tool, "code": error.code, **details,
            }))

    @contextmanager
    def _state_guard(self):
        if not self._state_lock.acquire(timeout=self.remaining()):
            raise self._deadline_failure(reason="flight_register")
        try:
            self.check_deadline()
            yield
        finally:
            self._state_lock.release()

    def check_deadline(self):
        if self._closed:
            raise self._closed_failure()
        flight = getattr(self._local, "flight", None)
        if flight is not None and flight.cancelled.is_set():
            # Preserve the terminal reason chosen by the watchdog/close().
            flight.future.result()
        if self.remaining() <= 0:
            raise self._deadline_failure(scope="producer" if flight is not None else "caller")

    def check_url_allowed(self, url):
        """Reject listed hosts and subdomains before cache reads or HTTP attempts."""
        if not self.config.blocked_domains:
            return
        hostname = (urlsplit(url).hostname or "").rstrip(".").lower()
        hostname = hostname.encode("idna").decode("ascii")
        if any(hostname == domain or hostname.endswith("." + domain)
               for domain in self.config.blocked_domains):
            self._note("blocked_domain_requests")
            raise GatewayFailure(f"网站死亡：{hostname}（域名黑名单）",
                                 code="domain_blocked", status_code=410, retryable=False)

    def _prepare(self, tool, params):
        prepared = self.providers.prepare(tool, params)
        if tool == "fetch":
            self.check_url_allowed(prepared["url"])
        return prepared

    def _note(self, field, count=1):
        self.metrics.add(field, count)
        context = getattr(self._local, "context", None)
        if context is not None:
            context[field] = context.get(field, 0) + count

    def cached(self, namespace, params, producer: Callable, ttl=None, persist=True):
        if not isinstance(namespace, str) or not namespace or len(namespace) > 128:
            raise ValueError("invalid cache namespace")
        self.check_deadline()
        deadline = self.deadline()
        trace = getattr(self._local, "trace", None)
        if trace is not None:
            trace.namespace = namespace
        key = hashlib.sha256(json_dumps(params, canonical=True).encode("utf-8")).hexdigest()
        flight_key = (namespace, key)
        cached = self._cache_get(namespace, key) if persist else None
        if cached is not None:
            self._note("cache_hits")
            self.set_phase("response_decode")
            value = json.loads(cached)
            self.check_deadline()
            return value
        self._note("cache_misses")
        self.set_phase("flight_register")
        with self._state_guard():
            with self._flight_lock:
                flight = self._flights.get(flight_key)
                owner = flight is None
                if owner:
                    # A short-lived caller must not lend its remaining budget
                    # to everyone sharing the same upstream operation. Nested
                    # producers still cannot extend a parent producer's budget.
                    producer_deadline = time.monotonic() + self.config.request_timeout
                    parent = getattr(self._local, "flight", None)
                    if parent is not None:
                        producer_deadline = min(producer_deadline, parent.deadline)
                    producer_trace = _Trace(producer_deadline,
                                            tool=trace.tool if trace is not None else None,
                                            request_id=trace.request_id if trace is not None else None)
                    producer_trace.namespace = namespace
                    flight = self._flights[flight_key] = _Flight(producer_deadline, producer_trace)
                inherited = flight.future.done() and not flight.future.cancelled()
                inherited_error = flight.future.exception() if inherited else None
            if owner:
                # Lookup/coalescing is already complete. Only this unique miss
                # gets an upstream execution thread; waiters do not get one.
                context = getattr(self._local, "context", None)
                thread = threading.Thread(
                    target=self._produce_cached,
                    args=(namespace, key, flight_key, flight, producer, ttl,
                          persist, context),
                    name="search-producer", daemon=True,
                )
                flight.timer = threading.Timer(
                    max(0.0, flight.deadline - time.monotonic()),
                    self._expire_flight, args=(flight,))
                flight.timer.daemon = True
                try:
                    flight.timer.start()
                    thread.start()
                except BaseException as exc:
                    flight.cancel(exc)
                    with self._flight_lock:
                        if self._flights.get(flight_key) is flight:
                            self._flights.pop(flight_key)
                    raise
        if not owner:
            self._note("coalesced_waiters")
        self.set_phase("shared_wait")
        if trace is not None:
            trace.set_shared(flight.trace)
        if isinstance(inherited_error, GatewayFailure) and inherited_error.code == "deadline_exceeded":
            if trace is not None:
                trace.add_yibu_usage(flight.trace.yibu_usage())
            raise self._deadline_failure("Previous shared search execution has expired",
                                         shared=flight.trace, scope="inherited",
                                         reason="previous_producer_deadline", code="shared_execution_expired")
        try:
            encoded = flight.future.result(timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError as exc:
            # Future.result also rethrows an exception produced *by* the task.
            # Identity distinguishes it from the wait operation timing out.
            if flight.future.done() and not flight.future.cancelled() and flight.future.exception() is exc:
                raise GatewayFailure("Search suboperation timed out", code="producer_timeout",
                                     status_code=504, retryable=True,
                                     diagnostics=flight.trace.snapshot()) from exc
            # A caller expiring must not cancel/pop a producer still serving
            # other callers, nor permit a second identical upstream operation.
            raise self._deadline_failure("Deadline expired waiting for shared search result",
                                         shared=flight.trace) from None
        except GatewayFailure as exc:
            if exc.code == "deadline_exceeded":
                error = self._deadline_failure(exc.message, shared=flight.trace,
                                               scope=exc.diagnostics.get("deadline_scope", "producer"),
                                               reason=exc.diagnostics.get("deadline_reason"))
                raise error from exc
            raise
        finally:
            # Every coalesced caller receives the same stable attempt IDs.
            # They live on the flight trace, never in the cached JSON value.
            if trace is not None:
                trace.add_yibu_usage(flight.trace.yibu_usage())
        self.set_phase("response_decode")
        value = json.loads(encoded)
        self.check_deadline()
        return value

    def _cache_get(self, namespace, key):
        self.check_deadline()
        try:
            with self.cache.operation(deadline=self.deadline(), phase_callback=self.set_phase):
                cached = self.cache.get(namespace, key)
        except RuntimeError:
            self.check_deadline()  # A concurrent close must stay service_closed.
            raise
        self.check_deadline()
        return cached

    def _expire_flight(self, flight):
        failure = self._deadline_failure("Shared search producer deadline exceeded",
                                         trace=flight.trace, shared=flight.trace.shared_trace(), scope="producer")
        if flight.cancel(failure):
            self.metrics.add("producer_deadlines_exceeded")
            details = failure.diagnostics
            self.metrics.add("producer_deadline_reasons/" + details["deadline_reason"])
            self.metrics.add("producer_deadline_phases/" + details.get("shared_phase", details["phase"]))
            LOG.warning("search_producer_deadline %s", json_dumps({
                "event": "search_producer_deadline", "request_id": flight.trace.request_id,
                "tool": flight.trace.tool, "code": failure.code, **details,
            }))
        # Keep the flight registered until its real execution thread exits.
        # New identical callers see the completed failure without duplicating IO.

    def _produce_cached(self, namespace, key, flight_key, flight, producer, ttl,
                        persist, context):
        self._local.deadline = flight.deadline
        self._local.flight = flight
        self._local.context = context
        self._local.trace = flight.trace
        try:
            # A previous owner may have completed between the first lookup and the lock.
            self.check_deadline()
            cached = self._cache_get(namespace, key) if persist else None
            produced = cached is None
            if cached is None:
                self.metrics.add("producer_executions")
                self.set_phase("producer_start")
                value = producer()
                self.check_deadline()
                self.set_phase("result_encode")
                cached = json_dumps(value)
                effective_ttl = ttl(value) if callable(ttl) else ttl
            # Publish validated material first. Keep the flight until the write
            # finishes, so a miss during write-back reuses this completed value.
            with self._state_guard():
                flight.complete(value=cached)
            if produced and persist:
                try:
                    with self.cache.operation(deadline=flight.deadline, phase_callback=self.set_phase):
                        if self.cache.put(namespace, key, cached, effective_ttl,
                                          before_commit=self.check_deadline):
                            self._note("cache_stores")
                except Exception as exc:
                    self.metrics.add("cache_write_errors")
                    code = exc.code if isinstance(exc, GatewayFailure) else "cache_database_error"
                    self.metrics.add("cache_write_errors/" + code)
                    LOG.warning("cache_write_failed %s", json_dumps({
                        "event": "cache_write_failed", "code": code,
                        "request_id": flight.trace.request_id, **flight.trace.snapshot(),
                    }))
        except TimeoutError as exc:
            flight.complete(error=GatewayFailure("Search suboperation timed out", code="producer_timeout",
                                                 status_code=504, retryable=True,
                                                 diagnostics=flight.trace.snapshot()))
        except BaseException as exc:
            if isinstance(exc, GatewayFailure) and exc.upstream is not None:
                flight.trace.failure(exc)
            flight.complete(error=exc)
        finally:
            flight.timer.cancel()
            with self._flight_lock:
                if self._flights.get(flight_key) is flight:
                    self._flights.pop(flight_key)
            flight.finished.set()
            self._local.__dict__.clear()

    def _budget(self, upstream, url):
        hostname = (urlsplit(url).hostname or "").lower()
        if upstream in {"wiki", "wikipedia"} or hostname.endswith(".wikipedia.org") or hostname == "wikipedia.org":
            # Classify metrics only; there are no service-wide upstream quotas.
            return "wikipedia", None
        elif upstream in {"brave", "yibu"}:
            return "brave", None
        return "web:" + hostname, None

    def _sleep(self, delay):
        if delay >= self.remaining():
            raise GatewayFailure("Retry wait exceeds search deadline", code="deadline_exceeded",
                                 status_code=504, retryable=True)
        flight = getattr(self._local, "flight", None)
        if flight is None:
            time.sleep(delay)
        else:
            flight.cancelled.wait(delay)
        self.check_deadline()

    def notify_cooldown(self, upstream, url, delay):
        """Compatibility hook; API throttling no longer pauses other requests."""
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
            raise ValueError("cooldown must be finite and nonnegative")

    @contextmanager
    def _abort_on_cancel(self, abort):
        flight = getattr(self._local, "flight", None)
        token = flight.register_abort(abort) if flight is not None and callable(abort) else None
        try:
            yield
        finally:
            if token is not None:
                flight.unregister_abort(token)

    def request(self, upstream, method, url, **kwargs):
        owns_deadline = not hasattr(self._local, "deadline")
        owns_trace = not hasattr(self._local, "trace")
        if owns_deadline:
            self._local.deadline = time.monotonic() + self.config.request_timeout
        if owns_trace:
            self._local.trace = _Trace(self.deadline())
        try:
            return self._request(upstream, method, url, **kwargs)
        finally:
            if owns_deadline:
                del self._local.deadline
            if owns_trace:
                del self._local.trace

    def _request(self, upstream, method, url, **kwargs):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise GatewayFailure("Invalid upstream URL", code="invalid_request", status_code=400)
        self.check_url_allowed(url)
        current_url, current_method = url, method.upper()
        options_base = dict(kwargs)
        options_base.pop("allow_redirects", None)
        total_attempts = 0
        usage_call_id = uuid.uuid4().hex
        with self._session_factory() as session, self._abort_on_cancel(getattr(session, "close", None)):
            # The server's existing proxy route is the only external network exit.
            session.trust_env = True
            for redirects in range(11):
                name, _ = self._budget(upstream, current_url)
                retry_statuses = {429, 500, 502, 503, 504} | ({422} if name == "brave" else set())
                next_url = None
                last_failure = None
                for attempt in range(self.config.max_retries + 1):
                    try:
                        self.check_deadline()
                    except GatewayFailure as exc:
                        if exc.code == "deadline_exceeded" and last_failure is not None:
                            raise last_failure from exc
                        raise
                    response = None
                    delay = None
                    was_429 = False
                    watchdog = None
                    abort_token = None
                    flight = getattr(self._local, "flight", None)
                    failure = None
                    retry_after = None
                    attempt_started = None
                    attempt_started_at = None
                    attempt_error_type = None
                    try:
                        remaining = self.remaining()
                        options = dict(options_base)
                        requested_timeout = options.pop("timeout", (self.config.connect_timeout, self.config.read_timeout))
                        if isinstance(requested_timeout, (int, float)):
                            requested_timeout = (requested_timeout, requested_timeout)
                        options["timeout"] = tuple(max(0.001, min(float(v), remaining)) for v in requested_timeout)
                        options.update(stream=True, allow_redirects=False)
                        self._note("upstream_attempts")
                        self.metrics.add("upstream_attempts/" + name)
                        total_attempts += 1
                        attempt_started = time.monotonic()
                        attempt_started_at = datetime.now(_YIBU_TZ).isoformat(timespec="milliseconds")
                        self.set_phase("upstream_headers")
                        response = session.request(current_method, current_url, **options)
                        # Observe status before checking expiry: a late 429 must
                        # remain visible even if the caller/watchdog already left.
                        status = response.status_code
                        if status == 429:
                            was_429 = True
                            self.metrics.add("upstream_429/" + name)
                            retry_after = _retry_after(response.headers.get("Retry-After"))
                            delay = (2 ** attempt) + random.uniform(0.0, 0.5) if retry_after is None else retry_after + 0.5
                        elif status in retry_statuses:
                            delay = float(2 ** attempt)
                        if not 200 <= status < 300 and status not in {301, 302, 303, 307, 308}:
                            failure = GatewayFailure(
                                f"Upstream returned HTTP {status}",
                                code="upstream_proxy_error" if status == 407 else "upstream_http_error",
                                status_code=429 if status == 429 else 502, retryable=status in retry_statuses,
                                upstream=name, upstream_status=status, retry_after=delay if was_429 else None,
                                attempts=total_attempts,
                            )
                            self.metrics.add("upstream_errors/" + failure.code)
                        trace = getattr(self._local, "trace", None)
                        if trace is not None:
                            trace.observe(name, status, total_attempts, failure)
                        if failure is not None and (self.remaining() <= 0 or (
                                flight is not None and flight.cancelled.is_set())):
                            self.metrics.add("late_upstream_status/" + str(status))
                            LOG.warning("search_late_upstream %s", json_dumps({
                                "event": "search_late_upstream", "request_id": trace.request_id,
                                "upstream": name, "upstream_status": status,
                                **trace.snapshot(),
                            }))
                        if flight is not None:
                            abort_token = flight.register_abort(lambda response=response: _abort_response(response))
                        watchdog = threading.Timer(self.remaining(), _abort_response, args=(response,))
                        watchdog.daemon = True
                        watchdog.start()
                        self.check_deadline()
                        if status in {301, 302, 303, 307, 308} and response.headers.get("Location"):
                            next_url = urljoin(current_url, response.headers["Location"])
                            redirect_parts = urlsplit(next_url)
                            if redirect_parts.scheme not in {"http", "https"} or not redirect_parts.hostname or redirect_parts.username or redirect_parts.password:
                                raise GatewayFailure("Invalid upstream redirect", code="upstream_redirect_error", upstream=name)
                            self.check_url_allowed(next_url)
                            if redirect_parts.netloc != urlsplit(current_url).netloc:
                                headers = dict(options_base.get("headers", {}))
                                options_base["headers"] = {k: v for k, v in headers.items() if k.lower() not in {"authorization", "cookie", "host"}}
                                options_base.pop("auth", None)
                                options_base.pop("cookies", None)
                            if status == 303 and current_method != "HEAD" or status == 302 and current_method != "HEAD" or status == 301 and current_method == "POST":
                                current_method = "GET"
                                for key in ("data", "json", "files"):
                                    options_base.pop(key, None)
                            options_base.pop("params", None)
                            self.metrics.add("upstream_redirects")
                            break
                        if not 200 <= status < 300:
                            if failure is None:
                                failure = GatewayFailure(f"Upstream returned HTTP {status}",
                                                         code="upstream_http_error", upstream=name,
                                                         upstream_status=status, attempts=total_attempts)
                            # Match the old Web/Yibu fast-fail policy. A long
                            # Retry-After affects this request only, never peers.
                            long_wait = was_429 and name != "wikipedia" and retry_after is not None and retry_after > 30
                            if status not in retry_statuses or attempt == self.config.max_retries or long_wait:
                                raise failure
                        else:
                            self.set_phase("upstream_body")
                            chunks, size = [], 0
                            for chunk in response.iter_content(chunk_size=4096):
                                self.check_deadline()
                                size += len(chunk)
                                if size > self.config.max_upstream_bytes:
                                    raise GatewayFailure("Upstream response exceeds size limit", code="upstream_too_large",
                                                         status_code=502, upstream=name)
                                chunks.append(chunk)
                            response._content = b"".join(chunks)
                            response._content_consumed = True
                            self.check_deadline()
                            self.set_phase("response_decode")
                            return response
                    except requests.exceptions.ProxyError as exc:
                        attempt_error_type = type(exc).__name__
                        status = _proxy_http_status(exc)
                        retryable = status not in {400, 401, 403, 407}
                        failure = GatewayFailure(
                            "Upstream proxy connection failed" + (f" (HTTP {status})" if status else ""),
                            code="upstream_proxy_error", status_code=502, retryable=retryable,
                            upstream=name, upstream_status=status, attempts=total_attempts,
                        )
                        self.metrics.add("upstream_errors/upstream_proxy_error")
                        self._local.trace.failure(failure)
                        if not retryable or attempt == self.config.max_retries or self.remaining() <= 0:
                            raise failure from exc
                        delay = float(2 ** attempt)
                    except requests.exceptions.SSLError as exc:
                        attempt_error_type = type(exc).__name__
                        self.metrics.add("upstream_errors/upstream_tls_error")
                        failure = GatewayFailure("Upstream TLS connection failed", code="upstream_tls_error",
                                                 status_code=502, retryable=False, upstream=name,
                                                 attempts=total_attempts)
                        self._local.trace.failure(failure)
                        raise failure from exc
                    except (requests.Timeout, requests.ConnectionError) as exc:
                        attempt_error_type = type(exc).__name__
                        timed_out = _is_request_timeout(exc)
                        failure = GatewayFailure(
                            "Upstream HTTP request timed out" if timed_out else "Upstream network connection failed",
                            code="upstream_timeout" if timed_out else "upstream_network_error",
                            status_code=504 if timed_out else 502, retryable=True,
                            upstream=name, attempts=total_attempts,
                        )
                        self.metrics.add("upstream_errors/" + failure.code)
                        self._local.trace.failure(failure)
                        if attempt == self.config.max_retries or self.remaining() <= 0:
                            raise failure from exc
                        delay = float(2 ** attempt)
                    except requests.RequestException as exc:
                        attempt_error_type = type(exc).__name__
                        self.metrics.add("upstream_errors/upstream_request_error")
                        failure = GatewayFailure("Upstream request failed", code="upstream_request_error",
                                                 status_code=502, upstream=name, attempts=total_attempts)
                        self._local.trace.failure(failure)
                        if self.remaining() <= 0:
                            raise self._deadline_failure(scope="producer" if flight is not None else "caller") from exc
                        raise failure from exc
                    finally:
                        if watchdog is not None:
                            watchdog.cancel()
                        if name == "brave" and attempt_started is not None:
                            self._record_yibu_attempt(
                                usage_call_id, total_attempts, options_base,
                                response, failure, attempt_error_type,
                                attempt_started, attempt_started_at,
                            )
                        try:
                            if response is not None:
                                response.close()
                        finally:
                            if flight is not None and abort_token is not None:
                                flight.unregister_abort(abort_token)
                    last_failure = failure
                    # Every upstream's retry wait belongs to this request only.
                    if delay is not None:
                        if failure is not None and delay >= self.remaining():
                            raise failure
                        try:
                            self.set_phase("retry_wait")
                            self._sleep(delay)
                        except GatewayFailure as exc:
                            if exc.code == "deadline_exceeded" and failure is not None:
                                raise failure from exc
                            raise
                if next_url is None:
                    break
                if redirects == 10:
                    raise GatewayFailure("Too many upstream redirects", code="upstream_redirect_error", attempts=total_attempts)
                current_url = next_url
        raise GatewayFailure("Upstream request failed")

    def _record_yibu_attempt(self, call_id, attempt, options, response, failure,
                             error_type, started, started_at):
        """Capture already-read data only; telemetry cannot trigger network IO."""
        try:
            trace = getattr(self._local, "trace", None)
            if trace is None:
                return
            params = options.get("params", {})
            query = params.get("q", "") if isinstance(params, dict) else ""
            status = response.status_code if response is not None else None
            body = None
            # On errors/redirects stream=True may leave the body unread. Access
            # private buffered bytes, never response.text/content or iter_content.
            if response is not None and response.__dict__.get("_content_consumed") is True:
                raw = response.__dict__.get("_content")
                if isinstance(raw, bytes):
                    encoding = response.__dict__.get("encoding") or "utf-8"
                    try:
                        body = raw.decode(encoding, errors="replace")
                    except LookupError:
                        body = raw.decode("utf-8", errors="replace")
            outcome = "success" if status is not None and 200 <= status < 300 and body is not None else (
                "http_error" if status is not None and status >= 400 else "network_error")
            event = {
                "call_id": call_id, "event_id": f"{call_id}:{attempt}",
                "attempt": attempt, "input_text": query, "query": query,
                "source": "shared_service", "event_type": "upstream_attempt", "http_attempt": True,
                "status": outcome, "http_status": status,
                "error_type": error_type or (failure.code if failure is not None else None),
                "started_at": started_at,
                "finished_at": datetime.now(_YIBU_TZ).isoformat(timespec="milliseconds"),
                "elapsed_s": round(max(0.0, time.monotonic() - started), 6),
                "response_body_text": body,
            }
            trace.add_yibu_usage([event])
        except Exception:
            # Like the main tool logger, accounting must not retry/fail a search.
            self.metrics.add("yibu_usage_record_errors")

    def _execute(self, tool, params, deadline, queued_at, request_id, trace=None):
        previous = dict(self._local.__dict__)
        self._local.deadline = deadline
        self._local.context = {}
        trace = trace or _Trace(deadline, tool=tool, request_id=request_id,
                                source="client" if deadline - queued_at < self.config.request_timeout - 0.01 else "service")
        self._local.trace = trace
        self.metrics.add("unique_executions")
        started = time.monotonic()
        try:
            self.check_deadline()
            result = self.providers.execute(tool, params, self)
            self.check_deadline()
            self.set_phase("result_encode")
            result = json_clone(result)
            self.check_deadline()
            provider_meta = result.pop("_meta", {})
            if not isinstance(provider_meta, dict):
                provider_meta = {}
            meta = dict(provider_meta)
            meta.update(self._local.context)
            meta.update(request_id=request_id, queue_wait_seconds=0.0,
                        duration_seconds=round(time.monotonic() - started, 6))
            usage_events = trace.yibu_usage()
            if usage_events:
                meta["yibu_usage_events"] = usage_events
            result["meta"] = meta
            if trace.claim_outcome():
                self.metrics.add("requests_succeeded")
            return result
        except GatewayFailure as exc:
            failure = self._decorate_failure(exc, trace).with_yibu_usage(trace.yibu_usage())
            self._record_failure(failure, trace)
            if failure is exc:
                raise
            raise failure from exc
        except TimeoutError as exc:
            failure = GatewayFailure("Search suboperation timed out", code="producer_timeout",
                                     status_code=504, retryable=True, diagnostics=trace.snapshot(),
                                     yibu_usage_events=trace.yibu_usage())
            self._record_failure(failure, trace)
            raise failure from exc
        except Exception as exc:
            failure = GatewayFailure("Search provider failed", code="provider_error", status_code=502,
                                     yibu_usage_events=trace.yibu_usage())
            self._record_failure(failure, trace)
            raise failure from exc
        finally:
            self._local.__dict__.clear()
            self._local.__dict__.update(previous)

    def _submit(self, tool, prepared, deadline, request_id, *, deadline_source="client", started_at=None):
        # Batch multiplexing only; regular HTTP requests run in their handler.
        # There is no shared worker pool, admission semaphore or waiting queue.
        if self._closed:
            raise self._closed_failure()
        future = Future()
        queued_at = time.monotonic() if started_at is None else started_at
        trace = _Trace(deadline, tool=tool, request_id=request_id,
                       source=deadline_source, started=queued_at)
        future.search_trace = trace

        def run():
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(self._execute(tool, prepared, deadline,
                                               queued_at, request_id, trace))
            except BaseException as exc:
                future.set_exception(exc)

        threading.Thread(target=run, name="search-batch", daemon=True).start()
        return future

    def _wait(self, future, deadline):
        try:
            return future.result(timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError as exc:
            if future.done() and not future.cancelled() and future.exception() is exc:
                raise GatewayFailure("Search suboperation timed out", code="producer_timeout",
                                     status_code=504, retryable=True) from exc
            future.cancel()
            trace = getattr(future, "search_trace", None)
            failure = self._deadline_failure("Search batch request deadline exceeded", trace=trace,
                                             shared=trace.shared_trace() if trace is not None else None)
            if trace is not None:
                shared = trace.shared_trace()
                usage = trace.yibu_usage() + (shared.yibu_usage() if shared is not None else [])
                failure = failure.with_yibu_usage(usage)
                self._record_failure(failure, trace)
            raise failure from None

    def handle(self, tool, params, *, deadline=None, request_id=None, deadline_source=None, started_at=None):
        source = deadline_source or ("service" if deadline is None else "client")
        deadline = deadline or time.monotonic() + self.config.request_timeout
        request_id = request_id or uuid.uuid4().hex
        self.metrics.add("logical_requests")
        prepared = self._prepare(tool, params)
        trace = _Trace(deadline, tool=tool, request_id=request_id, source=source, started=started_at)
        return self._execute(tool, prepared, deadline, trace.started, request_id, trace)

    def batch(self, requests_list, *, deadline=None, request_id=None, deadline_source=None, started_at=None):
        if not isinstance(requests_list, list) or not requests_list or len(requests_list) > self.config.max_batch:
            raise GatewayFailure(f"requests must contain 1..{self.config.max_batch} entries",
                                 code="invalid_request", status_code=400)
        source = deadline_source or ("service" if deadline is None else "client")
        deadline = deadline or time.monotonic() + self.config.request_timeout
        request_id = request_id or uuid.uuid4().hex
        self.metrics.add("logical_requests", len(requests_list))
        unique, order = {}, []
        for item in requests_list:
            try:
                if not isinstance(item, dict) or set(item) - {"tool", "params"}:
                    raise GatewayFailure("Invalid batch entry", code="invalid_request", status_code=400)
                tool, params = item.get("tool"), item.get("params")
                prepared = self._prepare(tool, params)
                key = json_dumps([tool, prepared], canonical=True)
                if key not in unique:
                    try:
                        unique[key] = self._submit(tool, prepared, deadline, request_id,
                                                   deadline_source=source, started_at=started_at)
                    except GatewayFailure as exc:
                        unique[key] = exc
                else:
                    self.metrics.add("batch_duplicates")
                order.append(key)
            except GatewayFailure as exc:
                order.append(exc)
        resolved = {}
        for key, future in unique.items():
            try:
                if isinstance(future, GatewayFailure):
                    raise future
                resolved[key] = {"ok": True, "data": self._wait(future, deadline)}
            except GatewayFailure as exc:
                resolved[key] = {"ok": False, "error": exc.as_dict()}
                if exc.yibu_usage_events:
                    resolved[key]["yibu_usage_events"] = exc.yibu_usage_events
        return {"results": [json_clone({"ok": False, "error": key.as_dict()} if isinstance(key, GatewayFailure)
                                      else resolved[key]) for key in order]}

    def metrics_snapshot(self):
        result = self.metrics.snapshot()
        result["cache"] = self.cache.stats()
        with self._flight_lock:
            result["inflight_keys"] = len(self._flights)
        return result
