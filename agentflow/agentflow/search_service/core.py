"""Persistence, request coalescing, and upstream budgets for one service process.

No planner, embedding, or training dependencies are imported here. The service
owns all network attempts (including retries); tool clients must not retry by
calling an external provider directly when this service fails.
"""
from __future__ import annotations

import email.utils
import fcntl
import hashlib
import json
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
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

import requests


CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_CACHE_DB_PATH = "/var/tmp/agentflow-search-cache/search_cache.sqlite3"


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
                 retry_after: float | None = None, attempts: int | None = None):
        self.message = " ".join(str(message).split())[:500]
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.upstream = upstream
        self.upstream_status = upstream_status
        self.retry_after = retry_after
        self.attempts = attempts
        super().__init__(self.message)

    def as_dict(self) -> dict:
        result = {"code": self.code, "message": self.message, "retryable": self.retryable}
        for name in ("upstream", "upstream_status", "retry_after", "attempts"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


def json_dumps(value: Any, *, canonical: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=canonical, separators=(",", ":"))


def json_clone(value: Any) -> Any:
    return json.loads(json_dumps(value))


@dataclass
class ServiceConfig:
    cache_dir: str = "/home/ma-user/work/code-rl/cache"
    token: str = ""
    # Legacy constructor fields: no pool/queue or Wiki-wide limiter uses them.
    workers: int = 32
    queue_size: int = 128
    request_timeout: float = 120.0
    max_batch: int = 128
    max_body_bytes: int = 8 * 1024 * 1024
    max_upstream_bytes: int = 20 * 1024 * 1024
    cache_max_bytes: int = 2 * 1024 * 1024 * 1024
    cache_max_ttl: float = float(CACHE_TTL_SECONDS)
    # Retained only for callers constructing old configs; Wiki ignores these.
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
                     "cache_max_bytes", "brave_concurrency",
                     "web_concurrency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_retries",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("request_timeout", "cache_max_ttl", "brave_rpm",
                     "web_rpm", "connect_timeout", "read_timeout"):
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
            "cache_max_bytes": "SEARCH_CACHE_MAX_BYTES", "cache_max_ttl": "SEARCH_CACHE_MAX_TTL",
            "brave_rpm": "SEARCH_BRAVE_RPM", "brave_concurrency": "SEARCH_BRAVE_CONCURRENCY",
            "web_rpm": "SEARCH_WEB_RPM", "web_concurrency": "SEARCH_WEB_CONCURRENCY",
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


class SQLiteCache:
    """Serialize local DB operations; reads update LRU access, never expiry."""

    def __init__(self, path, *, max_bytes, max_ttl):
        self.path = str(path)
        self.max_bytes = max_bytes
        self.max_ttl = max_ttl
        # Cache hits also write access times. Serializing connections
        # avoids unnecessary writer contention and overlapping WAL checkpoints.
        self._db_lock = threading.RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS entries (
                namespace TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
                expires REAL NOT NULL, accessed REAL NOT NULL, size INTEGER NOT NULL,
                ttl REAL NOT NULL DEFAULT 604800,
                PRIMARY KEY(namespace, key))""")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(entries)")}
            if "ttl" not in columns:
                # Keep healthy legacy schemas usable without changing expiry.
                conn.execute("ALTER TABLE entries ADD COLUMN ttl REAL NOT NULL DEFAULT 604800")
            conn.execute("CREATE INDEX IF NOT EXISTS entries_expiry ON entries(expires)")
            conn.execute("CREATE INDEX IF NOT EXISTS entries_access ON entries(accessed)")

    @contextmanager
    def connect(self):
        with self._db_lock:
            conn = sqlite3.connect(self.path, timeout=5.0)
            try:
                conn.execute("PRAGMA busy_timeout=5000")
                with conn:
                    yield conn
            finally:
                conn.close()

    def get(self, namespace, key):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            row = conn.execute("SELECT value, expires FROM entries WHERE namespace=? AND key=?",
                               (namespace, key)).fetchone()
            if row is None:
                return None
            if row[1] <= now:
                conn.execute("DELETE FROM entries WHERE namespace=? AND key=? AND expires<=?",
                             (namespace, key, now))
                return None
            conn.execute("UPDATE entries SET accessed=? WHERE namespace=? AND key=?",
                         (now, namespace, key))
        return row[0]

    def put(self, namespace, key, value_json, ttl):
        if isinstance(ttl, bool) or not isinstance(ttl, (float, int)) or not math.isfinite(ttl):
            raise ValueError("cache ttl must be finite")
        if ttl <= 0:
            return False
        size = len(value_json.encode("utf-8"))
        if size > self.max_bytes:
            return False
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = time.time()
            lifetime = min(float(ttl), self.max_ttl)
            conn.execute("DELETE FROM entries WHERE expires<=?", (now,))
            conn.execute("""INSERT INTO entries (namespace, key, value, expires, accessed, size, ttl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET value=excluded.value,
                expires=excluded.expires, accessed=excluded.accessed, size=excluded.size,
                ttl=excluded.ttl""",
                (namespace, key, value_json, now + lifetime, now, size, lifetime))
            total = conn.execute("SELECT COALESCE(SUM(size),0) FROM entries").fetchone()[0]
            if total > self.max_bytes:
                for old_namespace, old_key, old_size in conn.execute(
                    "SELECT namespace,key,size FROM entries ORDER BY accessed,namespace,key"
                ).fetchall():
                    if total <= self.max_bytes:
                        break
                    conn.execute("DELETE FROM entries WHERE namespace=? AND key=?", (old_namespace, old_key))
                    total -= old_size
        return True

    def stats(self):
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*),COALESCE(SUM(size),0) FROM entries WHERE expires>?",
                               (time.time(),)).fetchone()
        return {"entries": row[0], "bytes": row[1]}


class _RequestBudget:
    """Paced starts, bounded active requests, and a shared 429 cooldown."""

    def __init__(self, rpm, concurrency):
        self.interval = 60.0 / rpm
        self.concurrency = concurrency
        self.condition = threading.Condition()
        self.active = 0
        self.next_start = 0.0
        self.cooldown_until = 0.0

    def acquire(self, deadline):
        with self.condition:
            while True:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    raise GatewayFailure("Deadline expired while waiting for upstream budget",
                                         code="deadline_exceeded", status_code=504, retryable=True)
                ready = max(self.next_start, self.cooldown_until)
                if self.active < self.concurrency and ready <= now:
                    self.active += 1
                    self.next_start = now + self.interval
                    return
                wait = max(0.001, ready - now) if self.active < self.concurrency else remaining
                self.condition.wait(min(wait, remaining))

    def release(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def cooldown(self, delay):
        with self.condition:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + delay)
            self.condition.notify_all()


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
        self._budgets = {}
        self._budget_lock = threading.Lock()
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
            self._process_lock.close()
            raise

    def close(self):
        # Never wait indefinitely for a blocked network thread. Cache access
        # shares this lock, so old producers cannot write after DB ownership is
        # released. Their daemon threads finish cooperatively at their deadline.
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            with self._flight_lock:
                pending = list(self._flights.values())
            for future in pending:
                if not future.done():
                    future.set_exception(self._closed_failure())
            fcntl.flock(self._process_lock, fcntl.LOCK_UN)
            self._process_lock.close()

    @staticmethod
    def _closed_failure():
        return GatewayFailure("Search service is closed", code="service_closed",
                              status_code=503, retryable=True)

    def deadline(self):
        return getattr(self._local, "deadline", time.monotonic() + self.config.request_timeout)

    def remaining(self):
        return max(0.0, self.deadline() - time.monotonic())

    def check_deadline(self):
        if self._closed:
            raise self._closed_failure()
        if self.remaining() <= 0:
            raise GatewayFailure("Search request deadline exceeded", code="deadline_exceeded",
                                 status_code=504, retryable=True)

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

    def cached(self, namespace, params, producer: Callable, ttl, persist=True):
        if not isinstance(namespace, str) or not namespace or len(namespace) > 128:
            raise ValueError("invalid cache namespace")
        self.check_deadline()
        deadline = self.deadline()
        key = hashlib.sha256(json_dumps(params, canonical=True).encode("utf-8")).hexdigest()
        flight_key = (namespace, key)
        with self._state_lock:
            self.check_deadline()
            cached = self.cache.get(namespace, key) if persist else None
        if cached is not None:
            self._note("cache_hits")
            return json.loads(cached)
        self._note("cache_misses")
        with self._state_lock:
            self.check_deadline()
            with self._flight_lock:
                future = self._flights.get(flight_key)
                owner = future is None
                if owner:
                    future = self._flights[flight_key] = Future()
            if owner:
                # Lookup/coalescing is already complete. Only this unique miss
                # gets an upstream execution thread; waiters do not get one.
                context = getattr(self._local, "context", None)
                thread = threading.Thread(
                    target=self._produce_cached,
                    args=(namespace, key, flight_key, future, producer, ttl,
                          persist, deadline, context),
                    name="search-producer", daemon=True,
                )
                try:
                    thread.start()
                except BaseException:
                    with self._flight_lock:
                        if self._flights.get(flight_key) is future:
                            self._flights.pop(flight_key)
                    raise
        if not owner:
            self._note("coalesced_waiters")
        try:
            return json.loads(future.result(timeout=max(0.0, deadline - time.monotonic())))
        except TimeoutError:
            # A caller expiring must not cancel/pop a producer still serving
            # other callers, nor permit a second identical upstream operation.
            raise GatewayFailure("Deadline expired waiting for shared search result",
                                 code="deadline_exceeded", status_code=504, retryable=True) from None

    def _produce_cached(self, namespace, key, flight_key, future, producer, ttl,
                        persist, deadline, context):
        self._local.deadline = deadline
        self._local.context = context
        try:
            # A previous owner may have completed between the first lookup and the lock.
            with self._state_lock:
                self.check_deadline()
                cached = self.cache.get(namespace, key) if persist else None
            if cached is None:
                self.metrics.add("producer_executions")
                value = producer()
                self.check_deadline()
                cached = json_dumps(value)
                effective_ttl = ttl(value) if callable(ttl) else ttl
                with self._state_lock:
                    self.check_deadline()
                    if persist and self.cache.put(namespace, key, cached, effective_ttl):
                        self._note("cache_stores")
            with self._state_lock:
                self.check_deadline()
                if not future.done():
                    future.set_result(cached)
        except BaseException as exc:
            with self._state_lock:
                if not future.done():
                    future.set_exception(exc)
        finally:
            with self._flight_lock:
                if self._flights.get(flight_key) is future:
                    self._flights.pop(flight_key)
            self._local.__dict__.clear()

    def _budget(self, upstream, url):
        hostname = (urlsplit(url).hostname or "").lower()
        if upstream in {"wiki", "wikipedia"} or hostname.endswith(".wikipedia.org") or hostname == "wikipedia.org":
            # Wiki has no service-wide RPM, active-request limit or cooldown.
            return "wikipedia", None
        elif upstream in {"brave", "yibu"}:
            name, rpm, concurrency = "brave", self.config.brave_rpm, self.config.brave_concurrency
        else:
            name, rpm, concurrency = "web:" + hostname, self.config.web_rpm, self.config.web_concurrency
        with self._budget_lock:
            return name, self._budgets.setdefault(name, _RequestBudget(rpm, concurrency))

    def _sleep(self, delay):
        if delay >= self.remaining():
            raise GatewayFailure("Retry wait exceeds search deadline", code="deadline_exceeded",
                                 status_code=504, retryable=True)
        time.sleep(delay)

    def notify_cooldown(self, upstream, url, delay):
        """Providers can report API-level throttling carried inside HTTP 200 JSON."""
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
            raise ValueError("cooldown must be finite and nonnegative")
        name, budget = self._budget(upstream, url)
        if budget is None:
            return
        budget.cooldown(float(delay))
        self.metrics.add("api_cooldowns/" + name)

    def request(self, upstream, method, url, **kwargs):
        owns_deadline = not hasattr(self._local, "deadline")
        if owns_deadline:
            self._local.deadline = time.monotonic() + self.config.request_timeout
        try:
            return self._request(upstream, method, url, **kwargs)
        finally:
            if owns_deadline:
                del self._local.deadline

    def _request(self, upstream, method, url, **kwargs):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise GatewayFailure("Invalid upstream URL", code="invalid_request", status_code=400)
        self.check_url_allowed(url)
        current_url, current_method = url, method.upper()
        options_base = dict(kwargs)
        options_base.pop("allow_redirects", None)
        total_attempts = 0
        with self._session_factory() as session:
            # The server's existing proxy route is the only external network exit.
            session.trust_env = True
            for redirects in range(11):
                name, budget = self._budget(upstream, current_url)
                retry_statuses = {429, 500, 502, 503, 504} | ({422} if name == "brave" else set())
                next_url = None
                last_failure = None
                for attempt in range(self.config.max_retries + 1):
                    try:
                        self.check_deadline()
                        if budget is not None:
                            budget.acquire(self.deadline())
                    except GatewayFailure as exc:
                        if exc.code == "deadline_exceeded" and last_failure is not None:
                            raise last_failure from exc
                        raise
                    response = None
                    delay = None
                    was_429 = False
                    watchdog = None
                    failure = None
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
                        response = session.request(current_method, current_url, **options)
                        watchdog = threading.Timer(self.remaining(), _abort_response, args=(response,))
                        watchdog.daemon = True
                        watchdog.start()
                        self.check_deadline()
                        status = response.status_code
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
                        if status == 429:
                            was_429 = True
                            self.metrics.add("upstream_429/" + name)
                            delay = _retry_after(response.headers.get("Retry-After"))
                            delay = (2 ** attempt) + random.uniform(0.0, 0.5) if delay is None else delay + 0.5
                            if budget is not None:
                                budget.cooldown(delay)
                        elif status in retry_statuses:
                            delay = float(2 ** attempt)
                        if not 200 <= status < 300:
                            failure = GatewayFailure(
                                f"Upstream returned HTTP {status}",
                                code="upstream_proxy_error" if status == 407 else "upstream_http_error",
                                status_code=429 if status == 429 else 502, retryable=status in retry_statuses,
                                upstream=name, upstream_status=status, retry_after=delay if was_429 else None,
                                attempts=total_attempts,
                            )
                            self.metrics.add("upstream_errors/" + failure.code)
                            if status not in retry_statuses or attempt == self.config.max_retries:
                                raise failure
                        else:
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
                            return response
                    except requests.exceptions.ProxyError as exc:
                        status = _proxy_http_status(exc)
                        retryable = status not in {400, 401, 403, 407}
                        failure = GatewayFailure(
                            "Upstream proxy connection failed" + (f" (HTTP {status})" if status else ""),
                            code="upstream_proxy_error", status_code=502, retryable=retryable,
                            upstream=name, upstream_status=status, attempts=total_attempts,
                        )
                        self.metrics.add("upstream_errors/upstream_proxy_error")
                        if not retryable or attempt == self.config.max_retries or self.remaining() <= 0:
                            raise failure from exc
                        delay = float(2 ** attempt)
                    except requests.exceptions.SSLError as exc:
                        self.metrics.add("upstream_errors/upstream_tls_error")
                        raise GatewayFailure("Upstream TLS connection failed", code="upstream_tls_error",
                                             status_code=502, retryable=False, upstream=name,
                                             attempts=total_attempts) from exc
                    except (requests.Timeout, requests.ConnectionError) as exc:
                        timed_out = isinstance(exc, requests.Timeout)
                        failure = GatewayFailure(
                            "Upstream HTTP request timed out" if timed_out else "Upstream network connection failed",
                            code="upstream_timeout" if timed_out else "upstream_network_error",
                            status_code=504 if timed_out else 502, retryable=True,
                            upstream=name, attempts=total_attempts,
                        )
                        self.metrics.add("upstream_errors/" + failure.code)
                        if attempt == self.config.max_retries or self.remaining() <= 0:
                            raise failure from exc
                        delay = float(2 ** attempt)
                    except requests.RequestException as exc:
                        self.check_deadline()
                        self.metrics.add("upstream_errors/upstream_request_error")
                        raise GatewayFailure("Upstream request failed", code="upstream_request_error",
                                             status_code=502, upstream=name, attempts=total_attempts) from exc
                    finally:
                        if watchdog is not None:
                            watchdog.cancel()
                        if response is not None:
                            response.close()
                        if budget is not None:
                            budget.release()
                    last_failure = failure
                    # Wiki retries back off only this request. Brave/Web 429s
                    # continue waiting through their existing shared budgets.
                    if delay is not None and (not was_429 or budget is None):
                        if failure is not None and delay >= self.remaining():
                            raise failure
                        try:
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

    def _execute(self, tool, params, deadline, queued_at, request_id):
        previous = dict(self._local.__dict__)
        self._local.deadline = deadline
        self._local.context = {}
        self.metrics.add("unique_executions")
        started = time.monotonic()
        try:
            self.check_deadline()
            result = self.providers.execute(tool, params, self)
            self.check_deadline()
            result = json_clone(result)
            provider_meta = result.pop("_meta", {})
            if not isinstance(provider_meta, dict):
                provider_meta = {}
            meta = dict(provider_meta)
            meta.update(self._local.context)
            meta.update(request_id=request_id, queue_wait_seconds=0.0,
                        duration_seconds=round(time.monotonic() - started, 6))
            result["meta"] = meta
            self.metrics.add("requests_succeeded")
            return result
        except GatewayFailure as exc:
            self.metrics.add("requests_failed")
            self.metrics.add("requests_failed/" + exc.code)
            raise
        except Exception as exc:
            self.metrics.add("requests_failed")
            raise GatewayFailure("Search provider failed", code="provider_error", status_code=502) from exc
        finally:
            self._local.__dict__.clear()
            self._local.__dict__.update(previous)

    def _submit(self, tool, prepared, deadline, request_id):
        # Batch multiplexing only; regular HTTP requests run in their handler.
        # There is no shared worker pool, admission semaphore or waiting queue.
        if self._closed:
            raise self._closed_failure()
        future = Future()

        def run():
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(self._execute(tool, prepared, deadline,
                                               time.monotonic(), request_id))
            except BaseException as exc:
                future.set_exception(exc)

        threading.Thread(target=run, name="search-batch", daemon=True).start()
        return future

    @staticmethod
    def _wait(future, deadline):
        try:
            return future.result(timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            future.cancel()
            raise GatewayFailure("Search request deadline exceeded", code="deadline_exceeded",
                                 status_code=504, retryable=True) from None

    def handle(self, tool, params, *, deadline=None, request_id=None):
        deadline = deadline or time.monotonic() + self.config.request_timeout
        request_id = request_id or uuid.uuid4().hex
        self.metrics.add("logical_requests")
        prepared = self._prepare(tool, params)
        return self._execute(tool, prepared, deadline, time.monotonic(), request_id)

    def batch(self, requests_list, *, deadline=None, request_id=None):
        if not isinstance(requests_list, list) or not requests_list or len(requests_list) > self.config.max_batch:
            raise GatewayFailure(f"requests must contain 1..{self.config.max_batch} entries",
                                 code="invalid_request", status_code=400)
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
                        unique[key] = self._submit(tool, prepared, deadline, request_id)
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
        return {"results": [json_clone({"ok": False, "error": key.as_dict()} if isinstance(key, GatewayFailure)
                                      else resolved[key]) for key in order]}

    def metrics_snapshot(self):
        result = self.metrics.snapshot()
        result["cache"] = self.cache.stats()
        with self._flight_lock:
            result["inflight_keys"] = len(self._flights)
        return result
