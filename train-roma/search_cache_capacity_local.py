"""Offline capacity probes for the *real* search-service implementation.

All databases, listeners and providers in this module are isolated and temporary.
No provider credentials, deployment env files or external URLs are used. Results
describe the offered synthetic workload, not an Internet provider's quota.
"""
from __future__ import annotations

import gzip
import base64
import http.client
import json
import math
import random
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "agentflow"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from agentflow.search_service.core import CACHE_TTL_SECONDS, GatewayFailure, SearchService, ServiceConfig
from agentflow.search_service.server import ServiceHTTPServer


def percentile(values, quantile):
    """Linearly interpolated percentile; an empty sample has no estimate."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _rss_bytes():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)
    except (ImportError, OSError, ValueError):
        return None


class SyntheticProviders:
    def __init__(self, *, payload_bytes=16384, delay=0.05, gate=None):
        # A repeated 'x' payload gzips to almost nothing and overstates throughput.
        source = random.Random(20261012).randbytes(math.ceil(payload_bytes * 3 / 4))
        self.payload = base64.b64encode(source).decode("ascii")[:payload_bytes]
        self.delay, self.gate = delay, gate
        self.lock = threading.Lock()
        self.executions = self.producers = self.active = self.peak_active = 0
        self.execution_active = self.peak_execution_active = 0

    @staticmethod
    def prepare(tool, params):
        if tool not in {"brave", "fetch", "wikipedia"} or not isinstance(params, dict):
            raise GatewayFailure("Invalid synthetic request", code="invalid_request", status_code=400)
        key = "url" if tool == "fetch" else "query"
        if not isinstance(params.get(key), str) or not params[key]:
            raise GatewayFailure("Missing synthetic key", code="invalid_request", status_code=400)
        return {key: params[key]}

    def execute(self, tool, params, service):
        with self.lock:
            self.executions += 1
            self.execution_active += 1
            self.peak_execution_active = max(self.peak_execution_active, self.execution_active)
        try:
            def producer():
                with self.lock:
                    self.producers += 1
                    self.active += 1
                    self.peak_active = max(self.peak_active, self.active)
                try:
                    if self.gate is not None and not self.gate.wait(timeout=10):
                        raise GatewayFailure("Probe gate deadline", code="probe_gate_timeout", status_code=504)
                    if self.delay:
                        time.sleep(self.delay)
                    return {"text": self.payload, "synthetic_key": next(iter(params.values()))}
                finally:
                    with self.lock:
                        self.active -= 1
            return service.cached("capacity_probe:" + tool, params, producer, CACHE_TTL_SECONDS)
        finally:
            with self.lock:
                self.execution_active -= 1


@contextmanager
def _service_harness(*, workers=16, queue_size=128, request_timeout=12, providers=None,
                     session_factory=None, **config_values):
    with tempfile.TemporaryDirectory(prefix="agentflow-capacity-local-") as directory:
        config = ServiceConfig(cache_dir=directory, db_path=str(Path(directory) / "cache.sqlite3"),
                               token="local-probe-" + uuid.uuid4().hex, workers=workers,
                               queue_size=queue_size, request_timeout=request_timeout,
                               max_retries=0, **config_values)
        kwargs = {"session_factory": session_factory} if session_factory is not None else {}
        service = SearchService(config, providers or SyntheticProviders(), **kwargs)
        server = thread = None
        try:
            server = ServiceHTTPServer(("127.0.0.1", 0), service)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
            thread.start()
            yield service, server
        finally:
            gate = getattr(service.providers, "gate", None)
            if gate is not None:
                gate.set()
            if server is not None:
                if thread is not None and thread.is_alive():
                    server.shutdown()
                server.server_close()
            if thread is not None:
                thread.join(timeout=2)
            service.close()


def _wait_until(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def probe_admission_boundary(*, workers=16, queue_size=128):
    """Compatibility result: the old W+Q admission boundary was removed."""
    return {"status": "not_applicable", "scope": "removed_shared_admission_boundary",
            "legacy_ignored": {"workers": workers, "queue_size": queue_size},
            "expected_capacity": None,
            "note": "No shared worker pool, admission semaphore or waiting queue remains."}


def probe_connection_boundary(*, workers=16, queue_size=128):
    """Compatibility result: there is no W+Q+16 HTTP admission cap."""
    return {"status": "not_applicable", "scope": "removed_http_connection_boundary",
            "legacy_ignored": {"workers": workers, "queue_size": queue_size},
            "expected_connection_cap": None,
            "note": "The old derived cap is removed; OS/listen/FD/thread resources still apply."}


def _http_call(port, token, path, payload, *, timeout):
    started = time.monotonic()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        headers = {"Authorization": "Bearer " + token, "Accept-Encoding": "gzip"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode()
        connection.request("POST" if body is not None else "GET", path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if response.getheader("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        data = json.loads(raw)
        return {"status": response.status, "error_code": data.get("error", {}).get("code"),
                "latency_seconds": time.monotonic() - started,
                "queue_wait_seconds": data.get("meta", {}).get("queue_wait_seconds"),
                "cache_hits": data.get("meta", {}).get("cache_hits", 0),
                "coalesced_waiters": data.get("meta", {}).get("coalesced_waiters", 0)}
    except Exception as exc:
        return {"status": 0, "error_code": "client_" + type(exc).__name__,
                "latency_seconds": time.monotonic() - started, "queue_wait_seconds": None}
    finally:
        connection.close()


def summarize_samples(samples, elapsed):
    latencies = [sample["latency_seconds"] for sample in samples]
    waits = [sample["queue_wait_seconds"] for sample in samples if sample.get("queue_wait_seconds") is not None]
    successes = sum(sample["status"] == 200 for sample in samples)
    return {"requests": len(samples), "successes": successes, "errors": len(samples) - successes,
            "error_codes": dict(Counter(sample.get("error_code") or "http_" + str(sample["status"])
                                        for sample in samples if sample["status"] != 200)),
            "elapsed_seconds": elapsed, "throughput_rps": successes / elapsed if elapsed > 0 else 0,
            "success_rate": successes / len(samples) if samples else 0,
            "p50_seconds": percentile(latencies, 0.5), "p95_seconds": percentile(latencies, 0.95),
            "max_seconds": max(latencies) if latencies else None, "queue_wait_p95_seconds": percentile(waits, 0.95)}


def probe_worker_stage(*, workers, burst_size, duration, latency_limit, payload_bytes, producer_delay, mode):
    if mode not in {"warm_cache", "cold_unique", "same_key_singleflight"}:
        raise ValueError("invalid worker probe mode")
    provider = SyntheticProviders(payload_bytes=payload_bytes, delay=producer_delay)
    # workers/queue_size remain accepted only for old callers and report labels.
    queue_size = 0
    timeout = max(2.0, latency_limit + 2.0)
    with _service_harness(workers=workers, queue_size=queue_size, request_timeout=timeout, providers=provider) as (service, server):
        warm_key = "warm-" + uuid.uuid4().hex
        if mode == "warm_cache":
            prewarm = _http_call(server.server_port, service.config.token, "/v1/search/brave", {"query": warm_key}, timeout=timeout)
            if prewarm["status"] != 200:
                failed = summarize_samples([prewarm], prewarm["latency_seconds"])
                failed.update(mode=mode, workers=workers, status="failed", reason="Prewarm request failed",
                              prewarm=prewarm, experimental_queue_size=None,
                              legacy_ignored={"workers": workers, "queue_size": queue_size},
                              offered_client_concurrency=burst_size, producer_count=provider.producers,
                              peak_producer_active=provider.peak_active,
                              peak_executing_workers=provider.peak_execution_active,
                              latency_limit_seconds=latency_limit)
                return failed
        producers_before = provider.producers
        with provider.lock:
            provider.peak_active = provider.peak_execution_active = 0
        metrics_before = service.metrics.snapshot()
        gate = threading.Event()
        shared_key = "shared-" + uuid.uuid4().hex
        stop_at = [0.0]

        def client(index):
            gate.wait()
            samples = []
            iteration = 0
            while iteration == 0 or (mode != "same_key_singleflight" and time.monotonic() < stop_at[0]):
                query = warm_key if mode == "warm_cache" else shared_key if mode == "same_key_singleflight" else f"cold-{index}-{iteration}"
                samples.append(_http_call(server.server_port, service.config.token, "/v1/search/brave",
                                          {"query": query}, timeout=timeout + 1))
                iteration += 1
            return samples

        futures, samples = [], []
        resource_error = None
        rss_before = _rss_bytes()
        with ThreadPoolExecutor(max_workers=burst_size, thread_name_prefix="probe-client") as pool:
            try:
                for index in range(burst_size):
                    futures.append(pool.submit(client, index))
            except RuntimeError as exc:
                resource_error = type(exc).__name__ + ": unable to start every requested client thread"
            started = time.monotonic()
            stop_at[0] = started + duration
            gate.set()
            for future in futures:
                try:
                    samples.extend(future.result())
                except Exception as exc:
                    resource_error = type(exc).__name__ + ": client probe could not complete"
            elapsed = time.monotonic() - started
        summary = summarize_samples(samples, elapsed)
        metrics = service.metrics.snapshot()
        delta = {key: value - metrics_before.get(key, 0) for key, value in metrics.items()
                 if value != metrics_before.get(key, 0)}
        producers = provider.producers - producers_before
        summary.update(mode=mode, workers=workers, experimental_queue_size=None,
                       legacy_ignored={"workers": workers, "queue_size": queue_size},
                       offered_client_concurrency=burst_size, producer_count=producers,
                       peak_producer_active=provider.peak_active,
                       peak_executing_workers=provider.peak_execution_active, metrics_delta=delta,
                       process_rss_before_bytes=rss_before, process_rss_after_bytes=_rss_bytes(),
                       rss_note="Linux VmRSS where available; otherwise process lifetime peak RSS")
        latency_ok = summary["p95_seconds"] is not None and summary["p95_seconds"] <= latency_limit
        correct_producer_count = (producers == 0 if mode == "warm_cache" else producers == 1 if mode == "same_key_singleflight"
                                  else producers == summary["successes"])
        summary["status"] = "stable" if summary["errors"] == 0 and latency_ok and correct_producer_count and not resource_error else "failed"
        if resource_error or any(code in summary["error_codes"] for code in ("client_OSError", "client_RuntimeError")):
            summary["status"] = "inconclusive"
        summary["reason"] = resource_error
        summary["latency_limit_seconds"] = latency_limit
        summary["producer_count_expected"] = 0 if mode == "warm_cache" else 1 if mode == "same_key_singleflight" else summary["successes"]
        return summary


class _LoopbackSession:
    """Factory wrapper which prevents a caller's proxy env from escaping localhost."""
    def __init__(self):
        import requests
        self.session = requests.Session()
        self.trust_env = False

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.session.close()

    def request(self, method, url, **kwargs):
        from urllib.parse import urlsplit
        if urlsplit(url).hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Offline probe refuses a non-loopback URL")
        self.session.trust_env = False
        return self.session.request(method, url, **kwargs)


class _RecordingUpstream(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, delay):
        self.delay = delay
        self.lock = threading.Lock()
        self.records = []
        self.active = Counter()
        self.peak = Counter()
        self.total_active = self.total_peak = 0
        super().__init__(("127.0.0.1", 0), _RecordingHandler)


class _RecordingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        group = self.path.split("?", 1)[0].strip("/") or "all"
        server = self.server
        with server.lock:
            server.records.append((group, time.monotonic()))
            server.active[group] += 1
            server.total_active += 1
            server.peak[group] = max(server.peak[group], server.active[group])
            server.total_peak = max(server.total_peak, server.total_active)
        try:
            time.sleep(server.delay)
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with server.lock:
                server.active[group] -= 1
                server.total_active -= 1

    def log_message(self, *unused):
        pass


def probe_budget_rules(*, rpm=6000.0, concurrency=2, delay=0.05):
    """Verify removed shared limiters using loopback HTTP, never production."""
    scenarios = []
    for kind, hints, hosts in (("wiki_unlimited", ["wiki", "wikipedia"], ["127.0.0.1"]),
                               ("brave_unlimited", ["brave", "yibu"], ["127.0.0.1"]),
                               ("web_unlimited_per_hostname", ["web"], ["127.0.0.1", "localhost"])):
        upstream = _RecordingUpstream(delay)
        thread = threading.Thread(target=upstream.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            with _service_harness(workers=4, queue_size=4, session_factory=_LoopbackSession,
                                  wiki_rpm=rpm, brave_rpm=rpm, web_rpm=rpm,
                                  wiki_concurrency=concurrency, brave_concurrency=concurrency,
                                  web_concurrency=concurrency) as (service, _):
                def fetch(index):
                    host = hosts[index % len(hosts)]
                    hint = hints[index % len(hints)]
                    group = host if kind == "web_unlimited_per_hostname" else "shared"
                    response = service.request(hint, "GET", f"http://{host}:{upstream.server_port}/{group}", timeout=2)
                    response.close()
                offered = max(8, concurrency * len(hosts) * 2)
                started = time.monotonic()
                with ThreadPoolExecutor(max_workers=offered) as pool:
                    list(pool.map(fetch, range(offered)))
                elapsed = time.monotonic() - started
                records = list(upstream.records)
                intervals = {}
                for group in set(record[0] for record in records):
                    starts = sorted(record[1] for record in records if record[0] == group)
                    intervals[group] = [b - a for a, b in zip(starts, starts[1:])]
                routing = {}
                routes = [("request_hint_" + hint + "_" + host, hint, "http://" + host)
                          for hint in hints for host in hosts]
                if kind == "wiki_unlimited":
                    routes += [("wiki_html_hostname", "web", "https://en.wikipedia.org/wiki/Test"),
                               ("other_wiki_language", "web", "https://zh.wikipedia.org/wiki/Test")]
                for label, hint, url in routes:
                    target_name, bucket = service._budget(hint, url)
                    routing[label] = {"bucket": target_name, "unlimited": bucket is None}
                bucket_names = sorted({value["bucket"] for value in routing.values() if not value["unlimited"]})
                verified = not bucket_names
                scenarios.append({"name": kind, "status": "verified" if verified else "failed",
                                  "service_limit_applies": False,
                                  "legacy_limits_ignored": {"rpm": rpm, "concurrency": concurrency},
                                  "configured_rpm": None, "configured_concurrency_per_bucket": None,
                                  "expected_start_interval_seconds": None,
                                  "min_start_intervals_seconds": {key: min(value) if value else None for key, value in intervals.items()},
                                  "peak_concurrency_by_group": dict(upstream.peak), "overall_peak_concurrency": upstream.total_peak,
                                  "offered_requests": offered, "elapsed_seconds": elapsed,
                                  "bucket_names": bucket_names, "routing": routing})
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(timeout=2)
    return {"scope": "removed_limiter_verification_not_production_quota", "scenarios": scenarios}


def probe_deadline_behavior():
    gate = threading.Event()
    providers = SyntheticProviders(payload_bytes=32, delay=0, gate=gate)
    with _service_harness(workers=1, queue_size=1, request_timeout=0.15, providers=providers) as (service, _):
        # Give callers a longer wait so this exercises the producer watchdog,
        # rather than releasing the gate just before its independent budget.
        deadline = time.monotonic() + 1.5
        futures = [service._submit("brave", {"query": str(index)}, deadline, str(index))
                   for index in range(2)]
        active = _wait_until(lambda: providers.active == 2, seconds=0.08)
        failures = []
        started = time.monotonic()
        try:
            for future in futures:
                try:
                    service._wait(future, deadline)
                except GatewayFailure as exc:
                    failures.append(exc.code)
            # Logical callers can already return their deadline error while
            # daemon cache producers remain blocked behind the synthetic gate.
            still_running = providers.active == 2
            executions_before_release = providers.executions
        finally:
            gate.set()
            for future in futures:
                try:
                    future.result(timeout=2)
                except GatewayFailure:
                    pass
            _wait_until(lambda: providers.active == 0 and not service._flights, seconds=1)
        verified = (active and failures == ["deadline_exceeded", "deadline_exceeded"]
                    and still_running and executions_before_release == 2)
        return {"status": "verified" if verified else "failed", "scope": "current_deadline_cancellation_behavior",
                "budget_seconds": 0.15, "elapsed_wait_seconds": time.monotonic() - started,
                "error_codes": failures, "queued_task_cancelled": None,
                "running_task_continued_after_caller_deadline": still_running,
                "provider_executions_before_release": executions_before_release,
                "cache_entries_after_expired_producer": service.cache.stats()["entries"],
                "note": "Unique misses execute independently; a running producer cannot be force-killed, but expired work is not cached."}


def run_local_suite(*, workers, burst_size=256, duration=3.0, latency_limit=10.0,
                    payload_bytes=16384, producer_delay=0.05, include_connections=True):
    if not workers or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in workers):
        raise ValueError("workers must contain positive integers")
    if isinstance(burst_size, bool) or not isinstance(burst_size, int) or burst_size <= 0:
        raise ValueError("burst_size must be a positive integer")
    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes < 0:
        raise ValueError("payload_bytes must be a nonnegative integer")
    if any(not math.isfinite(value) or value < 0 for value in (duration, producer_delay)):
        raise ValueError("duration/producer_delay must be finite and nonnegative")
    if not math.isfinite(latency_limit) or latency_limit <= 0:
        raise ValueError("latency_limit must be finite and positive")
    workers = sorted(set(workers))
    stages = []
    # Repeating ignored worker values would benchmark identical settings.
    for count in workers[:1]:
        for mode in ("warm_cache", "cold_unique", "same_key_singleflight"):
            try:
                stages.append(probe_worker_stage(workers=count, burst_size=burst_size, duration=duration,
                                                 latency_limit=latency_limit, payload_bytes=payload_bytes,
                                                 producer_delay=producer_delay, mode=mode))
            except (OSError, RuntimeError) as exc:
                stages.append({"mode": mode, "workers": count, "status": "inconclusive", "reason": type(exc).__name__})
    boundaries = {}
    boundary_probes = [("admission", probe_admission_boundary), ("deadline", probe_deadline_behavior)]
    if include_connections:
        boundary_probes.append(("connections", probe_connection_boundary))
    for name, boundary_probe in boundary_probes:
        try:
            boundaries[name] = boundary_probe()
        except (OSError, RuntimeError) as exc:
            boundaries[name] = {"status": "inconclusive", "scope": "configured_" + name + "_boundary",
                                "reason": "Local resource failure: " + type(exc).__name__}
    try:
        limiter = probe_budget_rules()
    except (OSError, RuntimeError, GatewayFailure) as exc:
        limiter = {"status": "inconclusive", "scope": "limiter_mechanism_only_not_production_quota", "reason": type(exc).__name__}
    return {"scope": "offline_synthetic_loopback_only", "actual_production_service_touched": False,
            "settings": {"legacy_ignored_worker_candidates": workers, "client_burst": burst_size, "duration_per_sustained_stage_seconds": duration,
                         "latency_limit_seconds": latency_limit, "payload_bytes": payload_bytes,
                         "producer_delay_seconds": producer_delay},
            "boundaries": boundaries, "stages": stages, "limiter_mechanism": limiter,
            "recommendations": {"largest_tested_stable_workers": None,
                                "tested_client_burst": burst_size,
                                "scope": "Only this synthetic workload; shared worker/queue limits no longer apply",
                                "queue_size_upper_limit": None,
                                "queue_note": "No shared queue or derived HTTP admission cap remains.",
                                "production_rpm_upper_limit": None,
                                "rpm_note": "No provider has a service limiter; local tests do not establish upstream quotas."}}
