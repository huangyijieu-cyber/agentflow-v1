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
import socket
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
    """Freeze actual workers to prove W+Q admission and W execution limits."""
    gate = threading.Event()
    provider = SyntheticProviders(payload_bytes=32, delay=0, gate=gate)
    capacity = workers + queue_size
    with _service_harness(workers=workers, queue_size=queue_size, providers=provider) as (service, _):
        futures = []
        rejection = None
        try:
            deadline = time.monotonic() + 10
            for index in range(capacity):
                futures.append(service._submit("brave", {"query": str(index)}, deadline, "boundary"))
            saturated = _wait_until(lambda: provider.active == workers)
            try:
                service._submit("brave", {"query": "overflow"}, deadline, "boundary-overflow")
            except GatewayFailure as exc:
                rejection = {"status": exc.status_code, "code": exc.code}
            snapshot = service.metrics.snapshot()
        finally:
            gate.set()
            for future in futures:
                try:
                    future.result(timeout=10)
                except Exception:
                    pass
        verified = saturated and len(futures) == capacity and rejection == {"status": 503, "code": "queue_full"}
        return {"status": "verified" if verified else "failed", "scope": "configured_admission_boundary",
                "workers": workers, "queue_size": queue_size, "accepted": len(futures),
                "expected_capacity": capacity, "next_request": rejection,
                "peak_executing_workers": provider.peak_execution_active, "metrics": snapshot}


def probe_connection_boundary(*, workers=16, queue_size=128):
    """Hold incomplete POST bodies without occupying the logical work queue."""
    cap = workers + queue_size + 16
    sockets = []
    established = 0
    observed_slots = 0
    probe_response = None
    reason = None
    with _service_harness(workers=workers, queue_size=queue_size, request_timeout=12) as (service, server):
        try:
            acquisition_deadline = time.monotonic() + 8
            for _ in range(cap):
                if time.monotonic() >= acquisition_deadline:
                    reason = "Could not hold every connection before the server body-read timeout"
                    break
                connection = socket.create_connection(("127.0.0.1", server.server_port), timeout=1)
                sockets.append(connection)
                connection.sendall(("POST /v1/search/brave HTTP/1.1\r\nHost: localhost\r\n"
                                    "Authorization: Bearer " + service.config.token + "\r\n"
                                    "Content-Type: application/json\r\nContent-Length: 1024\r\n\r\n{").encode())
                established += 1
            all_accepted = _wait_until(lambda: server._connections._value == 0, seconds=1)
            observed_slots = cap - server._connections._value
            if established == cap and all_accepted:
                probe_response = _http_call(server.server_port, service.config.token, "/healthz", None, timeout=2)
            elif reason is None:
                reason = "OS/FD/thread/listen capacity prevented proving the service connection boundary"
        except (OSError, RuntimeError) as exc:
            reason = "Local resource failure: " + type(exc).__name__
        finally:
            for connection in sockets:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
            _wait_until(lambda: server._connections._value == cap, seconds=2)
        metrics = service.metrics.snapshot()
        rejected = (probe_response is not None and probe_response["status"] == 503
                    and probe_response["error_code"] == "queue_full"
                    and metrics.get("connection_rejections", 0) == 1
                    and metrics.get("queue_rejections", 0) == 0)
        if not rejected and reason is None:
            reason = "The expected connection-only 503 rejection was not observed"
        return {"status": "verified" if rejected else "inconclusive", "scope": "configured_http_connection_boundary",
                "expected_connection_cap": cap, "opened_sockets": established,
                "observed_occupied_slots": observed_slots, "next_request": probe_response,
                "connection_rejections": metrics.get("connection_rejections", 0),
                "queue_rejections": metrics.get("queue_rejections", 0), "reason": reason,
                "note": "An OS resource failure is not evidence of the service's configured upper limit."}


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
    # A full burst can be admitted: overload/fixed queue boundary is tested separately.
    queue_size = max(0, burst_size - workers)
    timeout = max(2.0, latency_limit + 2.0)
    with _service_harness(workers=workers, queue_size=queue_size, request_timeout=timeout, providers=provider) as (service, server):
        warm_key = "warm-" + uuid.uuid4().hex
        if mode == "warm_cache":
            prewarm = _http_call(server.server_port, service.config.token, "/v1/search/brave", {"query": warm_key}, timeout=timeout)
            if prewarm["status"] != 200:
                failed = summarize_samples([prewarm], prewarm["latency_seconds"])
                failed.update(mode=mode, workers=workers, status="failed", reason="Prewarm request failed",
                              prewarm=prewarm, experimental_queue_size=queue_size,
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
        summary.update(mode=mode, workers=workers, experimental_queue_size=queue_size,
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
    """Measure limiter enforcement against actual loopback HTTP, never production."""
    scenarios = []
    for kind, hints, hosts in (("wiki_shared", ["wiki", "wikipedia"], ["127.0.0.1"]),
                               ("brave_shared", ["brave", "yibu"], ["127.0.0.1"]),
                               ("web_per_hostname", ["web"], ["127.0.0.1", "localhost"])):
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
                    group = host if kind == "web_per_hostname" else "shared"
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
                interval = 60 / rpm
                # Arrival timestamps can differ slightly from budget-acquisition timestamps.
                tolerance = max(0.004, interval * 0.25)
                respects_rpm = all(gap >= interval - tolerance for gaps in intervals.values() for gap in gaps)
                respects_concurrency = all(peak <= concurrency for peak in upstream.peak.values())
                routing = {}
                if kind == "wiki_shared":
                    name, first = service._budget("wiki", "http://127.0.0.1")
                    for label, hint, url in (("wiki_alias", "wikipedia", "http://127.0.0.1"),
                                             ("wiki_html_hostname", "web", "https://en.wikipedia.org/wiki/Test"),
                                             ("other_wiki_language", "web", "https://zh.wikipedia.org/wiki/Test")):
                        target_name, bucket = service._budget(hint, url)
                        routing[label] = {"bucket": target_name, "same_bucket": bucket is first}
                scenarios.append({"name": kind, "status": "verified" if respects_rpm and respects_concurrency else "failed",
                                  "configured_rpm": rpm, "configured_concurrency_per_bucket": concurrency,
                                  "expected_start_interval_seconds": interval, "arrival_tolerance_seconds": tolerance,
                                  "min_start_intervals_seconds": {key: min(value) if value else None for key, value in intervals.items()},
                                  "peak_concurrency_by_group": dict(upstream.peak), "overall_peak_concurrency": upstream.total_peak,
                                  "offered_requests": offered, "elapsed_seconds": elapsed,
                                  "bucket_names": sorted(service._budgets), "routing": routing})
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(timeout=2)
    return {"scope": "limiter_mechanism_only_not_production_quota", "scenarios": scenarios}


def probe_deadline_behavior():
    gate = threading.Event()
    providers = SyntheticProviders(payload_bytes=32, delay=0, gate=gate)
    with _service_harness(workers=1, queue_size=1, request_timeout=0.15, providers=providers) as (service, _):
        deadline = time.monotonic() + 0.15
        running = service._submit("brave", {"query": "running"}, deadline, "running")
        active = _wait_until(lambda: providers.active == 1, seconds=0.05)
        queued = service._submit("brave", {"query": "queued"}, deadline, "queued")
        failures = []
        started = time.monotonic()
        try:
            for future in (running, queued):
                try:
                    service._wait(future, deadline)
                except GatewayFailure as exc:
                    failures.append(exc.code)
            still_running = not running.done()
            queued_cancelled = queued.cancelled()
            executions_before_release = providers.executions
        finally:
            gate.set()
            try:
                running.result(timeout=2)
            except GatewayFailure:
                pass
        verified = active and failures == ["deadline_exceeded", "deadline_exceeded"] and still_running and queued_cancelled and executions_before_release == 1
        return {"status": "verified" if verified else "failed", "scope": "current_deadline_cancellation_behavior",
                "budget_seconds": 0.15, "elapsed_wait_seconds": time.monotonic() - started,
                "error_codes": failures, "queued_task_cancelled": queued_cancelled,
                "running_task_continued_after_caller_deadline": still_running,
                "provider_executions_before_release": executions_before_release,
                "cache_entries_after_expired_producer": service.cache.stats()["entries"],
                "note": "Waiting/queue time consumes the budget; cancelling a running Future does not stop its worker."}


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
    for count in workers:
        for mode in ("warm_cache", "cold_unique", "same_key_singleflight"):
            try:
                stages.append(probe_worker_stage(workers=count, burst_size=burst_size, duration=duration,
                                                 latency_limit=latency_limit, payload_bytes=payload_bytes,
                                                 producer_delay=producer_delay, mode=mode))
            except (OSError, RuntimeError) as exc:
                stages.append({"mode": mode, "workers": count, "status": "inconclusive", "reason": type(exc).__name__})
    stable = [count for count in workers if all(stage["status"] == "stable" for stage in stages if stage["workers"] == count)]
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
            "settings": {"worker_candidates": workers, "client_burst": burst_size, "duration_per_sustained_stage_seconds": duration,
                         "latency_limit_seconds": latency_limit, "payload_bytes": payload_bytes,
                         "producer_delay_seconds": producer_delay},
            "boundaries": boundaries, "stages": stages, "limiter_mechanism": limiter,
            "recommendations": {"largest_tested_stable_workers": max(stable) if stable else None,
                                "tested_client_burst": burst_size,
                                "scope": "Only this synthetic workload; not a production mixed-network worker upper limit",
                                "queue_size_upper_limit": None,
                                "queue_note": "Queue capacity is a configured admission boundary, not extra throughput or a discoverable physical maximum.",
                                "production_rpm_upper_limit": None,
                                "rpm_note": "Local tests verify the limiter, not Wikipedia/Yibu/provider quotas."}}
