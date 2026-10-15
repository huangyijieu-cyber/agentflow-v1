"""Bounded, uncached upstream probes using the production network transport.

This module never starts or modifies the shared search service.  Each probe
owns a temporary database and bypasses ``handle``/``cached`` entirely.  The
production transport supplies proxy routing, response bounds and deadlines;
the test dispatcher supplies the offered RPM and concurrent-call ceiling.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Callable
from urllib.parse import urlsplit


# The platform copies this repository rather than installing a test package.
_AGENTFLOW_PACKAGE = Path(__file__).resolve().parents[1] / "agentflow"
if str(_AGENTFLOW_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_AGENTFLOW_PACKAGE))

from agentflow.search_service.core import GatewayFailure, SearchService, ServiceConfig
from agentflow.search_service import providers


@dataclass(frozen=True)
class ProbeOutcome:
    ok: bool
    status: int | None = None
    error_code: str = ""
    retry_after: float | None = None


_WIKI_ARTICLES = (
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/Ada_Lovelace",
    "https://en.wikipedia.org/wiki/Alan_Turing",
    "https://en.wikipedia.org/wiki/Marie_Curie",
    "https://en.wikipedia.org/wiki/Isaac_Newton",
)
_DEFAULT_QUERIES = (
    "Albert Einstein", "Ada Lovelace", "Alan Turing", "Marie Curie", "Isaac Newton",
)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class _ProbeTransport(SearchService):
    def request(self, upstream, method, url, **kwargs):
        response = super().request(upstream, method, url, **kwargs)
        self._local.probe_http_status = response.status_code
        return response


class UpstreamProbe:
    """One raw logical HTTP fetch per ``call``, with no cache or retries.

    ``wiki_api`` probes the search API, not the entire multi-page Wiki tool.
    ``wiki_identity`` probes metadata; ``wiki_extract`` retrieves a complete
    article by title instead of the production pageid parameter. Neither makes
    a hidden warm-up call. Those two kinds require article titles as queries.
    ``wiki_html``/``web`` probe an HTML fetch and text extraction.  Redirects
    can make a logical call issue multiple HTTP requests; ``metrics`` exposes
    the actual attempt count.  Inputs and exception messages are not returned
    in reports, because they can contain private search text or credentials.
    """

    def __init__(self, kind: str, queries: list[str], urls: list[str],
                 timeout: float = 30, max_concurrency: int = 64):
        if kind not in {"wiki_api", "wiki_identity", "wiki_extract", "wiki_html", "brave", "web"}:
            raise ValueError("unsupported probe kind")
        _positive(timeout, "timeout")
        _positive_integer(max_concurrency, "max_concurrency")
        self.kind = kind
        self.queries = list(queries or _DEFAULT_QUERIES)
        if any(not isinstance(query, str) or not query.strip() or len(query) > 32768 for query in self.queries):
            raise ValueError("queries must be nonempty strings no longer than 32768 characters")
        supplied_urls = list(urls or (_WIKI_ARTICLES if kind == "wiki_html" else ()))
        self.urls = [providers.prepare("fetch", {"url": url})["url"] for url in supplied_urls]
        if kind == "web" and not self.urls:
            raise ValueError("web probes require a URL list from one hostname")
        hostnames = {(urlsplit(url).hostname or "").lower() for url in self.urls}
        if kind == "web" and len(hostnames) != 1:
            raise ValueError("one web probe must use a single hostname")
        if kind == "wiki_html" and any(host != "wikipedia.org" and not host.endswith(".wikipedia.org") for host in hostnames):
            raise ValueError("wiki_html URLs must be Wikipedia article URLs")
        self.max_concurrency = max_concurrency
        self._temporary: tempfile.TemporaryDirectory | None = None
        self.service: SearchService | None = None
        self._endpoint = None
        self._api_key = None
        if kind == "brave":
            self._endpoint = providers._url(os.getenv("BRAVE_YIBU_BASE_URL", providers.BRAVE_ENDPOINT))
            self._api_key = os.getenv("BRAVE_API_KEY") or os.getenv("YIBU_BRAVE_API_KEY")
            if not self._api_key:
                raise ValueError("BRAVE_API_KEY or YIBU_BRAVE_API_KEY must be configured")
        # Honor production timeout/size settings, but never its DB, secret
        # token, worker queue, manual RPM ceiling, or retry policy.
        self._config = replace(
            ServiceConfig.from_env(), token="", workers=1, queue_size=0,
            request_timeout=float(timeout), max_retries=0,
            wiki_rpm=1e9, brave_rpm=1e9, web_rpm=1e9,
            wiki_concurrency=max_concurrency,
            brave_concurrency=max_concurrency, web_concurrency=max_concurrency,
        )

    def __enter__(self):
        if self.service is not None:
            raise RuntimeError("probe is already open")
        self._temporary = tempfile.TemporaryDirectory(prefix="agentflow-capacity-upstream-")
        config = replace(self._config, cache_dir=self._temporary.name,
                         db_path=str(Path(self._temporary.name) / "unused.sqlite3"))
        try:
            self.service = _ProbeTransport(config)
        except BaseException:
            self._temporary.cleanup()
            self._temporary = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self.service is not None:
                self.service.close()
        finally:
            self.service = None
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
        return False

    def call(self, index: int) -> ProbeOutcome:
        if self.service is None:
            raise RuntimeError("use UpstreamProbe as a context manager")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("index must be a nonnegative integer")
        service = self.service
        service.metrics.add("probe_logical_calls")
        service._local.probe_http_status = None
        try:
            if self.kind == "wiki_api":
                data = providers._wiki_request(service, "https://en.wikipedia.org/w/api.php", {
                    "list": "search", "srsearch": self.queries[index % len(self.queries)],
                    "srlimit": 10, "srprop": "",
                })
                items = data.get("search")
                if not isinstance(items, list) or any(not isinstance(item, dict) or not isinstance(item.get("title"), str) for item in items):
                    raise GatewayFailure("Wikipedia response is missing search results", code="invalid_upstream_response")
            elif self.kind in {"wiki_identity", "wiki_extract"}:
                query = self.queries[index % len(self.queries)]
                params = {"titles": query}
                if self.kind == "wiki_identity":
                    params.update(prop="info|pageprops", inprop="url", ppprop="disambiguation", redirects="")
                else:
                    params.update(prop="extracts", explaintext="")
                data = providers._wiki_request(service, "https://en.wikipedia.org/w/api.php", params)
                page = providers._only_wiki_page(data)
                if self.kind == "wiki_extract":
                    if not isinstance(page.get("extract"), str):
                        raise GatewayFailure("Wikipedia page is missing text", code="invalid_upstream_response")
                elif type(page.get("pageid")) is not int or page["pageid"] <= 0 or not isinstance(page.get("fullurl"), str) or not isinstance(page.get("title"), str):
                    raise GatewayFailure("Wikipedia page metadata is incomplete", code="invalid_upstream_response")
            elif self.kind == "brave":
                response = service.request(
                    "brave", "GET", self._endpoint,
                    params={"q": self.queries[index % len(self.queries)], "count": 10},
                    headers={"Accept": "application/json", "Authorization": f"Bearer {self._api_key}"},
                    timeout=20, verify=providers._tls_verify(),
                )
                providers._brave_results(providers._json(response, "Yibu"))
            else:
                url = self.urls[index % len(self.urls)]
                hostname = (urlsplit(url).hostname or "").lower()
                is_wiki = hostname == "wikipedia.org" or hostname.endswith(".wikipedia.org")
                headers = dict(providers.WEB_HEADERS)
                if is_wiki:
                    headers["User-Agent"] = os.getenv("WIKIMEDIA_USER_AGENT", providers.WIKIMEDIA_USER_AGENT)
                response = service.request(
                    "wikipedia" if is_wiki else f"web:{hostname}", "GET", url,
                    headers=headers, timeout=10, verify=providers._tls_verify(),
                )
                try:
                    providers.BeautifulSoup(response.content, "html.parser").get_text(separator="\n", strip=True)[:1000000]
                except Exception:
                    raise GatewayFailure("Could not extract webpage text", code="invalid_upstream_response") from None
                finally:
                    response.close()
            service.metrics.add("probe_succeeded")
            return ProbeOutcome(True, status=service._local.probe_http_status)
        except GatewayFailure as exc:
            service.metrics.add("probe_failed")
            status = exc.upstream_status or service._local.probe_http_status
            if status is None and (exc.status_code == 429 or exc.code == "rate_limited"):
                # MediaWiki can throttle inside an HTTP 200 JSON body. Preserve
                # the actual transport status and carry throttling as a code.
                status = 200 if exc.code == "rate_limited" else 429
            return ProbeOutcome(False, status=status, error_code=exc.code, retry_after=exc.retry_after)
        except Exception:
            service.metrics.add("probe_failed")
            return ProbeOutcome(False, status=service._local.probe_http_status, error_code="probe_exception")

    def metrics(self) -> dict:
        if self.service is None:
            raise RuntimeError("probe is not open")
        return self.service.metrics.snapshot()


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def probe_stage(call: Callable[[int], ProbeOutcome], *, duration: float,
                concurrency: int, rpm: float, max_requests: int,
                latency_limit: float, stop_on_429: bool = True,
                stop_event: threading.Event | None = None) -> dict:
    """Offer paced calls without a producer queue, then drain started calls.

    A full concurrent-call window drops the current offered slot rather than
    adding work to a queue. Missed scheduler slots are also dropped, avoiding
    catch-up bursts. A 429 (including an API error in HTTP 200) stops new calls
    immediately; calls already submitted are drained. There are no retries.
    A caller-supplied ``stop_event`` stops new submissions and drains existing
    calls, allowing a multi-probe run to share cancellation without leaving
    other dispatcher threads sending after the outer CLI is interrupted.

    ``achieved_rpm`` includes the full dispatch window, including initial
    startup delay, skipped trailing slots, and any post-dispatch call starts.
    ``sending_window_seconds`` separately records first-to-last actual start
    plus one pacing interval; ``goodput_rpm`` includes final draining time.
    ``stable`` deliberately does not claim a maximum: callers must also require
    enough samples, multiple stages and an appropriate workload.
    """
    for value, name in ((duration, "duration"), (rpm, "rpm"), (latency_limit, "latency_limit")):
        _positive(value, name)
    _positive_integer(concurrency, "concurrency")
    _positive_integer(max_requests, "max_requests")
    if not callable(call):
        raise ValueError("call must be callable")
    interval = 60.0 / float(rpm)
    began = time.monotonic()
    ends = began + float(duration)
    start_times = []
    starts_lock = threading.Lock()
    throttled = threading.Event()
    completed = []
    active = {}
    active_calls = 0
    peak_calls = 0
    submitted = 0
    slots = 0
    capacity_skips = 0
    scheduler_skips = 0
    peak_outstanding = 0
    stopped = began
    stop_reason = "duration"

    def invoke(index):
        nonlocal active_calls, peak_calls
        began_call = time.monotonic()
        with starts_lock:
            start_times.append(began_call)
            active_calls += 1
            peak_calls = max(peak_calls, active_calls)
        try:
            try:
                outcome = call(index)
                if not isinstance(outcome, ProbeOutcome):
                    outcome = ProbeOutcome(False, error_code="invalid_probe_outcome")
            except Exception:
                outcome = ProbeOutcome(False, error_code="probe_exception")
        finally:
            with starts_lock:
                active_calls -= 1
        if outcome.status == 429 or outcome.error_code == "rate_limited":
            throttled.set()
        return outcome, time.monotonic() - began_call

    pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="capacity-probe")
    try:
        while True:
            for future in list(active):
                if future.done():
                    completed.append(future.result())
                    del active[future]
            now = time.monotonic()
            if stop_event is not None and stop_event.is_set():
                stop_reason = "cancelled"
                stopped = now
                break
            if stop_on_429 and throttled.is_set():
                stop_reason = "rate_limited"
                stopped = now
                break
            if submitted >= max_requests:
                stop_reason = "max_requests"
                stopped = now
                break
            if now >= ends:
                stopped = now
                break
            next_slot = began + slots * interval
            if now < next_slot:
                pause = min(next_slot - now, ends - now, 0.05)
                if active:
                    wait(active, timeout=pause, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(pause)
                continue
            newest_slot = int((now - began) / interval)
            if newest_slot > slots:
                scheduler_skips += newest_slot - slots
                slots = newest_slot
            slots += 1
            if len(active) >= concurrency:
                capacity_skips += 1
                continue
            future = pool.submit(invoke, submitted)
            active[future] = submitted
            submitted += 1
            peak_outstanding = max(peak_outstanding, len(active))
    except KeyboardInterrupt:
        stopped = time.monotonic()
        stop_reason = "interrupted"
    finally:
        # There is no unbounded pending work: at most concurrency calls drain.
        for future in active:
            completed.append(future.result())
        pool.shutdown(wait=True, cancel_futures=True)
    finished = time.monotonic()
    outcomes = [item[0] for item in completed]
    latencies = [item[1] for item in completed]
    sent = len(start_times)
    succeeded = sum(outcome.ok for outcome in outcomes)
    failed = len(outcomes) - succeeded
    elapsed = max(finished - began, 1e-9)
    sending_window = max(start_times) - min(start_times) + interval if start_times else 0.0
    # Include initial startup delay as well as empty trailing capacity slots.
    # Excluding the initial interval could make a late-starting dispatcher
    # falsely claim it achieved the requested RPM. Pending workers can also
    # start during drain, so include their latest actual start in this window.
    rate_window = max(stopped - began,
                      max(start_times) - began + interval if start_times else 0.0)
    achieved = sent * 60.0 / rate_window if rate_window else 0.0
    p95 = _percentile(latencies, .95)
    limited = any(outcome.status == 429 or outcome.error_code == "rate_limited" for outcome in outcomes)
    target_reached = sent > 0 and achieved >= .9 * rpm
    success_rate = succeeded / sent if sent else 0.0
    statuses = Counter(str(outcome.status) for outcome in outcomes if outcome.status is not None)
    errors = Counter(outcome.error_code or "unspecified_error" for outcome in outcomes if not outcome.ok)
    retry_after_values = [outcome.retry_after for outcome in outcomes if outcome.retry_after is not None]
    return {
        "sent": sent, "succeeded": succeeded, "failed": failed,
        "success_rate": success_rate, "status_counts": dict(statuses),
        "error_codes": dict(errors), "elapsed_seconds": elapsed,
        "dispatch_seconds": max(0.0, stopped - began),
        "sending_window_seconds": sending_window,
        "rate_window_seconds": rate_window,
        "offered_rpm": float(rpm), "achieved_rpm": achieved,
        "goodput_rpm": succeeded * 60.0 / elapsed,
        "peak_inflight": peak_calls, "peak_outstanding": peak_outstanding,
        "target_concurrency": concurrency,
        "concurrency_reached": peak_calls >= concurrency,
        "capacity_skips": capacity_skips, "scheduler_skips": scheduler_skips,
        "missed_slots": capacity_skips + scheduler_skips, "offered_slots": slots,
        "latency_p50": _percentile(latencies, .50), "latency_p95": p95,
        "latency_p99": _percentile(latencies, .99),
        "latency_max": max(latencies, default=0.0),
        "latency_limit": float(latency_limit), "rate_limited": limited,
        "retry_after_seconds": max(retry_after_values, default=None),
        "aborted_reason": stop_reason if stop_reason in {"rate_limited", "interrupted", "cancelled"} else "",
        "stop_reason": stop_reason, "drain_seconds": max(0.0, finished - stopped),
        "target_reached": target_reached,
        "stable": bool(sent and stop_reason not in {"interrupted", "cancelled"}
                       and not limited and success_rate >= .98 and p95 <= latency_limit and target_reached),
        "sample_count": sent,
    }
