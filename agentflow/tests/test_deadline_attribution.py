"""Offline regressions for deadline attribution and nonblocking cache fill.

These tests use events and mocked HTTP sessions, never Wikimedia or real search
credentials. Short caller budgets are deliberately independent of producer
budgets so an upstream error cannot silently become an unexplained deadline.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.search_service.core import GatewayFailure, SearchService, ServiceConfig


class ProbeProviders:
    def __init__(self, produce):
        self.produce = produce
        self.calls = Counter()
        self.lock = threading.Lock()

    def prepare(self, tool, params):
        return dict(params)

    def execute(self, tool, params, service):
        def produce():
            with self.lock:
                self.calls[params["query"]] += 1
            return self.produce(tool, params, service)

        return service.cached("deadline-probe:" + tool, params, produce, 3600)


def response(status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result.headers.update(headers or {})
    result._content = b"{}"
    result._content_consumed = True
    return result


class DeadlineAttributionTests(unittest.TestCase):
    def make_service(self, produce, *, timeout=2, session_factory=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        providers = ProbeProviders(produce)
        config = ServiceConfig(cache_dir=directory.name,
                               db_path=str(Path(directory.name) / "cache.sqlite3"),
                               request_timeout=timeout, max_retries=1,
                               token="private-service-token")
        kwargs = {} if session_factory is None else {"session_factory": session_factory}
        service = SearchService(config, providers, **kwargs)
        self.addCleanup(service.close)
        return service, providers

    def wait_until(self, predicate, *, timeout=1, message="Condition did not become true"):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(0.005)
        self.fail(message)

    def test_producer_timeout_exception_is_not_mistaken_for_wait_timeout(self):
        def produce(*_):
            raise FutureTimeoutError("internal producer timeout")

        service, providers = self.make_service(produce)
        started = time.monotonic()
        with self.assertRaises(GatewayFailure) as caught:
            service.handle("wikipedia", {"query": "one"})
        self.assertEqual(caught.exception.code, "producer_timeout")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(providers.calls["one"], 1)
        self.assertEqual(service.metrics.snapshot().get("requests_failed/deadline_exceeded", 0), 0)

    def test_slow_cache_write_neither_delays_results_nor_other_producers(self):
        writing, release_write, other_started = (threading.Event() for _ in range(3))

        def produce(tool, params, service):
            if params["query"] == "other":
                other_started.set()
            return {"results": [params["query"]]}

        service, providers = self.make_service(produce)

        def slow_put(*args, **kwargs):
            writing.set()
            if not release_write.wait(2):
                raise AssertionError("Test cache writer was not released")
            return True

        # A fixed cache miss isolates request coalescing from SQLite reader
        # scheduling. A completed result must stay shareable until persistence
        # has finished, so a second request cannot restart the same producer.
        with patch.object(service.cache, "get", return_value=None), \
                patch.object(service.cache, "put", side_effect=slow_put), \
                ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(service.handle, "wikipedia", {"query": "same"})
            try:
                self.assertTrue(writing.wait(1))
                self.assertEqual(first.result(timeout=0.5)["results"], ["same"])
                duplicate = pool.submit(service.handle, "wikipedia", {"query": "same"})
                self.assertEqual(duplicate.result(timeout=0.5)["results"], ["same"])
                self.assertEqual(providers.calls["same"], 1)
                other = pool.submit(service.handle, "wikipedia", {"query": "other"})
                self.assertTrue(other_started.wait(0.5), "Cache persistence held the shared state lock")
                self.assertEqual(other.result(timeout=0.5)["results"], ["other"])
            finally:
                release_write.set()
            first.result(timeout=1)
            self.wait_until(lambda: not service._flights)
        self.assertEqual(providers.calls, {"same": 1, "other": 1})

    def test_short_caller_deadline_keeps_observed_429_and_retry_phase(self):
        factory = MagicMock()
        session = factory.return_value.__enter__.return_value
        session.request.side_effect = [response(429, {"Retry-After": "0"}), response()]

        def produce(tool, params, service):
            service.request("wikipedia", "GET", "https://en.wikipedia.org/w/api.php")
            return {"results": []}

        service, _ = self.make_service(produce, session_factory=factory)
        with self.assertRaises(GatewayFailure) as caught:
            service.handle("wikipedia", {"query": "retry"}, deadline=time.monotonic() + 0.12)
        failure = caught.exception.as_dict()
        self.assertEqual(failure["code"], "deadline_exceeded")
        self.assertEqual(failure["deadline_reason"], "retry_wait")
        self.assertEqual(failure["last_upstream_error"]["upstream_status"], 429)
        metrics = service.metrics.snapshot()
        self.assertEqual(metrics["upstream_429/wikipedia"], 1)
        self.assertEqual(metrics["deadline_reasons/retry_wait"], 1)
        self.assertEqual(metrics["deadline_tools/wikipedia"], 1)
        self.assertEqual(metrics["deadline_scopes/" + failure["deadline_scope"]], 1)
        self.assertEqual(metrics["requests_failed/deadline_exceeded"], 1)
        # Let the independent producer finish while the mocked session is alive.
        self.wait_until(lambda: not service._flights)

    def test_expired_shared_execution_is_distinct_from_new_request_deadline(self):
        entered, release = threading.Event(), threading.Event()

        def produce(*_):
            entered.set()
            if not release.wait(2):
                raise AssertionError("Test producer was not released")
            return {"results": []}

        service, providers = self.make_service(produce, timeout=0.10)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(service.handle, "wikipedia", {"query": "stuck"},
                                deadline=time.monotonic() + 1)
            try:
                self.assertTrue(entered.wait(1))
                with self.assertRaises(GatewayFailure) as first_failure:
                    first.result(timeout=0.7)
                diagnostics = first_failure.exception.as_dict()
                self.assertEqual(diagnostics["code"], "deadline_exceeded")
                self.assertIn(diagnostics["deadline_reason"], {"producer_start", "producer_execution", "upstream_headers"})
                self.assertEqual(len(service._flights), 1)
                began = time.monotonic()
                with self.assertRaises(GatewayFailure) as inherited:
                    service.handle("wikipedia", {"query": "stuck"}, deadline=began + 1)
                data = inherited.exception.as_dict()
                self.assertEqual(data["code"], "shared_execution_expired")
                self.assertEqual(data["deadline_scope"], "inherited")
                self.assertEqual(data["deadline_reason"], "previous_producer_deadline")
                self.assertLess(time.monotonic() - began, 0.3)
                self.assertEqual(providers.calls["stuck"], 1)
                metrics = service.metrics.snapshot()
                self.assertEqual(metrics["deadline_reasons/previous_producer_deadline"], 1)
                self.assertEqual(metrics["deadline_scopes/inherited"], 1)
                self.assertEqual(metrics["deadline_tools/wikipedia"], 2)
                self.wait_until(lambda: service.metrics.snapshot().get("producer_deadline_reasons/producer_start") == 1)
                metrics = service.metrics.snapshot()
                self.assertEqual(metrics["producer_deadlines_exceeded"], 1)
                self.assertEqual(metrics["producer_deadline_phases/producer_start"], 1)
            finally:
                release.set()
            self.wait_until(lambda: not service._flights)

    def test_batch_outer_wait_retains_retry_cause_and_counts_failure_once(self):
        factory = MagicMock()
        session = factory.return_value.__enter__.return_value
        session.request.side_effect = [response(429, {"Retry-After": "0"}), response()]
        retry_entered = threading.Event()

        def produce(tool, params, service):
            service.request("wikipedia", "GET", "https://en.wikipedia.org/w/api.php")
            return {"results": []}

        service, _ = self.make_service(produce, timeout=1, session_factory=factory)
        original_sleep = service._sleep

        def retry_wait(delay):
            retry_entered.set()
            return original_sleep(delay)

        # Deliberately give the batch collector an earlier expiry than its
        # worker. This deterministically covers _wait's own timeout path,
        # instead of relying on which caller thread wins an identical deadline.
        with patch.object(service, "_sleep", side_effect=retry_wait):
            future = service._submit("wikipedia", {"query": "batch-retry"},
                                     time.monotonic() + 0.25, "batch-deadline-request")
            self.assertTrue(retry_entered.wait(1))
            with self.assertRaises(GatewayFailure) as caught:
                service._wait(future, time.monotonic() + 0.04)
            data = caught.exception.as_dict()
            self.assertEqual(data["code"], "deadline_exceeded")
            self.assertEqual(data["deadline_reason"], "retry_wait")
            self.assertEqual(data["last_upstream_error"]["upstream_status"], 429)
            self.assertFalse(future.done(), "The test must exercise the outer collector timeout")
            # The worker subsequently reaches its own deadline. Both paths
            # refer to one logical request and must not double-count it.
            with self.assertRaises(GatewayFailure):
                future.result(timeout=0.7)
            self.wait_until(lambda: not service._flights)
        metrics = service.metrics.snapshot()
        self.assertEqual(metrics["requests_failed"], 1)
        self.assertEqual(metrics["requests_failed/deadline_exceeded"], 1)
        self.assertEqual(metrics["deadline_reasons/retry_wait"], 1)
        self.assertEqual(metrics["deadline_tools/wikipedia"], 1)
        self.assertEqual(metrics["upstream_429/wikipedia"], 1)

    def test_late_429_headers_are_counted_after_producer_deadline(self):
        entered, release_headers = threading.Event(), threading.Event()
        factory = MagicMock()
        session = factory.return_value.__enter__.return_value

        def request(*args, **kwargs):
            entered.set()
            if not release_headers.wait(2):
                raise AssertionError("Test HTTP headers were not released")
            return response(429, {"Retry-After": "1"})

        session.request.side_effect = request

        def produce(tool, params, service):
            service.request("wikipedia", "GET", "https://en.wikipedia.org/w/api.php")
            return {"results": []}

        service, _ = self.make_service(produce, timeout=0.10, session_factory=factory)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(service.handle, "wikipedia", {"query": "late"},
                                 deadline=time.monotonic() + 1)
            try:
                self.assertTrue(entered.wait(1))
                with self.assertRaises(GatewayFailure) as caught:
                    result.result(timeout=0.7)
                self.assertEqual(caught.exception.code, "deadline_exceeded")
                self.assertEqual(caught.exception.as_dict()["deadline_reason"], "upstream_headers")
            finally:
                release_headers.set()
            self.wait_until(lambda: service.metrics.snapshot().get("upstream_429/wikipedia") == 1)
            self.wait_until(lambda: not service._flights)
        self.assertEqual(session.request.call_count, 1)

    def test_deadline_failure_log_has_attribution_without_query_or_token(self):
        release = threading.Event()

        def produce(*_):
            release.wait(2)
            return {"results": []}

        service, _ = self.make_service(produce)
        try:
            with self.assertLogs(level="WARNING") as logs:
                for request_id in ("deadline-request-A", "deadline-request-B"):
                    with self.assertRaises(GatewayFailure):
                        service.handle("wikipedia", {"query": "private-query-do-not-log"},
                                       deadline=time.monotonic() + 0.05, request_id=request_id)
            messages = [record.getMessage() for record in logs.records]
            for request_id in ("deadline-request-A", "deadline-request-B"):
                matching = [message for message in messages if request_id in message]
                self.assertTrue(matching, f"No attribution log for {request_id}")
                text = " ".join(matching)
                for field in ("request_id", "phase", "deadline_reason", "budget_seconds", "elapsed_seconds"):
                    self.assertIn(field, text)
                self.assertNotIn("private-query-do-not-log", text)
                self.assertNotIn("private-service-token", text)
        finally:
            release.set()
            self.wait_until(lambda: not service._flights)


if __name__ == "__main__":
    unittest.main()
