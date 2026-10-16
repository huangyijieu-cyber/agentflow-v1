"""Shared-search deadline causes survive the client and tool error boundary."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.search_service.core import GatewayFailure
from agentflow.tools.execution_context import safe_failure, sanitize_search_diagnostics
from agentflow.tools.search_gateway import SearchGatewayClient, SearchGatewayError


class Response:
    status_code = 504
    headers = {"X-Request-ID": "deadline-test"}

    def __init__(self, error):
        self.error = error
        self.closed = False

    def json(self):
        return {"error": self.error}

    def close(self):
        self.closed = True


class Session:
    def __init__(self, response):
        self.response = response
        self.trust_env = True

    def request(self, *args, **kwargs):
        return self.response


class DeadlineClientDiagnosticsTests(unittest.TestCase):
    def diagnostics(self):
        return {
            "phase": "retry_wait", "deadline_reason": "retry_wait", "deadline_scope": "producer",
            "deadline_source": "service", "elapsed_seconds": 180.01, "budget_seconds": 180,
            "phase_elapsed_seconds": 40.5, "phase_durations": {"upstream_headers": 139.51, "retry_wait": 40.5},
            "shared_phase": "upstream_body", "shared_elapsed_seconds": 120, "shared_budget_seconds": 180,
            "shared_phase_durations": {"upstream_headers": 20, "upstream_body": 100},
            "last_upstream_error": {"code": "upstream_http_error", "upstream": "wikipedia",
                                    "upstream_status": 429, "attempts": 2, "retry_after": 60},
            "upstream_recovered": False, "cache_namespace": "wikipedia_page_text", "tool": "wikipedia",
        }

    def test_http_deadline_keeps_429_cause_through_safe_failure_and_copy(self):
        diagnostics = self.diagnostics()
        # Use the real service serializer: diagnostics are flat in error, not nested.
        response = Response(GatewayFailure(
            "Search request deadline exceeded", code="deadline_exceeded", status_code=504,
            retryable=True, diagnostics=diagnostics,
        ).as_dict())
        client = SearchGatewayClient("http://cache.test", "test-token", session=Session(response))
        with self.assertRaises(SearchGatewayError) as caught:
            client.wikipedia_search("a query that must never enter diagnostics")
        error = caught.exception
        self.assertTrue(response.closed)
        self.assertEqual(error.diagnostics, diagnostics)
        self.assertEqual((error.phase, error.deadline_reason, error.deadline_scope),
                         ("retry_wait", "retry_wait", "producer"))
        for expected in ("cause=retry_wait", "scope=producer", "elapsed=180s", "budget=180s", "HTTP 429"):
            self.assertIn(expected, str(error))
        failure = safe_failure(error, "search.wikipedia")
        self.assertEqual(failure.phase, "search.wikipedia")
        self.assertEqual(failure.search_phase, "retry_wait")
        self.assertEqual(failure.deadline_reason, "retry_wait")
        self.assertEqual(failure.deadline_scope, "producer")
        self.assertEqual(failure.diagnostics, diagnostics)
        for expected in ("cause=retry_wait", "scope=producer", "HTTP 429", "upstream_recovered=false"):
            self.assertIn(expected, str(failure))
        duplicate = failure.copy()
        self.assertEqual(str(duplicate), str(failure))
        self.assertEqual(duplicate.diagnostics, failure.diagnostics)
        duplicate.diagnostics["last_upstream_error"]["attempts"] = 999
        self.assertEqual(failure.diagnostics["last_upstream_error"]["attempts"], 2)
        self.assertIs(safe_failure(failure, "other.phase"), failure)

    def test_nested_diagnostics_remain_compatible_and_override_flat_fields(self):
        diagnostics = self.diagnostics()
        response = Response({"code": "deadline_exceeded", "message": "Search request deadline exceeded",
                             "phase": "shared_wait", "diagnostics": diagnostics})
        client = SearchGatewayClient("http://cache.test", "test-token", session=Session(response))
        with self.assertRaises(SearchGatewayError) as caught:
            client.wikipedia_search("private-query")
        self.assertEqual(caught.exception.diagnostics, diagnostics)
        self.assertEqual(caught.exception.phase, "retry_wait")

    def test_flat_error_diagnostics_are_allowlisted_before_logging(self):
        error = GatewayFailure("Search request deadline exceeded", code="deadline_exceeded",
                               diagnostics=self.diagnostics()).as_dict()
        error.update({"query": "private-query", "token": "bearer-secret",
                      "url": "https://user:password@host/"})
        error["last_upstream_error"]["message"] = "private-query bearer-secret"
        response = Response(error)
        client = SearchGatewayClient("http://cache.test", "test-token", session=Session(response))
        with self.assertRaises(SearchGatewayError) as caught:
            client.wikipedia_search("private-query")
        failure = safe_failure(caught.exception, "search.wikipedia")
        self.assertEqual(failure.last_upstream_error["upstream_status"], 429)
        rendered = str(caught.exception) + str(failure) + repr(failure.diagnostics)
        for forbidden in ("private-query", "bearer-secret", "password", "https://"):
            self.assertNotIn(forbidden, rendered)

    def test_untrusted_nested_fields_and_nonfinite_timings_are_not_logged(self):
        diagnostics = self.diagnostics()
        diagnostics.update({"query": "private-query", "token": "bearer-secret", "url": "https://user:password@host/",
                            "caller_phase": "https://user:password@host/", "elapsed_seconds": float("nan"),
                            "phase_elapsed_seconds": True})
        diagnostics["phase_durations"].update({"https://secret.test/": 30, "cache_write": "bearer-secret"})
        diagnostics["last_upstream_error"].update({"message": "private-query bearer-secret", "token": "bearer-secret",
                                                   "url": "https://user:password@host/", "retry_after": float("inf")})
        original = copy.deepcopy(diagnostics)
        error = SearchGatewayError("Search request deadline exceeded", code="deadline_exceeded", diagnostics=diagnostics)
        failure = safe_failure(error, "search.wikipedia")
        rendered = str(error) + str(failure) + repr(failure.diagnostics)
        for forbidden in ("private-query", "bearer-secret", "password", "https://", "nan", "inf"):
            self.assertNotIn(forbidden, rendered)
        self.assertEqual(diagnostics["last_upstream_error"], original["last_upstream_error"])
        self.assertNotIn("elapsed_seconds", failure.diagnostics)
        self.assertNotIn("phase_elapsed_seconds", failure.diagnostics)
        self.assertEqual(failure.last_upstream_error["upstream_status"], 429)

    def test_malformed_diagnostic_shapes_do_not_break_error_handling(self):
        for diagnostics in (None, "secret", [], {"phase": ["secret"], "elapsed_seconds": 10 ** 1000,
                                               "last_upstream_error": {"attempts": True, "upstream_status": "429"}}):
            with self.subTest(diagnostics_type=type(diagnostics).__name__):
                self.assertEqual(sanitize_search_diagnostics(diagnostics), {})
                error = SearchGatewayError("Search failed", diagnostics=diagnostics)
                self.assertEqual(safe_failure(error, "search.wikipedia").diagnostics, {})

    def test_errors_without_diagnostics_keep_existing_behavior(self):
        error = SearchGatewayError("page unavailable", code="upstream_http_error", status_code=502,
                                   request_id="test-req", upstream="wikipedia", upstream_status_code=404, attempts=1)
        self.assertEqual(str(error), "page unavailable (upstream_http_error, HTTP 502, request_id=test-req, upstream=wikipedia, attempts=1)")
        self.assertEqual(error.diagnostics, {})
        self.assertIsNone(error.phase)
        self.assertFalse(error.is_shared_failure)
        failure = safe_failure(error, "search.wikipedia")
        self.assertEqual(str(failure), "Shared search request failed (upstream_http_error, phase=search.wikipedia, type=SearchGatewayError, HTTP=404)")
        self.assertEqual(str(failure.copy()), str(failure))

    def test_expired_shared_producer_stops_more_page_requests(self):
        for code in ("shared_execution_expired", "producer_timeout"):
            with self.subTest(code=code):
                error = SearchGatewayError("Shared search unavailable", code=code)
                self.assertTrue(error.is_shared_failure)


if __name__ == "__main__":
    unittest.main()
