"""Offline capacity-probe checks; never sends requests to a real upstream."""
import importlib.util
from concurrent.futures import Future
import json
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[2] / "train-roma" / "search_cache_capacity_upstream.py"
SPEC = importlib.util.spec_from_file_location("search_cache_capacity_upstream", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ProbeOutcome = MODULE.ProbeOutcome
UpstreamProbe = MODULE.UpstreamProbe
probe_stage = MODULE.probe_stage


class Response:
    def __init__(self, data=None, *, content=None, status=200, headers=None):
        self.status_code = status
        self.headers = dict(headers or {})
        self._content = content if content is not None else json.dumps(data).encode()
        self._content_consumed = True
        self.closed = False
        self.close_count = 0

    @property
    def content(self):
        return self._content

    def json(self):
        return json.loads(self._content)

    def iter_content(self, chunk_size):
        yield self._content

    def close(self):
        self.closed = True
        self.close_count += 1


class SessionFactory:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.sessions = []

    def __call__(self):
        factory = self

        class Session:
            trust_env = False
            exited = False

            def __enter__(self):
                factory.sessions.append(self)
                return self

            def __exit__(self, *args):
                self.exited = True

            def request(self, method, url, **kwargs):
                factory.requests.append((method, url, kwargs))
                return next(factory.responses)

        return Session()


class UpstreamProbeTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def with_response(self, kind, response, queries=None, urls=None):
        probe = UpstreamProbe(kind, queries or ["Ada Lovelace"], urls or [])
        probe.__enter__()
        self.addCleanup(probe.__exit__, None, None, None)
        factory = SessionFactory([response])
        probe.service._session_factory = factory
        return probe, factory

    def test_wiki_search_uses_production_params_transport_and_no_cache(self):
        response = Response({"query": {"search": [{"title": "Ada Lovelace"}]}})
        probe, factory = self.with_response("wiki_api", response)
        self.assertEqual(probe.call(0), ProbeOutcome(True, status=200))
        method, url, kwargs = factory.requests[0]
        self.assertEqual(method, "GET")
        self.assertEqual(kwargs["params"], {
            "action": "query", "format": "json", "list": "search", "srsearch": "Ada Lovelace",
            "srlimit": 10, "srprop": "",
        })
        self.assertEqual(kwargs["timeout"], (20.0, 20.0))
        self.assertEqual(kwargs["headers"]["User-Agent"], MODULE.providers.WIKIMEDIA_USER_AGENT)
        self.assertTrue(kwargs["stream"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertFalse(kwargs["verify"])
        self.assertTrue(factory.sessions[0].trust_env)
        self.assertTrue(factory.sessions[0].exited)
        self.assertTrue(response.closed)
        self.assertEqual(probe.service.config.max_retries, 0)
        self.assertEqual(probe.service.cache.stats()["entries"], 0)
        self.assertEqual(probe.metrics()["upstream_attempts"], 1)

    def test_wiki_http200_throttle_is_failure_and_response_closed(self):
        response = Response({"error": {"code": "maxlag", "info": "not copied to report"}})
        probe, factory = self.with_response("wiki_api", response)
        outcome = probe.call(0)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.status, 200)
        self.assertEqual(outcome.error_code, "rate_limited")
        self.assertEqual(outcome.retry_after, 5)
        self.assertEqual(len(factory.requests), 1)
        self.assertTrue(response.closed)
        self.assertNotIn("api_cooldowns/wikipedia", probe.metrics())

    def test_wiki_http200_bad_shape_does_not_count_as_success(self):
        response = Response({"query": {"search": "wrong"}})
        probe, _ = self.with_response("wiki_api", response)
        outcome = probe.call(0)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "invalid_upstream_response")
        self.assertEqual(outcome.status, 200)
        self.assertTrue(response.closed)

    def test_wiki_extract_has_no_hidden_identity_warmup(self):
        response = Response({"query": {"pages": {"1": {"pageid": 1, "extract": "full article"}}}})
        probe, factory = self.with_response("wiki_extract", response)
        self.assertTrue(probe.call(0).ok)
        self.assertEqual(len(factory.requests), 1)
        self.assertEqual(factory.requests[0][2]["params"]["prop"], "extracts")
        self.assertEqual(factory.requests[0][2]["params"]["titles"], "Ada Lovelace")

    def test_wiki_identity_validates_page_metadata(self):
        response = Response({"query": {"pages": {"1": {
            "pageid": 1, "title": "Ada Lovelace", "fullurl": "https://en.wikipedia.org/wiki/Ada_Lovelace",
        }}}})
        probe, factory = self.with_response("wiki_identity", response)
        self.assertTrue(probe.call(0).ok)
        self.assertEqual(factory.requests[0][2]["params"]["prop"], "info|pageprops")

    def test_brave_key_precedence_and_http200_error(self):
        with patch.dict(os.environ, {"BRAVE_API_KEY": "preferred", "YIBU_BRAVE_API_KEY": "fallback"}):
            response = Response({"error": {"message": "private provider error"}})
            probe, factory = self.with_response("brave", response)
            outcome = probe.call(0)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.status, 200)
        self.assertEqual(outcome.error_code, "upstream_error")
        self.assertEqual(factory.requests[0][2]["headers"]["Authorization"], "Bearer preferred")
        self.assertNotIn("private", str(outcome))
        self.assertNotIn("preferred", str(outcome))
        self.assertTrue(response.closed)

    def test_missing_brave_key_is_detected_before_network(self):
        with self.assertRaisesRegex(ValueError, "API_KEY"):
            UpstreamProbe("brave", ["Ada"], [])

    def test_http429_has_no_retry(self):
        response = Response({}, status=429, headers={"Retry-After": "2"})
        probe, factory = self.with_response("wiki_api", response)
        outcome = probe.call(0)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.status, 429)
        self.assertEqual(outcome.retry_after, 2.5)
        self.assertEqual(len(factory.requests), 1)
        self.assertEqual(probe.metrics()["upstream_attempts"], 1)
        self.assertTrue(response.closed)

    def test_web_has_production_timeout_and_host_validation(self):
        response = Response(content=b"<html><body>article</body></html>")
        probe, factory = self.with_response("web", response, urls=["https://example.org/article"])
        self.assertTrue(probe.call(0).ok)
        self.assertEqual(factory.requests[0][2]["timeout"], (10.0, 10.0))
        self.assertTrue(response.closed)
        with self.assertRaisesRegex(ValueError, "single hostname"):
            UpstreamProbe("web", [], ["https://example.org/a", "https://other.org/b"])
        with self.assertRaisesRegex(ValueError, "require a URL"):
            UpstreamProbe("web", [], [])

    def test_wiki_html_has_article_defaults_and_wikimedia_ua(self):
        response = Response(content=b"<html>article</html>")
        probe, factory = self.with_response("wiki_html", response)
        self.assertTrue(probe.call(0).ok)
        self.assertEqual(factory.requests[0][2]["headers"]["User-Agent"], MODULE.providers.WIKIMEDIA_USER_AGENT)
        with self.assertRaisesRegex(ValueError, "Wikipedia"):
            UpstreamProbe("wiki_html", [], ["https://example.org/"])

    def test_redirects_are_real_attempts_but_one_logical_call(self):
        first = Response({}, status=302, headers={"Location": "/canonical"})
        second = Response(content=b"<html>text</html>")
        with UpstreamProbe("web", [], ["https://example.org/article"]) as probe:
            factory = SessionFactory([first, second])
            probe.service._session_factory = factory
            outcome = probe.call(0)
            self.assertTrue(outcome.ok)
            self.assertEqual(probe.metrics()["upstream_attempts"], 2)
            self.assertEqual(probe.metrics()["probe_logical_calls"], 1)
            self.assertTrue(first.closed)
            self.assertTrue(second.closed)

    def test_response_size_cap_is_honored(self):
        with patch.dict(os.environ, {"SEARCH_SERVICE_MAX_UPSTREAM_BYTES": "10"}):
            response = Response(content=b"x" * 11)
            probe, _ = self.with_response("web", response, urls=["https://example.org/"])
        outcome = probe.call(0)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "upstream_too_large")
        self.assertTrue(response.closed)


class StageTests(unittest.TestCase):
    def stage(self, call, **kwargs):
        settings = dict(duration=.15, concurrency=2, rpm=6000, max_requests=10, latency_limit=1)
        settings.update(kwargs)
        return probe_stage(call, **settings)

    def test_pacing_avoids_unbounded_dispatch_burst(self):
        starts = []
        lock = threading.Lock()

        def call(index):
            with lock:
                starts.append(time.monotonic())
            return ProbeOutcome(True, status=200)

        result = self.stage(call, rpm=3000, max_requests=5)
        self.assertEqual(result["sent"], 5)
        self.assertGreaterEqual(starts[-1] - starts[0], .06)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["status_counts"], {"200": 5})
        self.assertLessEqual(result["achieved_rpm"], 3300)
        self.assertTrue(result["target_reached"])

    def test_concurrency_ceiling_drops_slots_without_queue(self):
        lock = threading.Lock()
        active = peak = 0

        def call(index):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.06)
            with lock:
                active -= 1
            return ProbeOutcome(True, status=200)

        result = self.stage(call, duration=.12, max_requests=100)
        self.assertLessEqual(peak, 2)
        self.assertEqual(result["peak_inflight"], 2)
        self.assertGreater(result["capacity_skips"], 0)
        self.assertLessEqual(result["sent"], 4)
        self.assertFalse(result["target_reached"])
        self.assertFalse(result["stable"])

    def test_queued_futures_do_not_claim_actual_concurrency(self):
        class DeferredFuture(Future):
            def __init__(self, function, arguments):
                super().__init__()
                self.function = function
                self.arguments = arguments

            def result(self, timeout=None):
                if not self.done():
                    self.set_result(self.function(*self.arguments))
                return super().result(timeout)

        class DeferredPool:
            def __init__(self, **kwargs):
                pass

            def submit(self, function, *arguments):
                return DeferredFuture(function, arguments)

            def shutdown(self, **kwargs):
                pass

        # Four submitted futures execute sequentially only during drain. The
        # dispatcher must not report these as four simultaneous real calls.
        with patch.object(MODULE, "ThreadPoolExecutor", DeferredPool):
            result = self.stage(lambda _: ProbeOutcome(True, status=200),
                                concurrency=4, max_requests=4)
        self.assertEqual(result["sent"], 4)
        self.assertEqual(result["peak_outstanding"], 4)
        self.assertEqual(result["peak_inflight"], 1)
        self.assertFalse(result["concurrency_reached"])

    def test_trailing_capacity_skips_do_not_falsely_reach_target(self):
        def call(index):
            time.sleep(.15)
            return ProbeOutcome(True, status=200)

        result = self.stage(call, concurrency=1, duration=.08, max_requests=100)
        self.assertEqual(result["sent"], 1)
        self.assertGreaterEqual(result["capacity_skips"], 5)
        self.assertLess(result["achieved_rpm"], 1000)
        self.assertFalse(result["target_reached"])
        self.assertGreaterEqual(result["drain_seconds"], .05)

    def test_initial_worker_startup_delay_is_in_rpm_denominator(self):
        real_pool = MODULE.ThreadPoolExecutor

        class SlowStartPool:
            def __init__(self, **kwargs):
                self.pool = real_pool(**kwargs)

            def submit(self, function, *arguments):
                def delayed():
                    time.sleep(.04)
                    return function(*arguments)
                return self.pool.submit(delayed)

            def shutdown(self, **kwargs):
                self.pool.shutdown(**kwargs)

        with patch.object(MODULE, "ThreadPoolExecutor", SlowStartPool):
            result = self.stage(lambda _: ProbeOutcome(True, status=200),
                                concurrency=4, max_requests=4)
        self.assertEqual(result["sent"], 4)
        self.assertGreaterEqual(result["rate_window_seconds"], .07)
        self.assertGreater(result["rate_window_seconds"], result["sending_window_seconds"] + .025)
        self.assertLess(result["achieved_rpm"], 4000)
        self.assertFalse(result["target_reached"])
        self.assertFalse(result["stable"])

    def test_429_stops_new_calls_and_drains_existing(self):
        def call(index):
            if index == 0:
                time.sleep(.015)
                return ProbeOutcome(False, status=429, error_code="upstream_http_error", retry_after=2)
            time.sleep(.04)
            return ProbeOutcome(True, status=200)

        result = self.stage(call, duration=.5, max_requests=100)
        self.assertLessEqual(result["sent"], 2)
        self.assertEqual(result["sent"], result["succeeded"] + result["failed"])
        self.assertTrue(result["rate_limited"])
        self.assertEqual(result["aborted_reason"], "rate_limited")
        self.assertEqual(result["retry_after_seconds"], 2)
        self.assertFalse(result["stable"])

    def test_http200_api_throttle_also_stops(self):
        result = self.stage(lambda _: ProbeOutcome(False, status=200, error_code="rate_limited"))
        self.assertEqual(result["sent"], 1)
        self.assertTrue(result["rate_limited"])
        self.assertEqual(result["status_counts"], {"200": 1})
        self.assertEqual(result["error_codes"], {"rate_limited": 1})

    def test_shared_stop_event_stops_submission_and_drains(self):
        cancelled = threading.Event()

        def call(index):
            cancelled.set()
            time.sleep(.04)
            return ProbeOutcome(True, status=200)

        result = self.stage(call, duration=.5, max_requests=100, stop_event=cancelled)
        self.assertEqual(result["sent"], 1)
        self.assertEqual(result["succeeded"], 1)
        self.assertEqual(result["aborted_reason"], "cancelled")
        self.assertGreaterEqual(result["drain_seconds"], .025)
        self.assertFalse(result["stable"])

    def test_preexisting_stop_event_sends_nothing(self):
        cancelled = threading.Event()
        cancelled.set()

        def call(index):
            self.fail("a cancelled stage must not issue requests")

        result = self.stage(call, stop_event=cancelled)
        self.assertEqual(result["sent"], 0)
        self.assertEqual(result["aborted_reason"], "cancelled")

    def test_failure_latencies_and_exception_sanitizing(self):
        def call(index):
            if index == 0:
                time.sleep(.005)
                return ProbeOutcome(False, status=502, error_code="upstream_http_error")
            time.sleep(.015)
            raise RuntimeError("https://secret:credential@example.org/?private=query")

        result = self.stage(call, max_requests=2, latency_limit=.001)
        self.assertEqual(result["failed"], 2)
        self.assertEqual(result["error_codes"], {"upstream_http_error": 1, "probe_exception": 1})
        self.assertGreater(result["latency_p95"], .01)
        self.assertGreaterEqual(result["latency_max"], result["latency_p99"])
        self.assertNotIn("credential", json.dumps(result))
        self.assertFalse(result["stable"])

    def test_invalid_stage_inputs(self):
        for settings in ({"rpm": 0}, {"concurrency": True}, {"duration": float("inf")}, {"max_requests": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.stage(lambda _: ProbeOutcome(True, status=200), **settings)


if __name__ == "__main__":
    unittest.main()
