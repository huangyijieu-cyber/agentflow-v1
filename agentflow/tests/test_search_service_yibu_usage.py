"""Offline real-provider accounting, without contacting Yibu or loading a model."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.search_service.core import GatewayFailure, SearchService, ServiceConfig
from agentflow.search_service.server import ServiceRequestHandler


class Response:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self.headers = {}
        self.encoding = "utf-8"
        self._content_consumed = False
        self.payload = json.dumps(data if data is not None else {
            "web": {"results": [{"title": "结果", "url": "https://example.test"}]},
        }, ensure_ascii=False).encode()
        self.close = Mock()

    def iter_content(self, chunk_size):
        yield self.payload

    @property
    def text(self):
        raise AssertionError("Statistics must never trigger response.text/network reads")

    def json(self):
        return json.loads(self._content)


class Session:
    def __init__(self):
        self.request = Mock()
        self.close = Mock()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class YibuServiceUsageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = patch.dict(os.environ, {"YIBU_BRAVE_API_KEY": "test-secret",
                                      "BRAVE_YIBU_BASE_URL": "https://example.test/search"})
        env.start()
        self.addCleanup(env.stop)
        self.session = Session()
        self.session.request.return_value = Response()
        self.service = SearchService(ServiceConfig(
            cache_dir=self.directory.name,
            db_path=str(Path(self.directory.name) / "cache.sqlite3"), max_retries=1,
        ), session_factory=lambda: self.session)
        self.addCleanup(self.service.close)

    def search(self, query="query"):
        return self.service.handle("brave", {"query": query})

    def test_cold_then_cache_hit_does_not_repeat_actual_traffic(self):
        cold = self.search()
        events = cold["meta"]["yibu_usage_events"]
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["event_id"], f'{event["call_id"]}:1')
        self.assertEqual(event["input_text"], "query")
        self.assertEqual(event["response_body_text"], self.session.request.return_value.payload.decode())
        self.assertTrue(event["http_attempt"])
        self.assertEqual(event["status"], "success")
        self.service.flush_cache()
        warm = self.search()
        self.assertNotIn("yibu_usage_events", warm["meta"])
        self.assertEqual(cold["data"], warm["data"])
        self.assertEqual(self.session.request.call_count, 1)
        self.assertNotIn("test-secret", json.dumps(events))

    def test_coalesced_callers_share_event_identity(self):
        started, release = threading.Event(), threading.Event()

        def request(*args, **kwargs):
            started.set()
            if not release.wait(2):
                raise AssertionError("producer was not released")
            return Response()

        self.session.request.side_effect = request
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.search)
            self.assertTrue(started.wait(1))
            second = pool.submit(self.search)
            until = time.monotonic() + 1
            while self.service.metrics.snapshot().get("coalesced_waiters", 0) < 1:
                if time.monotonic() >= until:
                    self.fail("second caller never joined the flight")
                time.sleep(0.002)
            release.set()
            events = [result.result()["meta"]["yibu_usage_events"] for result in (first, second)]
        self.assertEqual(events[0], events[1])
        self.assertEqual(self.session.request.call_count, 1)

    def test_retry_records_each_attempt_without_reading_error_body(self):
        self.session.request.side_effect = [Response(503), Response()]
        with patch.object(self.service, "_sleep"):
            events = self.search()["meta"]["yibu_usage_events"]
        self.assertEqual([event["attempt"] for event in events], [1, 2])
        self.assertEqual(len({event["call_id"] for event in events}), 1)
        self.assertEqual([event["http_status"] for event in events], [503, 200])
        self.assertIsNone(events[0]["response_body_text"])
        self.assertIsNotNone(events[1]["response_body_text"])

    def test_http_failure_has_usage_separate_from_safe_diagnostics(self):
        self.session.request.return_value = Response(403)
        with self.assertRaises(GatewayFailure) as caught:
            self.search("private query")
        error = caught.exception
        self.assertEqual(error.code, "upstream_http_error")
        self.assertEqual(len(error.yibu_usage_events), 1)
        event = error.yibu_usage_events[0]
        self.assertIsNone(event["response_body_text"])
        self.assertEqual(event["http_status"], 403)
        self.assertNotIn("private query", json.dumps(error.as_dict()))
        self.assertNotIn("private query", str(error))
        copied = error.with_diagnostics(phase="retry_wait")
        self.assertEqual(copied.yibu_usage_events, error.yibu_usage_events)
        handler = Mock()
        ServiceRequestHandler._error(handler, error, "request-id")
        payload = handler._send_json.call_args.args[1]
        self.assertEqual(payload["yibu_usage_events"], error.yibu_usage_events)
        self.assertNotIn("yibu_usage_events", payload["error"])

    def test_network_retries_are_counted_with_null_body(self):
        self.session.request.side_effect = requests.ConnectTimeout("private raw failure")
        with patch.object(self.service, "_sleep"), self.assertRaises(GatewayFailure) as caught:
            self.search()
        events = caught.exception.yibu_usage_events
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event["http_status"] is None for event in events))
        self.assertTrue(all(event["response_body_text"] is None for event in events))
        self.assertEqual(events[0]["error_type"], "ConnectTimeout")
        self.assertNotIn("private raw failure", json.dumps(events))

    def test_invalid_provider_payload_preserves_http_attempt(self):
        self.session.request.return_value = Response(200, {"unexpected": "payload"})
        with self.assertRaises(GatewayFailure) as caught:
            self.search()
        self.assertEqual(caught.exception.code, "invalid_upstream_response")
        self.assertEqual(len(caught.exception.yibu_usage_events), 1)
        self.assertIn("unexpected", caught.exception.yibu_usage_events[0]["response_body_text"])


if __name__ == "__main__":
    unittest.main()
