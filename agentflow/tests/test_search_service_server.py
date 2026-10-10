"""Actual loopback HTTP protocol tests; never contact an external provider."""
import gzip
import http.client
import io
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.search_service.core import DEFAULT_CACHE_DB_PATH, GatewayFailure, SearchService, ServiceConfig
from agentflow.search_service.server import ServiceHTTPServer, main


class ProtocolProviders:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def prepare(self, tool, params):
        if tool not in {"wikipedia", "brave", "fetch"} or not isinstance(params, dict):
            raise GatewayFailure("Invalid params", code="invalid_request", status_code=400)
        key = "url" if tool == "fetch" else "query"
        if not isinstance(params.get(key), str) or not params[key]:
            raise GatewayFailure("Missing key", code="invalid_request", status_code=400)
        return {key: params[key]}

    def execute(self, tool, params, service):
        def producer():
            with self.lock:
                self.calls += 1
            value = next(iter(params.values()))
            if value == "429":
                raise GatewayFailure("Upstream limited", code="upstream_http_error", status_code=429,
                                     retryable=True, upstream="wikipedia", upstream_status=429,
                                     retry_after=2.5, attempts=4)
            if tool == "fetch":
                return {"text": value * 200}
            if tool == "brave":
                return {"data": {"web": {"results": [{"title": value}, {"title": "second"}]}}}
            return {"results": [{"title": value, "ordered": [3, 1, 2]}, {"title": "second"}]}
        return service.cached("protocol:" + tool, params, producer, 3600)


class ServiceHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.providers = ProtocolProviders()
        self.start()
        self.addCleanup(self.stop)

    def start(self):
        self.service = SearchService(ServiceConfig(cache_dir=self.directory.name,
            db_path=str(Path(self.directory.name) / "cache.sqlite3"), token="secret-test-token"), self.providers)
        self.server = ServiceHTTPServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
        self.service.close()

    def request(self, path, body=None, *, method=None, auth=True, headers=None, raw=None):
        options = {"Authorization": "Bearer secret-test-token"} if auth else {}
        if body is not None or raw is not None:
            options["Content-Type"] = "application/json"
        options.update(headers or {})
        payload = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.request(method or ("POST" if payload is not None else "GET"), path, payload, options)
            response = connection.getresponse()
            result, response_headers = response.read(), dict(response.getheaders())
            if response_headers.get("Content-Encoding") == "gzip":
                result = gzip.decompress(result)
            return response.status, response_headers, json.loads(result)
        finally:
            connection.close()

    def test_health_and_metrics_require_correct_bearer_token(self):
        status, _, body = self.request("/healthz")
        self.assertEqual((status, body), (200, {"status": "ok"}))
        for path in ("/healthz", "/metrics", "/v1/search/wikipedia"):
            status, _, body = self.request(path, {"query": "A"} if path.startswith("/v1") else None,
                                           headers={"Authorization": "Bearer wrong"})
            self.assertEqual(status, 401)
            self.assertEqual(body["error"]["code"], "unauthorized")
        self.assertEqual(self.providers.calls, 0)

    def test_metrics_database_failure_returns_json_and_recovers(self):
        with patch.object(self.service.cache, "stats", side_effect=sqlite3.DatabaseError("database disk image is malformed")):
            with self.assertLogs("search_service", level="ERROR") as logs:
                status, headers, body = self.request("/metrics")
            self.assertEqual(status, 503)
            self.assertEqual(body["error"]["code"], "cache_database_error")
            self.assertEqual(body["request_id"], headers["X-Request-ID"])
            self.assertNotIn("malformed", json.dumps(body))
            self.assertTrue(any("malformed" in line for line in logs.output))
            self.assertEqual(self.request("/healthz")[0], 200)
            self.assertEqual(self.request("/metrics", auth=False)[0], 401)
        self.assertEqual(self.request("/metrics")[0], 200)

    def test_metrics_unexpected_failure_returns_json(self):
        with patch.object(self.service, "metrics_snapshot", side_effect=RuntimeError("internal detail")):
            with self.assertLogs("search_service", level="ERROR"):
                status, _, body = self.request("/metrics")
        self.assertEqual(status, 500)
        self.assertEqual(body["error"]["code"], "internal_error")
        self.assertNotIn("internal detail", json.dumps(body))

    def test_all_tool_protocols_preserve_complete_multiple_results(self):
        status, headers, body = self.request("/v1/search/wikipedia", {"query": "A"})
        self.assertEqual(status, 200)
        self.assertEqual([item["title"] for item in body["results"]], ["A", "second"])
        self.assertEqual(body["results"][0]["ordered"], [3, 1, 2])
        self.assertEqual(body["meta"]["request_id"], headers["X-Request-ID"])
        status, _, body = self.request("/v1/search/brave", {"query": "B"})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["data"]["web"]["results"]), 2)
        status, _, body = self.request("/v1/fetch", {"url": "https://test.invalid/page"})
        self.assertEqual(status, 200)
        self.assertTrue(body["text"].startswith("https://test.invalid/page"))

    def test_restart_reads_persisted_cache_without_calling_provider(self):
        self.request("/v1/search/wikipedia", {"query": "persistent"})
        self.assertEqual(self.providers.calls, 1)
        self.stop()
        self.start()
        status, _, body = self.request("/v1/search/wikipedia", {"query": "persistent"})
        self.assertEqual(status, 200)
        self.assertEqual(body["meta"]["cache_hits"], 1)
        self.assertEqual(self.providers.calls, 1)

    def test_batch_preserves_order_deduplicates_and_reports_per_item_errors(self):
        items = [{"tool": "wikipedia", "params": {"query": query}} for query in ["B", "A", "B"]]
        items += [{"tool": "unknown", "params": {}}, {"tool": "wikipedia", "params": {"query": "429"}}]
        status, _, body = self.request("/v1/batch", {"requests": items})
        self.assertEqual(status, 200)
        self.assertEqual([item["data"]["results"][0]["title"] for item in body["results"][:3]], ["B", "A", "B"])
        self.assertFalse(body["results"][3]["ok"])
        self.assertEqual(body["results"][4]["error"]["upstream_status"], 429)
        _, _, metrics = self.request("/metrics")
        self.assertEqual(metrics["logical_requests"], 5)
        self.assertEqual(metrics["unique_executions"], 3)
        self.assertEqual(metrics["batch_duplicates"], 1)

    def test_simultaneous_http_calls_share_cached_result(self):
        with ThreadPoolExecutor(max_workers=16) as executor:
            replies = list(executor.map(lambda _: self.request("/v1/search/wikipedia", {"query": "same"}), range(32)))
        self.assertTrue(all(reply[0] == 200 for reply in replies))
        self.assertEqual(self.providers.calls, 1)

    def test_gzip_input_and_output_roundtrip(self):
        query = "long-query-" * 100
        status, headers, body = self.request("/v1/search/wikipedia", raw=gzip.compress(json.dumps({"query": query}).encode()),
                                             headers={"Content-Encoding": "gzip", "Accept-Encoding": "gzip"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Encoding"], "gzip")
        self.assertEqual(body["results"][0]["title"], query)

    def test_malformed_and_nonfinite_json_cannot_enter_cache(self):
        for raw in (b"{", b"[]", b'{"query":NaN}', b'{"query":Infinity}', b'{"query":1e999}'):
            status, _, body = self.request("/v1/search/wikipedia", raw=raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(body["error"]["code"], "invalid_request")
        self.assertEqual(self.providers.calls, 0)
        self.assertEqual(self.service.cache.stats()["entries"], 0)

    def test_oversized_payload_and_gzip_expansion_are_rejected(self):
        self.service.config.max_body_bytes = 100
        for raw, headers in ((b"x" * 101, {}),
                             (gzip.compress(b'{"query":"' + b"x" * 500 + b'"}'), {"Content-Encoding": "gzip"})):
            status, _, body = self.request("/v1/search/wikipedia", raw=raw, headers=headers)
            self.assertEqual(status, 413)
            self.assertEqual(body["error"]["code"], "request_too_large")

    def test_permanent_protocol_errors_are_structured(self):
        for path, payload, headers, expected in (
            ("/unknown", {}, {}, 404),
            ("/v1/search/wikipedia", {}, {}, 400),
            ("/v1/search/wikipedia", {"query": "A"}, {"Content-Type": "text/plain"}, 415),
            ("/v1/batch", {"requests": []}, {}, 400),
            ("/v1/batch", {"requests": [], "extra": 1}, {}, 400),
        ):
            status, _, body = self.request(path, payload, headers=headers)
            self.assertEqual(status, expected)
            self.assertIn("error", body)

    def test_final_upstream_failure_has_retry_after_and_is_not_cached(self):
        for _ in range(2):
            status, headers, body = self.request("/v1/search/wikipedia", {"query": "429"})
            self.assertEqual(status, 429)
            self.assertEqual(headers["Retry-After"], "3")
            self.assertEqual(body["error"]["attempts"], 4)
            self.assertEqual(body["error"]["upstream"], "wikipedia")
        self.assertEqual(self.providers.calls, 2)
        self.assertEqual(self.service.cache.stats()["entries"], 0)


class ServerConfigurationTests(unittest.TestCase):
    def test_database_default_is_independent_of_state_directory(self):
        config = ServiceConfig.from_env({"SEARCH_CACHE_DIR": "/mounted/s3/state"})
        self.assertEqual(config.db_path, "/var/tmp/agentflow-search-cache/search_cache.sqlite3")
        self.assertEqual(ServiceConfig().db_path, DEFAULT_CACHE_DB_PATH)
        custom = ServiceConfig.from_env({"SEARCH_CACHE_DB_PATH": "/local/custom/cache.sqlite3"})
        self.assertEqual(custom.db_path, "/local/custom/cache.sqlite3")

    def test_missing_token_is_rejected_before_starting_service(self):
        with patch.dict("os.environ", {}, clear=True), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--port", "0"])
        self.assertEqual(error.exception.code, 2)

    def test_server_token_can_use_client_token_as_explicit_fallback(self):
        config = ServiceConfig.from_env({"SEARCH_CACHE_TOKEN": "shared-test-token"})
        self.assertEqual(config.token, "shared-test-token")

    def test_invalid_nonfinite_configuration_is_rejected(self):
        for name in ("SEARCH_WIKI_RPM", "SEARCH_SERVICE_REQUEST_TIMEOUT", "SEARCH_CACHE_MAX_TTL"):
            with self.assertRaises(ValueError):
                ServiceConfig.from_env({name: "nan"})


if __name__ == "__main__":
    unittest.main()
