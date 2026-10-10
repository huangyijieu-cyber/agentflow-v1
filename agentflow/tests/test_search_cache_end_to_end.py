"""Real local HTTP: client -> service -> provider, without Internet or models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agentflow"))

from agentflow.search_service.core import SearchService, ServiceConfig
from agentflow.search_service.server import ServiceHTTPServer
from agentflow.tools.search_gateway import SearchGatewayClient


class LocalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.calls = Counter()
        self.lock = threading.Lock()
        test = self

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                params = parse_qs(urlsplit(self.path).query)
                query = params.get("q", ["page"])[0]
                with test.lock:
                    test.calls[query] += 1
                time.sleep(0.03)
                if self.path.startswith("/page"):
                    body = b"<html><body><p>Alpha fact</p><p>Beta fact</p></body></html>"
                    kind = "text/html"
                else:
                    data = {"error": "provider failure"} if query == "bad" else {
                        "web": {"results": [
                            {"title": query + " second", "url": "https://example.test/2", "description": "Fact B"},
                            {"title": query + " first", "url": "https://example.test/1", "description": "Fact A", "extra_snippets": ["extra"]},
                        ]}}
                    body = json.dumps(data).encode()
                    kind = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        self.upstream_url = f"http://127.0.0.1:{self.upstream.server_port}"
        self.environ = patch.dict(os.environ, {
            "BRAVE_YIBU_BASE_URL": self.upstream_url + "/search",
            "YIBU_BRAVE_API_KEY": "test-upstream-key", "BRAVE_API_KEY": "",
            "HTTP_PROXY": "", "HTTPS_PROXY": "", "http_proxy": "", "https_proxy": "",
            "ALL_PROXY": "", "all_proxy": "",
        })
        self.environ.start()
        self.start_service()

    def start_service(self):
        self.service = SearchService(ServiceConfig(
            cache_dir=self.temp.name, token="test-shared-token", max_retries=0,
            workers=8, queue_size=32, request_timeout=5,
            wiki_rpm=100000, brave_rpm=100000, web_rpm=100000,
        ))
        self.http = ServiceHTTPServer(("127.0.0.1", 0), self.service)
        self.http_thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.http_thread.start()
        self.base_url = f"http://127.0.0.1:{self.http.server_port}"

    def stop_service(self):
        self.http.shutdown()
        self.http.server_close()
        self.http_thread.join(timeout=3)
        self.service.close()

    def tearDown(self):
        self.stop_service()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upstream_thread.join(timeout=3)
        self.environ.stop()
        self.temp.cleanup()

    def client(self):
        return SearchGatewayClient(self.base_url, "test-shared-token", read_timeout=10)

    def test_batch_preserves_cardinality_order_and_independent_objects(self):
        queries = ["A", "B", "A", "C", "B"]
        with self.client() as client:
            results = client.batch([{"tool": "brave", "params": {"query": q}} for q in queries])
        self.assertTrue(all(item["ok"] for item in results))
        self.assertEqual([r["data"]["data"]["web"]["results"][0]["title"] for r in results],
                         [q + " second" for q in queries])
        self.assertEqual(self.calls, Counter(A=1, B=1, C=1))
        results[0]["data"]["data"]["web"]["results"][0]["title"] = "changed"
        self.assertEqual(results[2]["data"]["data"]["web"]["results"][0]["title"], "A second")

    def test_concurrent_jobs_share_one_fetch_and_persist_across_restart(self):
        barrier = threading.Barrier(8)

        def run():
            barrier.wait()
            with self.client() as client:
                return client.brave_search("shared")

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(lambda _: run(), range(8)))
        self.assertEqual(self.calls["shared"], 1)
        self.assertTrue(all(r == responses[0] for r in responses))
        self.stop_service()
        self.start_service()
        with self.client() as client:
            self.assertEqual(client.brave_search("shared"), responses[0])
        self.assertEqual(self.calls["shared"], 1)
        self.assertEqual(self.service.metrics.snapshot().get("upstream_attempts", 0), 0)

    def test_duplicate_failures_shared_within_batch_but_not_cached(self):
        req = {"tool": "brave", "params": {"query": "bad"}}
        with self.client() as client:
            for _ in range(2):
                results = client.batch([req, req])
                self.assertEqual(len(results), 2)
                self.assertTrue(all(not r["ok"] for r in results))
        self.assertEqual(self.calls["bad"], 2)

    def test_same_page_different_lengths_share_raw_text_and_preflight_connects(self):
        with self.client() as client:
            a = client.fetch(self.upstream_url + "/page", max_length=5)
            b = client.fetch(self.upstream_url + "/page", max_length=1000)
        self.assertEqual(a["text"], "Alpha")
        self.assertEqual(b["text"], "Alpha fact\nBeta fact")
        self.assertEqual(self.calls["page"], 1)
        env = dict(os.environ, SEARCH_CACHE_BASE_URL=self.base_url,
                   SEARCH_CACHE_TOKEN="test-shared-token", SEARCH_CACHE_PYTHON=sys.executable)
        command = ["bash", "-c", 'source train-roma/enable_search_cache.sh && test "$SEARCH_CACHE_ENABLED" = 1']
        good = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(good.returncode, 0, good.stderr)
        env["SEARCH_CACHE_TOKEN"] = "wrong-token"
        bad = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(bad.returncode, 0)
        self.assertNotIn("wrong-token", bad.stderr)


if __name__ == "__main__":
    unittest.main()
