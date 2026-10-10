"""Offline tests of production retry paths, without tool imports or live services."""

import ast
import contextlib
import importlib.util
import io
import json
import os
import unittest
from datetime import datetime, timezone
from email.utils import format_datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup


TOOLS = Path(__file__).resolve().parents[1] / "agentflow/tools"


def production_function(path, name, scope):
    """Compile the unchanged production body without import-time global patches."""
    tree = ast.parse((TOOLS / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


spec = importlib.util.spec_from_file_location("search_network_retry", TOOLS / "network_retry.py")
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)

gateway_spec = importlib.util.spec_from_file_location("search_gateway", TOOLS / "search_gateway.py")
gateway = importlib.util.module_from_spec(gateway_spec)
gateway_spec.loader.exec_module(gateway)


def http_response(status, *, headers=None, content=b"", data=None):
    # Real requests status/JSON handling; only transport and response closing are mocked.
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    response.url = "https://test.invalid/"
    response._content = json.dumps(data).encode() if data is not None else content
    response._content_consumed = True
    response.close = Mock()
    return response


def retry_scope():
    return dict(
        requests=requests, time=NS(sleep=Mock()),
        MAX_NETWORK_RETRIES=retry.MAX_NETWORK_RETRIES,
        MAX_RETRY_WAIT_SECONDS=retry.MAX_RETRY_WAIT_SECONDS,
        RETRYABLE_HTTP_STATUSES=retry.RETRYABLE_HTTP_STATUSES,
        YIBU_RETRYABLE_HTTP_STATUSES=retry.YIBU_RETRYABLE_HTTP_STATUSES,
        retry_wait_seconds=retry.retry_wait_seconds,
        SearchGatewayClient=gateway.SearchGatewayClient,
        SearchGatewayError=gateway.SearchGatewayError,
        search_cache_enabled=gateway.search_cache_enabled,
    )


class RetryWaitTests(unittest.TestCase):
    def test_numeric_retry_after_including_long_cooldown(self):
        self.assertEqual(retry.retry_wait_seconds(0, status_code=429, retry_after="11"), 11.5)
        self.assertEqual(retry.retry_wait_seconds(2, status_code=429, retry_after="120"), 120.5)
        self.assertEqual(retry.retry_wait_seconds(0, status_code=429, retry_after="0"), 0.5)

    def test_http_date_retry_after(self):
        fixed_now = datetime(2026, 10, 9, 8, 0, 0, tzinfo=timezone.utc)

        class FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now

        with patch.object(retry, "datetime", FixedDatetime):
            future = format_datetime(fixed_now.replace(minute=1), usegmt=True)
            self.assertEqual(retry.retry_wait_seconds(0, status_code=429, retry_after=future), 60.5)
            past = format_datetime(fixed_now.replace(hour=7), usegmt=True)
            self.assertEqual(retry.retry_wait_seconds(0, status_code=429, retry_after=past), 0.5)

    def test_missing_or_invalid_429_header_uses_bounded_jitter(self):
        for header in (None, "", "invalid", "nan", "inf"):
            for jitter in (0.0, 0.5):
                with self.subTest(header=header, jitter=jitter), patch.object(
                    retry.random, "uniform", return_value=jitter
                ) as uniform:
                    self.assertEqual(
                        retry.retry_wait_seconds(2, status_code=429, retry_after=header),
                        4.0 + jitter,
                    )
                    uniform.assert_called_once_with(0.0, 0.5)

    def test_non_429_uses_exponential_wait_without_retry_after(self):
        for status in (None, 422, 503):
            with self.subTest(status=status):
                self.assertEqual(
                    [retry.retry_wait_seconds(i, status_code=status, retry_after="120") for i in range(3)],
                    [1.0, 2.0, 4.0],
                )


class WikipediaRetryTests(unittest.TestCase):
    def setUp(self):
        self.scope = retry_scope()
        self.error = type("WikipediaRateLimitError", (Exception,), {})
        self.scope.update(WikipediaRateLimitError=self.error, _original_get=Mock())
        self.get = production_function("wikipedia_search/tool.py", "_patched_get", self.scope)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_persistent_429_makes_four_requests_and_honors_long_retry_after(self):
        responses = [http_response(429, headers={"Retry-After": "60"}) for _ in range(4)]
        self.scope["_original_get"].side_effect = responses
        with self.assertRaises(self.error):
            self.get("https://en.wikipedia.org/w/api.php", params={"action": "query"})
        self.assertEqual(self.scope["_original_get"].call_count, 4)
        self.assertEqual([call.args[0] for call in self.scope["time"].sleep.call_args_list], [60.5] * 3)
        for response in responses:
            response.close.assert_called_once()

    def test_success_after_429_returns_original_response(self):
        limited = http_response(429, headers={"Retry-After": "11"})
        success = http_response(200, data={"query": {"pages": {"1": {"title": "Moon"}}}})
        self.scope["_original_get"].side_effect = [limited, success]
        self.assertIs(self.get("https://en.wikipedia.org/w/api.php"), success)
        self.assertEqual(success.json()["query"]["pages"]["1"]["title"], "Moon")
        self.assertEqual(self.scope["_original_get"].call_count, 2)
        self.scope["time"].sleep.assert_called_once_with(11.5)
        self.assertIs(self.scope["_original_get"].call_args.kwargs["verify"], False)

    def test_permanent_403_is_not_retried(self):
        forbidden = http_response(403)
        self.scope["_original_get"].return_value = forbidden
        self.assertIs(self.get("https://en.wikipedia.org/w/api.php"), forbidden)
        self.scope["_original_get"].assert_called_once()
        self.scope["time"].sleep.assert_not_called()

    def test_timeout_and_503_can_recover(self):
        success = http_response(200)
        self.scope["_original_get"].side_effect = [requests.exceptions.Timeout(), http_response(503), success]
        self.assertIs(self.get("https://en.wikipedia.org/w/api.php"), success)
        self.assertEqual(self.scope["_original_get"].call_count, 3)
        self.assertEqual([call.args[0] for call in self.scope["time"].sleep.call_args_list], [1.0, 2.0])


class WebRetryTests(unittest.TestCase):
    def setUp(self):
        # These tests exercise direct transport retries, independently of the
        # caller's shared search-service configuration.
        cache_patch = patch.dict(os.environ, {"SEARCH_CACHE_ENABLED": "0"})
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.scope = retry_scope()
        self.scope.update(os=os, urlsplit=urlsplit, BeautifulSoup=BeautifulSoup)
        self.fetch = production_function("web_search/tool.py", "_get_website_content", self.scope)
        self.session = Mock()
        self.session.__enter__ = Mock(return_value=self.session)
        self.session.__exit__ = Mock(return_value=False)
        session_patch = patch.object(requests, "Session", return_value=self.session)
        session_patch.start()
        self.addCleanup(session_patch.stop)
        global_get_patch = patch.object(requests, "get", side_effect=AssertionError("nested requests.get retry"))
        self.global_get = global_get_patch.start()
        self.addCleanup(global_get_patch.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_wiki_429_is_bounded_without_nested_retry_and_keeps_identifiable_ua(self):
        self.session.get.side_effect = [http_response(429, headers={"Retry-After": "11"}) for _ in range(4)]
        with patch.dict(os.environ, {"WIKIMEDIA_USER_AGENT": "ResearchBot/1.0 (https://test.invalid/contact)"}):
            result = self.fetch(NS(max_window_size=100), "https://en.wikipedia.org/wiki/Moon")
        self.assertTrue(result.startswith("Error fetching URL:"), result)
        self.assertIn("429", result)
        self.assertEqual(self.session.get.call_count, 4)
        self.global_get.assert_not_called()
        self.assertEqual([call.args[0] for call in self.scope["time"].sleep.call_args_list], [11.5] * 3)
        for call in self.session.get.call_args_list:
            self.assertEqual(call.kwargs["headers"]["User-Agent"], "ResearchBot/1.0 (https://test.invalid/contact)")
            self.assertIs(call.kwargs["verify"], False)

    def test_success_after_429_extracts_html_and_preserves_text_limit(self):
        self.session.get.side_effect = [
            http_response(429, headers={"Retry-After": "0"}),
            http_response(200, content=b"<html><body><p>Moon evidence</p><p>Second paragraph</p></body></html>"),
        ]
        result = self.fetch(NS(max_window_size=13), "https://example.test/page")
        self.assertEqual(result, "Moon evidence")
        self.assertEqual(self.session.get.call_count, 2)
        self.scope["time"].sleep.assert_called_once_with(0.5)

    def test_permanent_error_or_long_retry_after_does_not_retry(self):
        for status, headers in ((403, {}), (429, {"Retry-After": "60"})):
            with self.subTest(status=status):
                self.session.get.reset_mock()
                self.scope["time"].sleep.reset_mock()
                self.session.get.return_value = http_response(status, headers=headers)
                result = self.fetch(NS(max_window_size=100), "https://example.test/page")
                self.assertTrue(result.startswith("Error fetching URL:"), result)
                self.session.get.assert_called_once()
                self.scope["time"].sleep.assert_not_called()


class BraveRetryTests(unittest.TestCase):
    def setUp(self):
        cache_patch = patch.dict(os.environ, {"SEARCH_CACHE_ENABLED": "0"})
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.scope = retry_scope()
        self.scope.update(Any=object, Dict=dict, List=list, Optional=Optional)
        self.search = production_function("brave_search/tool.py", "_execute_search", self.scope)
        self.tool = NS(api_key="test", endpoint="https://search.test/v1", timeout=20,
                       max_retries=retry.MAX_NETWORK_RETRIES)
        self.tool._as_text = production_function("brave_search/tool.py", "_as_text", self.scope)
        self.tool._extract_results = production_function("brave_search/tool.py", "_extract_results", self.scope)
        self.tool._format_results = partial(production_function("brave_search/tool.py", "_format_results", self.scope), self.tool)
        self.get_patch = patch.object(requests, "get")
        self.get = self.get_patch.start()
        self.addCleanup(self.get_patch.stop)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_422_can_recover_and_returns_actual_formatted_search_results(self):
        self.get.side_effect = [http_response(422), http_response(200, data={"web": {"results": [{
            "title": "Moon", "url": "https://example.test/moon", "description": "Useful evidence",
        }]}})]
        result = self.search(self.tool, "moon mass", count=99)
        self.assertIn("[1] Moon", result)
        self.assertIn("Description: Useful evidence", result)
        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(self.get.call_args.kwargs["params"], {"q": "moon mass", "count": 20})
        self.assertEqual(self.get.call_args.kwargs["headers"]["Authorization"], "Bearer test")
        self.scope["time"].sleep.assert_called_once_with(1.0)

    def test_persistent_429_makes_four_requests(self):
        self.get.side_effect = [http_response(429, headers={"Retry-After": "11"}) for _ in range(4)]
        result = self.search(self.tool, "moon mass")
        self.assertIn("tried 4 times but failed", result)
        self.assertIn("429", result)
        self.assertEqual(self.get.call_count, 4)
        self.assertEqual([call.args[0] for call in self.scope["time"].sleep.call_args_list], [11.5] * 3)

    def test_permanent_401_or_long_retry_after_does_not_retry(self):
        for status, headers in ((401, {}), (429, {"Retry-After": "60"})):
            with self.subTest(status=status):
                self.get.reset_mock()
                self.scope["time"].sleep.reset_mock()
                self.get.return_value = http_response(status, headers=headers)
                result = self.search(self.tool, "moon mass")
                self.assertIn("tried 1 time but failed", result)
                self.get.assert_called_once()
                self.scope["time"].sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
