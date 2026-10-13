"""Offline checks of the real client and tool functions, without model servers."""
import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock, patch

import requests
import httpx
import openai
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "agentflow/agentflow/tools"
sys.path.insert(0, str(ROOT / "agentflow"))
from agentflow.tools import execution_context as execution
spec = importlib.util.spec_from_file_location("cache_gateway_under_test", TOOLS / "search_gateway.py")
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)


def extract(path, name, scope):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class Response:
    def __init__(self, data=None, status=200, headers=None):
        self.data = data
        self.status_code = status
        self.headers = headers or {}
        self.content = b"<html><body>Page body</body></html>"

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def close(self):
        pass


class Session:
    def __init__(self, response=None, error=None):
        self.response = response or Response({})
        self.error = error
        self.calls = []
        self.trust_env = True
        self.closed = False

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


class GatewayClientTests(unittest.TestCase):
    def client(self, data=None, **kwargs):
        session = Session(Response(data, **kwargs))
        return gateway.SearchGatewayClient("https://development.test/search", "test-token", session=session), session

    def test_existing_ec2_env_does_not_enable_cache(self):
        self.assertFalse(gateway.search_cache_enabled({"SEARCH_GATEWAY_BASE_URL": "http://ec2.test"}))
        self.assertFalse(gateway.search_cache_enabled({"SEARCH_CACHE_ENABLED": "0"}))
        self.assertTrue(gateway.search_cache_enabled({"SEARCH_CACHE_ENABLED": "1"}))

    def test_cache_configuration_takes_precedence_and_finite_timeouts(self):
        client = gateway.SearchGatewayClient.from_env(environ={
            "SEARCH_CACHE_BASE_URL": "https://development.test",
            "SEARCH_CACHE_TOKEN": "cache-token",
            "SEARCH_GATEWAY_BASE_URL": "https://ec2.test",
            "GATEWAY_TOKEN": "old-token",
            "SEARCH_CACHE_CONNECT_TIMEOUT_SECONDS": "2",
            "SEARCH_CACHE_READ_TIMEOUT_SECONDS": "40",
        }, session=Session())
        self.assertEqual(client.base_url, "https://development.test")
        self.assertEqual(client.token, "cache-token")
        self.assertEqual((client.connect_timeout, client.read_timeout), (2.0, 40.0))
        with self.assertRaises(gateway.SearchGatewayConfigurationError):
            gateway.SearchGatewayClient("http://test", "token", read_timeout=float("inf"))

    def test_legacy_gateway_configuration_still_works(self):
        client = gateway.SearchGatewayClient.from_env(environ={
            "SEARCH_GATEWAY_BASE_URL": "http://ec2.test",
            "GATEWAY_TOKEN": "legacy-token",
        }, session=Session())
        self.assertEqual(client.base_url, "http://ec2.test")
        self.assertEqual(client.token, "legacy-token")
        self.assertEqual(client.read_timeout, 150.0)

    def test_proxy_failure_preserves_upstream_status_attempts_and_stops_shared_route(self):
        client, session = self.client({"error": {
            "code": "upstream_proxy_error", "message": "Upstream proxy connection failed (HTTP 407)",
            "upstream_status": 407, "upstream": "wikipedia", "attempts": 1, "retryable": False,
        }}, status=502)
        with self.assertRaises(gateway.SearchGatewayError) as caught:
            client.fetch("https://page.test")
        error = caught.exception
        self.assertEqual((error.code, error.upstream_status_code, error.attempts), ("upstream_proxy_error", 407, 1))
        self.assertEqual(error.upstream, "wikipedia")
        self.assertFalse(error.retryable)
        self.assertTrue(error.is_shared_failure)
        self.assertEqual(len(session.calls), 1)

    def test_single_page_failure_is_not_mistaken_for_shared_outage(self):
        for code, status in (("domain_blocked", 410), ("upstream_http_error", 404)):
            error = gateway.SearchGatewayError("page unavailable", code=code, upstream_status_code=status)
            self.assertFalse(error.is_shared_failure)
        error = gateway.SearchGatewayError("service unavailable", code="gateway_http_503", status_code=503)
        self.assertTrue(error.is_shared_failure)

    def test_auth_proxy_bypass_tls_and_no_redirect(self):
        client, session = self.client({"text": "body", "meta": {"cache": "hit"}})
        with client:
            client.fetch("https://example.test/page", max_length=123)
        method, url, kwargs = session.calls[0]
        self.assertFalse(session.trust_env)
        self.assertTrue(session.closed)
        self.assertEqual((method, url), ("POST", "https://development.test/search/v1/fetch"))
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-token")
        self.assertIs(kwargs["verify"], True)
        self.assertIs(kwargs["allow_redirects"], False)
        self.assertEqual(kwargs["json"], {"url": "https://example.test/page", "max_length": 123})

    def test_wikipedia_preserves_multiple_results_order_and_effective_parameters(self):
        pages = [{"title": title, "url": f"https://en.wikipedia.org/wiki/{title}", "abstract": title} for title in ["B", "A"]]
        client, session = self.client({"results": pages, "meta": {"cache": "hit"}})
        result = client.wikipedia_search("  moon mass  ", max_pages=2, max_length=-1, language="zh")
        self.assertEqual([p["title"] for p in result["results"]], ["B", "A"])
        self.assertEqual(session.calls[0][2]["json"], {
            "query": "  moon mass  ", "max_pages": 2, "max_length": -1, "language": "zh",
        })

    def test_results_are_isolated_between_rollouts(self):
        data = {"results": [{"title": "Moon", "abstract": "body", "url": "https://wiki.test/Moon"}]}
        client, _ = self.client(data)
        first = client.wikipedia_search("Moon")
        first["results"][0]["retrieved_information"] = "rollout A"
        second = client.wikipedia_search("Moon")
        self.assertNotIn("retrieved_information", second["results"][0])
        self.assertNotIn("retrieved_information", data["results"][0])

    def test_brave_unwraps_service_metadata_and_accepts_legacy_raw_response(self):
        raw = {"web": {"results": [{"title": "A"}, {"title": "B"}]}}
        for data in (raw, {"data": raw, "meta": {"cache": "hit"}}):
            client, session = self.client(data)
            self.assertEqual(client.brave_search("query", count=2, freshness="pd"), raw)
            self.assertEqual(session.calls[0][2]["json"], {"query": "query", "count": 2, "freshness": "pd"})

    def test_batch_preserves_per_item_success_failure_and_order(self):
        items = [{"tool": "wikipedia", "params": {"query": "moon"}}, {"tool": "fetch", "params": {"url": "https://example.test"}}]
        expected = [{"ok": True, "data": {"results": ["moon"]}}, {"ok": False, "error": {"code": "upstream_timeout"}}]
        client, session = self.client({"results": expected})
        self.assertEqual(client.batch(items), expected)
        self.assertEqual(session.calls[0][2]["json"], {"requests": items})
        self.assertTrue(session.calls[0][1].endswith("/v1/batch"))

    def test_invalid_batch_response_is_rejected(self):
        for result in ({"results": []}, {"results": [{"ok": "true", "data": []}]}, {"results": [{"ok": False}]}):
            client, _ = self.client(result)
            with self.assertRaises(gateway.SearchGatewayError):
                client.batch([{"tool": "fetch", "params": {}}])

    def test_service_timeout_has_one_attempt_and_no_direct_fallback(self):
        session = Session(error=requests.Timeout("secret transport detail"))
        client = gateway.SearchGatewayClient("https://development.test", "token", session=session)
        with self.assertRaises(gateway.SearchGatewayError) as caught:
            client.fetch("https://example.test")
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(caught.exception.code, "gateway_unreachable")
        self.assertNotIn("secret", str(caught.exception))

    def test_tool_remaining_time_reaches_rpc_and_socket_timeouts(self):
        client, session = self.client({"text": "body"})
        with execution.ensure_tool_context(timeout=0.2):
            self.assertEqual(client.fetch("https://page.test")["text"], "body")
        kwargs = session.calls[0][2]
        self.assertLessEqual(float(kwargs["headers"]["X-Search-Timeout-Seconds"]), 0.2)
        self.assertTrue(all(0 < value <= 0.2 for value in kwargs["timeout"]))

    def test_service_429_keeps_upstream_details_and_retry_after(self):
        client, _ = self.client({"error": {"code": "wikipedia_http_429", "message": "Wiki rate limited", "upstream_status": 429}, "request_id": "test-req"}, status=502, headers={"Retry-After": "11"})
        with self.assertRaises(gateway.SearchGatewayError) as caught:
            client.wikipedia_search("Moon")
        self.assertEqual(caught.exception.upstream_status_code, 429)
        self.assertEqual(caught.exception.retry_after, "11")
        self.assertEqual(caught.exception.request_id, "test-req")

    def test_invalid_json_and_missing_results_are_explicit_errors(self):
        for data in (ValueError("not JSON"), {"results": "bad"}, {"meta": {}}, {"results": ["bad page"]}, {"results": [{"title": "no content"}]}):
            client, _ = self.client(data)
            with self.assertRaises(gateway.SearchGatewayError):
                client.wikipedia_search("Moon")


class ToolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class BaseTool:
            def __init__(self, **kwargs):
                pass
        common = {
            "os": os, "requests": requests, "BaseTool": BaseTool,
            "SearchGatewayClient": gateway.SearchGatewayClient,
            "SearchGatewayError": gateway.SearchGatewayError,
            "search_cache_enabled": gateway.search_cache_enabled,
            "MAX_NETWORK_RETRIES": 3, "MAX_RETRY_WAIT_SECONDS": 30,
            "RETRYABLE_HTTP_STATUSES": {429, 500, 502, 503, 504},
            "YIBU_RETRYABLE_HTTP_STATUSES": {422, 429, 500, 502, 503, 504},
            "retry_wait_seconds": Mock(return_value=0), "time": NS(sleep=Mock()),
            "create_llm_engine": lambda **kwargs: NS(), "TOOL_NAME": "test_tool",
            "LIMITATION": "", "LIMITATIONS": "", "BEST_PRACTICE": "", "BEST_PRACTICES": "",
            "SUMMARIZE_PROMPT_TEMPLATE": "{query}\n{reference_information}",
        }
        for name in ("ToolExecutionError", "check_api_result", "current_context", "ensure_tool_context",
                     "owned_api_client", "run_bounded", "safe_failure"):
            common[name] = getattr(execution, name)
        from typing import Any, Dict, List, Optional
        cls.brave_scope = dict(common, Any=Any, Dict=Dict, List=List, Optional=Optional,
                               DEFAULT_ENDPOINT="https://upstream.test/brave")
        cls.Brave = extract(TOOLS / "brave_search/tool.py", "Brave_Search_Tool", cls.brave_scope)
        cls.web_scope = dict(common, BeautifulSoup=BeautifulSoup, urlsplit=__import__("urllib.parse", fromlist=["urlsplit"]).urlsplit)
        cls.Web = extract(TOOLS / "web_search/tool.py", "Web_Search_Tool", cls.web_scope)
        cls.RateLimit = extract(TOOLS / "wikipedia_search/tool.py", "WikipediaRateLimitError", {})
        cls.wiki_scope = dict(common, sys=NS(exit=Mock()), wikipedia=Mock(),
                              WikipediaRateLimitError=cls.RateLimit,
                              select_relevant_queries=Mock(return_value=(["B", "A"], [1, 0])),
                              Web_Search_Tool=Mock())
        cls.Wiki = extract(TOOLS / "wikipedia_search/tool.py", "Wikipedia_Search_Tool", cls.wiki_scope)

    def setUp(self):
        self.env = patch.dict(os.environ, {
            "SEARCH_CACHE_ENABLED": "1", "OPENAI_API_KEY": "test-key",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.wiki_scope["wikipedia"].reset_mock()
        self.wiki_scope["Web_Search_Tool"].reset_mock(side_effect=True)
        self.wiki_scope["select_relevant_queries"].reset_mock()

    def service(self, response=None, error=None):
        session = Session(response=response, error=error)
        client = gateway.SearchGatewayClient("https://development.test", "test-token", session=session)
        return patch.object(gateway.SearchGatewayClient, "from_env", return_value=client), session

    def test_yibu_initializes_without_training_machine_provider_key(self):
        tool = self.Brave()
        self.assertIsNone(tool.api_key)
        os.environ["SEARCH_CACHE_ENABLED"] = "0"
        with self.assertRaisesRegex(Exception, "API key not found"):
            self.Brave()

    def test_yibu_forwards_effective_parameters_and_keeps_formatted_multi_results(self):
        raw = {"web": {"results": [{"title": "B", "url": "https://b.test", "description": "snippet B"}, {"title": "A", "url": "https://a.test", "extra_snippets": ["extra A"]}]}}
        service, session = self.service(Response({"data": raw, "meta": {"cache": "hit"}}))
        with service, patch.object(requests, "get", side_effect=AssertionError("must not search directly")):
            result = self.Brave().execute("query", count=50, country="US", search_lang="en", ui_lang="en-US", freshness="pw")
        self.assertEqual(session.calls[0][2]["json"], {
            "query": "query", "count": 20, "country": "US", "search_lang": "en", "ui_lang": "en-US", "freshness": "pw",
        })
        self.assertLess(result.index("[1] B"), result.index("[2] A"))
        self.assertIn("Extra snippets: extra A", result)
        self.assertNotIn("meta", result)
        self.assertTrue(session.closed)

    def test_yibu_service_failure_does_not_call_provider_or_retry(self):
        service, session = self.service(error=requests.Timeout())
        with service, patch.object(requests, "get", side_effect=AssertionError("must not search directly")):
            result = self.Brave().execute("query")
        self.assertIn("gateway request failed", result)
        self.assertEqual(len(session.calls), 1)

    def test_disabled_yibu_keeps_direct_main_behavior_even_with_old_gateway_env(self):
        os.environ.update(SEARCH_CACHE_ENABLED="0", BRAVE_API_KEY="provider-key", SEARCH_GATEWAY_BASE_URL="http://old.test")
        with patch.object(requests, "get", return_value=Response({"web": {"results": [{"title": "direct"}]}})) as get:
            result = self.Brave().execute("query", count=2)
        self.assertIn("[1] direct", result)
        self.assertEqual(get.call_args.kwargs["params"], {"q": "query", "count": 2})
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer provider-key")

    def test_web_gateway_rewrites_arxiv_and_passes_size_limit(self):
        service, session = self.service(Response({"text": "body", "meta": {"cache": "hit"}}))
        tool = self.Web.__new__(self.Web)
        tool.max_window_size = 123
        with service:
            self.assertEqual(tool._get_website_content("https://arxiv.org/pdf/1234"), "body")
        self.assertEqual(session.calls[0][2]["json"], {"url": "https://arxiv.org/abs/1234", "max_length": 123})

    def test_disabled_web_keeps_direct_main_fetch_and_text_extraction(self):
        os.environ["SEARCH_CACHE_ENABLED"] = "0"
        session = MagicMock()
        session.__enter__.return_value = session
        session.get.return_value = Response()
        tool = self.Web.__new__(self.Web)
        tool.max_window_size = 100
        with patch.object(requests, "Session", return_value=session), patch.object(
            gateway.SearchGatewayClient, "from_env", side_effect=AssertionError("disabled")
        ):
            self.assertEqual(tool._get_website_content("https://page.test"), "Page body")
        self.assertEqual(session.get.call_args.args, ("https://page.test",))
        self.assertIs(session.get.call_args.kwargs["verify"], False)

    def test_enabled_missing_service_configuration_never_falls_back(self):
        with patch.object(requests, "get", side_effect=AssertionError("must not go direct")):
            result = self.Brave().execute("query")
        self.assertIn("gateway request failed", result)
        with self.assertRaises(gateway.SearchGatewayConfigurationError):
            self.Wiki.__new__(self.Wiki).search_wikipedia("query")
        self.wiki_scope["wikipedia"].search.assert_not_called()

    def test_web_failure_stops_before_local_embedding_and_summary(self):
        service, session = self.service(error=requests.ConnectionError())
        tool = self.Web.__new__(self.Web)
        tool.max_window_size = 100
        tool._embed_strings = Mock(side_effect=AssertionError("error must not be embedded"))
        tool._construct_final_output = Mock(side_effect=AssertionError("error must not be summarized"))
        with service:
            result = tool.execute("query", "https://page.test")
        self.assertTrue(result.startswith("Error fetching URL:"))
        tool._embed_strings.assert_not_called()
        tool._construct_final_output.assert_not_called()
        self.assertEqual(len(session.calls), 1)

    def test_web_local_rag_still_runs_with_gateway_body(self):
        service, _ = self.service(Response({"text": "cached page body"}))
        tool = self.Web.__new__(self.Web)
        tool.max_window_size, tool.top_k = 100, 1
        tool._chunk_website_content = Mock(return_value=["chunk"])
        tool._embed_strings = Mock(return_value=[[1], [2]])
        tool._rank_chunks = Mock(return_value=[0])
        tool._concatenate_chunks = Mock(return_value="reference")
        tool._construct_final_output = Mock(return_value="local summary")
        with service:
            result = tool.execute("current query", "https://page.test")
        self.assertEqual(result, "local summary")
        tool._embed_strings.assert_called_once_with(["current query", "chunk"])
        tool._construct_final_output.assert_called_once_with("current query", "reference")

    def test_embedding_api_error_is_visible_and_stops_before_summary(self):
        client = NS(close=Mock(), embeddings=NS(create=Mock(side_effect=openai.APIConnectionError(
            message="secret-api-key", request=httpx.Request("POST", "http://embedding.test")
        ))))
        constructor = Mock(return_value=client)
        tool = self.Web.__new__(self.Web)
        tool.embeddings_model = "test-model"
        tool._get_website_content = Mock(return_value="page")
        tool._chunk_website_content = Mock(return_value=["chunk"])
        tool._construct_final_output = Mock()
        with patch.dict(self.web_scope, {"openai": NS(OpenAI=constructor)}):
            result = tool.execute("query", "https://page.test")
        self.assertIn("api_connection_error", result)
        self.assertIn("phase=web.embedding", result)
        self.assertNotIn("secret", result)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        client.embeddings.create.assert_called_once()
        client.close.assert_called_once()
        tool._construct_final_output.assert_not_called()

    def test_summary_sdk_error_keeps_http_status_instead_of_none(self):
        error = openai.InternalServerError("secret-body", response=httpx.Response(
            503, request=httpx.Request("POST", "http://model.test")), body={"secret": "prompt"})
        tool = self.Web.__new__(self.Web)
        tool.top_k, tool.temperature, tool.llm_engine = 1, 0, Mock(side_effect=error)
        tool._get_website_content = Mock(return_value="page")
        tool._chunk_website_content = Mock(return_value=["chunk"])
        tool._embed_strings = Mock(return_value=[[1], [2]])
        tool._rank_chunks = Mock(return_value=[0])
        tool._concatenate_chunks = Mock(return_value="reference")
        result = tool.execute("query", "https://page.test")
        self.assertIn("api_http_error", result)
        self.assertIn("phase=web.summary", result)
        self.assertIn("HTTP=503", result)
        self.assertNotIn("secret", result)

    def test_wiki_propagates_embedding_failure_and_does_not_try_next_page(self):
        pages = [{"title": title, "url": f"https://wiki.test/{title}", "abstract": title} for title in ["A", "B"]]
        service, _ = self.service(Response({"results": pages}))
        rag = self.wiki_scope["Web_Search_Tool"].return_value
        rag.execute.side_effect = execution.ToolExecutionError(
            "embedding failed", code="api_connection_error", phase="web.embedding"
        )
        tool = self.Wiki.__new__(self.Wiki)
        tool.model_string, tool.llm_engine = "model", object()
        with service, self.assertRaises(execution.ToolExecutionError) as caught:
            tool.execute("query")
        self.assertEqual(caught.exception.phase, "web.embedding")
        self.assertEqual(rag.execute.call_count, 1)

    def test_wiki_selection_sdk_timeout_does_not_turn_into_empty_selection(self):
        scope = dict(self.wiki_scope, Select_Relevant_Queries=object)
        select = extract(TOOLS / "wikipedia_search/tool.py", "select_relevant_queries", scope)
        engine = NS(generate=Mock(side_effect=openai.APITimeoutError(
            request=httpx.Request("POST", "http://model.test"))))
        with self.assertRaises(execution.ToolExecutionError) as caught:
            select("question", ["A", "B"], engine)
        self.assertEqual(caught.exception.code, "api_timeout")
        self.assertEqual(caught.exception.phase, "wiki.select_pages")

    def test_wiki_other_page_exception_is_visible_and_does_not_become_none(self):
        pages = [{"title": title, "url": f"https://wiki.test/{title}", "abstract": title} for title in ["A", "B"]]
        service, _ = self.service(Response({"results": pages}))
        rag = self.wiki_scope["Web_Search_Tool"].return_value
        rag.execute.side_effect = [ValueError("secret-prompt"), "other page answer"]
        tool = self.Wiki.__new__(self.Wiki)
        tool.model_string, tool.llm_engine = "model", object()
        with service:
            result = tool.execute("query")
        first, second = result["relevant_pages (to the query)"]
        self.assertIn("type=ValueError", first["retrieved_information"])
        self.assertNotIn("secret", first["retrieved_information"])
        self.assertEqual(second["retrieved_information"], "other page answer")

    def test_disabled_wiki_keeps_library_search_and_original_truncation(self):
        os.environ["SEARCH_CACHE_ENABLED"] = "0"
        self.wiki_scope["wikipedia"].search.return_value = ["A", "B"]
        self.wiki_scope["wikipedia"].page.return_value = NS(content="abcdef", url="https://wiki.test/A")
        with patch.object(gateway.SearchGatewayClient, "from_env", side_effect=AssertionError("disabled")):
            result = self.Wiki.__new__(self.Wiki).search_wikipedia("query", max_pages=1, max_length=3)
        self.assertEqual(result, [{"title": "A", "url": "https://wiki.test/A", "abstract": "abc... [truncated]"}])
        self.wiki_scope["wikipedia"].search.assert_called_once_with("query")

    def test_wiki_gateway_keeps_selection_and_rag_on_training_machine(self):
        pages = [{"title": title, "url": f"https://wiki.test/{title}", "abstract": title} for title in ["A", "B", "C"]]
        service, _ = self.service(Response({"results": pages, "meta": {"cache": "hit"}}))
        rag = self.wiki_scope["Web_Search_Tool"].return_value
        rag.execute.side_effect = lambda query, url, **kwargs: f"{query}:{url}"
        tool = self.Wiki.__new__(self.Wiki)
        tool.model_string, tool.llm_engine = "local-model", object()
        with service:
            result = tool.execute("current query")
        self.assertEqual([p["title"] for p in result["relevant_pages (to the query)"]], ["B", "A"])
        self.assertEqual([p["title"] for p in result["other_pages (may be irrelevant to the query)"]], ["C"])
        self.assertEqual(rag.execute.call_count, 2)
        self.assertNotIn("retrieved_information", pages[0])
        self.assertNotIn("meta", result)
        self.wiki_scope["wikipedia"].search.assert_not_called()
        self.wiki_scope["select_relevant_queries"].assert_called_once_with("current query", ["A", "B", "C"], tool.llm_engine)

    def test_nested_web_returns_structured_failure_before_embedding_or_summary(self):
        service, session = self.service(Response({"error": {
            "code": "upstream_proxy_error", "message": "proxy failed", "upstream_status": 407,
        }}, status=502))
        tool = self.Web.__new__(self.Web)
        tool.max_window_size = 100
        tool._embed_strings = Mock(side_effect=AssertionError("must not embed error"))
        with service, self.assertRaises(gateway.SearchGatewayError) as caught:
            tool.execute("query", "https://page.test", raise_network_errors=True)
        self.assertEqual(caught.exception.upstream_status_code, 407)
        self.assertEqual(len(session.calls), 1)
        tool._embed_strings.assert_not_called()

    def test_wiki_stops_after_first_shared_proxy_failure_instead_of_next_page(self):
        pages = [{"title": title, "url": f"https://wiki.test/{title}", "abstract": title} for title in ["A", "B"]]
        service, _ = self.service(Response({"results": pages}))
        rag = self.wiki_scope["Web_Search_Tool"].return_value
        rag.execute.side_effect = gateway.SearchGatewayError(
            "proxy failed", code="upstream_proxy_error", upstream_status_code=407
        )
        tool = self.Wiki.__new__(self.Wiki)
        tool.model_string, tool.llm_engine = "local-model", object()
        with service, self.assertRaises(gateway.SearchGatewayError) as caught:
            tool.execute("query")
        self.assertEqual(caught.exception.code, "upstream_proxy_error")
        self.assertEqual(rag.execute.call_count, 1)
        self.assertTrue(rag.execute.call_args.kwargs["raise_network_errors"])

    def test_wiki_retains_single_page_error_and_tries_the_next_page(self):
        pages = [{"title": title, "url": f"https://wiki.test/{title}", "abstract": title} for title in ["A", "B"]]
        service, _ = self.service(Response({"results": pages}))
        rag = self.wiki_scope["Web_Search_Tool"].return_value
        rag.execute.side_effect = [gateway.SearchGatewayError(
            "网站死亡", code="domain_blocked", status_code=410
        ), "other page answer"]
        tool = self.Wiki.__new__(self.Wiki)
        tool.model_string, tool.llm_engine = "local-model", object()
        with service:
            result = tool.execute("query")
        selected = result["relevant_pages (to the query)"]
        self.assertIn("domain_blocked", selected[0]["retrieved_information"])
        self.assertEqual(selected[1]["retrieved_information"], "other page answer")
        self.assertEqual(rag.execute.call_count, 2)

    def test_wiki_gateway_failure_and_429_do_not_fallback_to_library(self):
        for response, error, exception in [
            (None, requests.Timeout(), gateway.SearchGatewayError),
            (Response({"error": {"code": "wikipedia_http_429", "message": "limited"}}, status=429), None, self.RateLimit),
        ]:
            service, session = self.service(response, error)
            with service, self.assertRaises(exception):
                self.Wiki.__new__(self.Wiki).search_wikipedia("Moon")
            self.assertEqual(len(session.calls), 1)
        self.wiki_scope["wikipedia"].search.assert_not_called()

    def test_wiki_global_ssl_patch_preserves_gateway_explicit_tls(self):
        scope = {"_original_session_request": Mock(return_value="sent")}
        patched = extract(TOOLS / "wikipedia_search/tool.py", "_patched_session_request", scope)
        patched(object(), "POST", "https://development.test", verify=True)
        self.assertIs(scope["_original_session_request"].call_args.kwargs["verify"], True)
        patched(object(), "GET", "https://external.test")
        self.assertIs(scope["_original_session_request"].call_args.kwargs["verify"], False)


if __name__ == "__main__":
    unittest.main()
