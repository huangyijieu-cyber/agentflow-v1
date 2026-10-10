"""Offline network-provider checks; no model, proxy or live API is required."""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.search_service.core import GatewayFailure
from agentflow.search_service.providers import execute, prepare


class Response:
    def __init__(self, data=None, content=b""):
        self.data = data
        self.content = content
        self.closed = False

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return copy.deepcopy(self.data)

    def close(self):
        self.closed = True


class Service:
    def __init__(self, responder):
        self.responder = responder
        self.requests = []
        self.entries = {}
        self.ttls = {}
        self.cooldowns = []

    def notify_cooldown(self, upstream, url, delay):
        self.cooldowns.append((upstream, url, delay))

    def request(self, upstream, method, url, **kwargs):
        self.requests.append((upstream, method, url, kwargs))
        return self.responder(upstream, method, url, kwargs)

    def cached(self, namespace, params, producer, ttl, persist=True):
        key = (namespace, json.dumps(params, sort_keys=True))
        if key not in self.entries:
            value = producer()
            lifetime = ttl(value) if callable(ttl) else ttl
            self.ttls[key] = lifetime
            if persist and lifetime > 0:
                self.entries[key] = copy.deepcopy(value)
            return copy.deepcopy(value)
        return copy.deepcopy(self.entries[key])


def wiki_response(kwargs, titles=("Second", "First"), pageids=None):
    params = kwargs["params"]
    if params.get("list") == "search":
        return Response({"query": {"search": [{"title": title} for title in titles]}})
    if params.get("prop") == "info|pageprops":
        title = params["titles"]
        pageid = (pageids or {}).get(title, {"Second": 2, "First": 1}.get(title, 9))
        return Response({"query": {"pages": {str(pageid): {
            "pageid": pageid, "title": title,
            "fullurl": f"https://en.wikipedia.org/wiki/{title}",
        }}}})
    pageid = params["pageids"]
    return Response({"query": {"pages": {str(pageid): {
        "pageid": pageid, "extract": f"content for {pageid}",
    }}}})


class PrepareTests(unittest.TestCase):
    def test_effective_counts_and_exact_queries(self):
        self.assertEqual(prepare("brave", {"query": " A ", "count": 90, "country": "US"}),
                         {"query": " A ", "count": 20, "country": "US"})
        self.assertEqual(prepare("wikipedia", {"query": "A"}),
                         {"query": "A", "max_pages": 10, "max_length": 256, "language": "en"})
        self.assertNotEqual(prepare("brave", {"query": " A "}), prepare("brave", {"query": "A"}))

    def test_fetch_preserves_url_query_and_existing_arxiv_rule(self):
        self.assertEqual(prepare("fetch", {"url": "https://arxiv.org/pdf/1?x=2&y=3"}),
                         {"url": "https://arxiv.org/abs/1?x=2&y=3", "max_length": 1000000})

    def test_invalid_requests(self):
        cases = [
            ("wikipedia", {"query": ""}),
            ("wikipedia", {"query": "A", "language": "en/../../"}),
            ("wikipedia", {"query": "A", "max_pages": -1}),
            ("brave", {"query": "A", "count": True}),
            ("brave", {"query": "A", "count": 1.5}),
            ("fetch", {"url": "file:///tmp/a"}),
            ("fetch", {"url": "https://user:password@example.com"}),
            ("fetch", {"url": "https://example.com:invalid"}),
            ("fetch", {"url": "https://example.com", "max_length": -1}),
            ("fetch", {"url": "https://example.com", "max_length": 1000001}),
            ("fetch", {"url": "https://example.com", "max_length": "invalid"}),
            ("fetch", {"url": "https://example.com", "max_length": True}),
            ("unknown", {}),
        ]
        for tool, params in cases:
            with self.subTest(tool=tool, params=params), self.assertRaises(GatewayFailure):
                prepare(tool, params)


class WikipediaTests(unittest.TestCase):
    def test_order_field_order_truncation_and_independent_objects(self):
        service = Service(lambda upstream, method, url, kwargs: wiki_response(kwargs))
        params = prepare("wikipedia", {"query": "A", "max_length": 7})
        result = execute("wikipedia", params, service)
        self.assertEqual([page["title"] for page in result["results"]], ["Second", "First"])
        self.assertEqual(list(result["results"][0]), ["title", "url", "abstract"])
        self.assertEqual(result["results"][0]["abstract"], "content... [truncated]")
        result["results"][0]["retrieved_information"] = "query-specific answer"
        repeated = execute("wikipedia", params, service)
        self.assertNotIn("retrieved_information", repeated["results"][0])
        self.assertEqual(len(service.requests), 5)
        self.assertTrue(all(request[0] == "wikipedia" for request in service.requests))
        extract_params = next(request[3]["params"] for request in service.requests if request[3]["params"].get("prop") == "extracts")
        self.assertIn("explaintext", extract_params)
        self.assertNotIn("exsectionformat", extract_params)  # wikipedia 1.4.0 default

    def test_page_cache_is_shared_across_queries_and_lengths(self):
        service = Service(lambda upstream, method, url, kwargs: wiki_response(kwargs))
        execute("wikipedia", prepare("wikipedia", {"query": "A", "max_pages": 1}), service)
        result = execute("wikipedia", prepare("wikipedia", {"query": "B", "max_pages": 1, "max_length": -1}), service)
        self.assertEqual(result["results"][0]["abstract"], "content for 2")
        self.assertEqual(len(service.requests), 4)  # two lists, one identity, one extract

    def test_confirmed_aliases_share_canonical_page_text(self):
        service = Service(lambda upstream, method, url, kwargs: wiki_response(kwargs, titles=("Alias A", "Alias B"), pageids={"Alias A": 7, "Alias B": 7}))
        result = execute("wikipedia", prepare("wikipedia", {"query": "A"}), service)
        self.assertEqual([page["title"] for page in result["results"]], ["Alias A", "Alias B"])
        extracts = [call for call in service.requests if call[3]["params"].get("prop") == "extracts"]
        self.assertEqual(len(extracts), 1)

    def test_empty_result_is_short_cached_with_original_placeholder(self):
        service = Service(lambda upstream, method, url, kwargs: wiki_response(kwargs, titles=()))
        params = prepare("wikipedia", {"query": "no hits"})
        result = execute("wikipedia", params, service)
        self.assertEqual(result, {"results": [{"title": None, "url": None, "abstract": None, "error": "No results found for query: no hits"}]})
        execute("wikipedia", params, service)
        self.assertEqual(len(service.requests), 1)
        self.assertEqual(list(service.ttls.values()), [300])

    def test_partial_failure_keeps_position_and_is_not_cached_as_success(self):
        attempts = {"failed": False}
        def respond(upstream, method, url, kwargs):
            if kwargs["params"].get("titles") == "Second" and not attempts["failed"]:
                attempts["failed"] = True
                return Response({"query": {"pages": {"-1": {"title": "Second", "missing": ""}}}})
            return wiki_response(kwargs)
        service = Service(respond)
        params = prepare("wikipedia", {"query": "A"})
        first = execute("wikipedia", params, service)
        self.assertTrue(first["_meta"]["partial"])
        self.assertEqual(first["results"][0]["abstract"], "Please use the URL to get the full text further if needed.")
        self.assertEqual(first["results"][1]["abstract"], "content for 1")
        second = execute("wikipedia", params, service)
        self.assertNotIn("_meta", second)
        self.assertEqual(second["results"][0]["abstract"], "content for 2")
        self.assertEqual(len(service.requests), 6)

    def test_rate_limit_stops_before_next_page(self):
        def respond(upstream, method, url, kwargs):
            if kwargs["params"].get("prop") == "info|pageprops":
                raise GatewayFailure("rate limited", code="rate_limited", status_code=429, retryable=True)
            return wiki_response(kwargs)
        service = Service(respond)
        with self.assertRaises(GatewayFailure) as raised:
            execute("wikipedia", prepare("wikipedia", {"query": "A"}), service)
        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(len(service.requests), 2)
        self.assertFalse(any(key[0] == "wikipedia_page_identity" for key in service.entries))

    def test_api_rate_limit_in_json_is_failure_not_page_placeholder(self):
        for code in ("ratelimited", "maxlag"):
            with self.subTest(code=code):
                service = Service(lambda *args: Response({"error": {"code": code}}))
                with self.assertRaises(GatewayFailure) as raised:
                    execute("wikipedia", prepare("wikipedia", {"query": "A"}), service)
                self.assertEqual(raised.exception.status_code, 429)
                self.assertEqual(raised.exception.upstream, "wikipedia")
                self.assertEqual(raised.exception.retry_after, 5.0)
                self.assertEqual(service.cooldowns, [("wikipedia", "https://en.wikipedia.org/w/api.php", 5.0)])
                self.assertEqual(len(service.requests), 1)
                self.assertFalse(service.entries)

    def test_regular_api_error_does_not_cool_down_wikipedia(self):
        service = Service(lambda *args: Response({"error": {"code": "badvalue"}}))
        with self.assertRaises(GatewayFailure) as raised:
            execute("wikipedia", prepare("wikipedia", {"query": "A"}), service)
        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(service.cooldowns, [])

    def test_disambiguation_page_preserves_original_fallback(self):
        def respond(upstream, method, url, kwargs):
            if kwargs["params"].get("prop") == "info|pageprops":
                return Response({"query": {"pages": {"1": {"pageid": 1, "pageprops": {"disambiguation": ""}}}}})
            return wiki_response(kwargs, titles=("Ambiguous Name",))
        service = Service(respond)
        result = execute("wikipedia", prepare("wikipedia", {"query": "A"}), service)
        self.assertEqual(result["results"][0]["url"], "https://en.wikipedia.org/wiki/Ambiguous_Name")
        self.assertTrue(result["_meta"]["partial"])
        self.assertEqual(len(service.requests), 2)


class BraveTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"YIBU_BRAVE_API_KEY": "offline-test-token", "BRAVE_API_KEY": "", "BRAVE_YIBU_BASE_URL": "https://provider.test/search"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_complete_ordered_response_and_full_parameter_keys(self):
        data = {"web": {"results": [{"title": "B", "url": "https://b", "extra_snippets": ["two", "one"]}, {"title": "A", "url": "https://a"}]}, "extra": {"provider": True}}
        service = Service(lambda *args: Response(data))
        params = prepare("brave", {"query": "A", "count": 2, "country": "US", "search_lang": "en", "ui_lang": "en-US", "freshness": "pw"})
        self.assertEqual(execute("brave", params, service), {"data": data})
        repeated = execute("brave", params, service)
        repeated["data"]["web"]["results"].reverse()
        self.assertEqual(execute("brave", params, service)["data"], data)
        self.assertEqual(len(service.requests), 1)
        changed = dict(params, country="CN")
        execute("brave", changed, service)
        self.assertEqual(len(service.requests), 2)
        self.assertEqual(service.requests[0][3]["params"], {"q": "A", "count": 2, "country": "US", "search_lang": "en", "ui_lang": "en-US", "freshness": "pw"})
        self.assertNotIn("offline-test-token", str(service.entries))
        self.assertEqual(set(service.ttls.values()), {900})

    def test_explicit_empty_is_short_cached(self):
        service = Service(lambda *args: Response({"organic_results": []}))
        params = prepare("brave", {"query": "A"})
        execute("brave", params, service)
        execute("brave", params, service)
        self.assertEqual(len(service.requests), 1)
        self.assertEqual(set(service.ttls.values()), {300})

    def test_unknown_or_error_json_is_not_cached(self):
        for data in ({"message": "not authorized"}, {"error": {"message": "error"}, "web": {"results": []}}, {"web": {"results": [None]}}, [1, 2]):
            with self.subTest(data=data):
                service = Service(lambda *args: Response(data))
                with self.assertRaises(GatewayFailure):
                    execute("brave", prepare("brave", {"query": "A"}), service)
                self.assertFalse(service.entries)

    def test_missing_key_does_not_call_network(self):
        service = Service(lambda *args: self.fail("unexpected HTTP request"))
        with patch.dict(os.environ, {"BRAVE_API_KEY": "", "YIBU_BRAVE_API_KEY": ""}):
            with self.assertRaises(GatewayFailure) as raised:
                execute("brave", prepare("brave", {"query": "A"}), service)
        self.assertEqual(raised.exception.code, "configuration_error")


class FetchTests(unittest.TestCase):
    def test_different_return_limits_share_full_text_cache(self):
        service = Service(lambda *args: Response(content=b"<p>abcdef</p>"))
        short = prepare("fetch", {"url": "https://example.org/page", "max_length": 2})
        long = prepare("fetch", {"url": "https://example.org/page", "max_length": 5})
        self.assertNotEqual(short, long)  # Distinct logical batch entries.
        self.assertEqual(execute("fetch", short, service), {"text": "ab"})
        self.assertEqual(execute("fetch", long, service), {"text": "abcde"})
        self.assertEqual(execute("fetch", dict(long, max_length=0), service), {"text": ""})
        self.assertEqual(len(service.requests), 1)  # Same raw network snapshot.

    def test_exact_current_extraction_and_url_key(self):
        service = Service(lambda *args: Response(content=b"<html><h1>A</h1><p>B <b>C</b></p></html>"))
        params = prepare("fetch", {"url": "https://example.org/page?x=1"})
        self.assertEqual(execute("fetch", params, service), {"text": "A\nB\nC"})
        execute("fetch", params, service)
        execute("fetch", prepare("fetch", {"url": "https://example.org/page?x=2"}), service)
        self.assertEqual(len(service.requests), 2)
        self.assertEqual(service.requests[0][0], "web:example.org")
        self.assertEqual(service.requests[0][3]["headers"]["Accept-Language"], "en-US,en;q=0.5")

    def test_wikipedia_html_uses_shared_wikipedia_budget(self):
        service = Service(lambda *args: Response(content=b"<p>Wiki</p>"))
        with patch.dict(os.environ, {"WIKIMEDIA_USER_AGENT": "OfflineTestBot/contact"}):
            execute("fetch", prepare("fetch", {"url": "https://en.wikipedia.org/wiki/A"}), service)
        self.assertEqual(service.requests[0][0], "wikipedia")
        self.assertEqual(service.requests[0][3]["headers"]["User-Agent"], "OfflineTestBot/contact")

    def test_text_length_matches_existing_million_character_limit(self):
        service = Service(lambda *args: Response(content=("<p>" + "x" * 1000005 + "</p>").encode()))
        result = execute("fetch", prepare("fetch", {"url": "https://example.org/page"}), service)
        self.assertEqual(len(result["text"]), 1000000)

    def test_network_failure_is_not_cached_as_error_text(self):
        attempts = [0]
        def respond(*args):
            attempts[0] += 1
            if attempts[0] == 1:
                raise GatewayFailure("network timeout", code="upstream_timeout", retryable=True)
            return Response(content=b"<p>actual text</p>")
        service = Service(respond)
        params = prepare("fetch", {"url": "https://example.org/page"})
        with self.assertRaises(GatewayFailure):
            execute("fetch", params, service)
        self.assertFalse(service.entries)
        self.assertEqual(execute("fetch", params, service), {"text": "actual text"})


if __name__ == "__main__":
    unittest.main()
