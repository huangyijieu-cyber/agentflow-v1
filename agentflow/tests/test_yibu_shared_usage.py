"""Yibu traffic accounting through the shared cache; no real network/tokenizer downloads."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agentflow"))


def load_tool():
    base = ModuleType("agentflow.tools.base")
    base.BaseTool = type("ToolBase", (), {"__init__": lambda self, **kwargs: None})
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    spec = importlib.util.spec_from_file_location(
        "shared_yibu_tool_under_test", ROOT / "agentflow/agentflow/tools/brave_search/tool.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"agentflow.tools.base": base, "dotenv": dotenv}):
        spec.loader.exec_module(module)
    return module


TOOL = load_tool()
GatewayClient = TOOL.SearchGatewayClient


class Tokenizer:
    name_or_path = "local/test-tokenizer"

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Traffic must exclude special tokens")
        return list(text)


def envelope(status=200, payload=None):
    response = requests.Response()
    response.status_code = status
    response.encoding = "utf-8"
    response._content = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    response._content_consumed = True
    return response


def event(call_id="flight-1", attempt=1, **kwargs):
    return {
        "call_id": call_id, "attempt": attempt, "input_text": "query",
        "started_at": "2026-10-16T10:00:00+08:00", "elapsed_s": 0.2,
        "http_status": 200, "status": "success", "response_body_text": '{"body":"原始结果"}',
        "error_type": None, **kwargs,
    }


DATA = {"web": {"results": [{"title": "标题", "url": "https://example.test/", "description": "摘要"}]}}


class SharedUsageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = patch.dict(os.environ, {
            "SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_BASE_URL": "http://cache.test",
            "SEARCH_CACHE_TOKEN": "test-not-a-secret", "BRAVE_API_KEY": "", "YIBU_BRAVE_API_KEY": "",
            "AGENTFLOW_YIBU_TOKEN_USAGE_DIR": self.directory.name, "AGENTFLOW_YIBU_TOKEN_USAGE": "1",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.print = patch("builtins.print")
        self.print.start()
        self.addCleanup(self.print.stop)
        self.tool = TOOL.Brave_Search_Tool()
        self.tool.token_usage.set_tokenizer(Tokenizer())
        self.session = SimpleNamespace(request=Mock(), close=Mock())
        self.gateway = GatewayClient("http://cache.test", "test-not-a-secret", session=self.session)
        self.from_env = patch.object(GatewayClient, "from_env", return_value=self.gateway)
        self.from_env.start()
        self.addCleanup(self.from_env.stop)
        self.direct = patch.object(TOOL.requests, "get", side_effect=AssertionError("No direct request"))
        self.direct.start()
        self.addCleanup(self.direct.stop)

    def records(self):
        path = Path(self.directory.name) / "yibu_token.log"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def success(self, events):
        return envelope(payload={"data": DATA, "meta": {"yibu_usage_events": events}})

    def test_cache_hit_records_logical_tokens_but_no_http_attempt(self):
        self.session.request.return_value = self.success([])
        result = self.tool.execute("query")
        record, = self.records()
        self.assertEqual(record["output_text"], result)
        self.assertEqual(record["input_tokens"], 5)
        self.assertEqual(record["output_tokens"], len(result))
        self.assertEqual(record["event_type"], "tool_call")
        self.assertFalse(record["http_attempt"])
        summary = TOOL.summarize([self.directory.name])
        self.assertEqual(summary["http_attempts"], 0)
        self.assertEqual(summary["tool_calls"], 1)
        self.assertEqual(summary["search_calls"], 1)

    def test_actual_events_count_query_body_and_exact_final_tool_output(self):
        events = [event(attempt=1, status="http_retry", http_status=429, response_body_text="limited"),
                  event(attempt=2)]
        self.session.request.return_value = self.success(events)
        result = self.tool.execute("query")
        first, last, logical = self.records()
        self.assertEqual([first["event_id"], last["event_id"]], ["flight-1:1", "flight-1:2"])
        self.assertEqual(first["output_tokens"], 0)
        self.assertEqual(first["response_body_tokens"], len("limited"))
        self.assertEqual(last["output_text"], result)
        self.assertEqual(last["response_body_tokens"], len(events[-1]["response_body_text"]))
        self.assertEqual(logical["event_type"], "tool_call")
        self.assertEqual(self.session.request.call_count, 1)

    def test_one_http_attempt_preserves_both_tokenizer_views(self):
        self.session.request.side_effect = [self.success([event()]), self.success([event()])]
        self.tool.execute("query")
        self.tool.token_usage.set_tokenizer(SimpleNamespace(
            name_or_path="local/other-tokenizer", encode=lambda text, **kw: [1],
        ))
        self.tool.execute("query")
        summary = TOOL.summarize([self.directory.name])
        self.assertEqual(summary["http_attempts"], 1)
        self.assertEqual(summary["upstream_calls"], 1)
        self.assertEqual(summary["duplicates_ignored"], 0)
        self.assertEqual(summary["tool_calls"], 2)
        actual = [g for g in summary["groups"] if g["event_type"] == "upstream_attempt"]
        logical = [g for g in summary["groups"] if g["event_type"] == "tool_call"]
        self.assertEqual(len(actual), 2)
        self.assertEqual(len(logical), 2)
        self.assertEqual(sorted(g["sums"]["input_tokens"] for g in actual), [1, 5])

    def test_failure_events_keep_unknown_body_null_and_do_not_leak_into_error(self):
        telemetry = event(status="network_error", http_status=None,
                          response_body_text=None, error_type="ReadTimeout")
        self.session.request.return_value = envelope(504, {
            "error": {"code": "upstream_timeout", "message": "Request timed out"},
            "yibu_usage_events": [telemetry],
        })
        result = self.tool.execute("query")
        self.assertIn("upstream_timeout", result)
        attempt, logical = self.records()
        self.assertIsNone(attempt["response_body_tokens"])
        self.assertIsNone(attempt["http_status"])
        self.assertEqual(attempt["error_type"], "ReadTimeout")
        self.assertEqual(attempt["output_text"], result)
        self.assertEqual(logical["status"], "gateway_error")
        self.assertNotIn("response_body_text", result)
        self.assertEqual(self.session.request.call_count, 1)

    def test_client_telemetry_resets_after_previous_request(self):
        self.session.request.side_effect = [self.success([event()]), self.success([])]
        self.gateway.brave_search("query")
        self.assertEqual(len(self.gateway.last_usage_events), 1)
        self.gateway.brave_search("query")
        self.assertEqual(self.gateway.last_usage_events, [])

    def test_client_accepts_service_alias_fields_without_changing_result(self):
        telemetry = event()
        telemetry.pop("elapsed_s")
        telemetry.pop("input_text")
        telemetry.pop("response_body_text")
        telemetry.update(query="query", raw_response_text="body", duration_ms=123,
                         finished_at="2026-10-16T10:00:00.123+08:00")
        self.session.request.return_value = self.success([telemetry])
        self.tool.execute("query")
        attempt, _ = self.records()
        self.assertEqual(attempt["response_body_tokens"], 4)
        self.assertEqual(attempt["elapsed_s"], 0.123)
        self.assertEqual(attempt["finished_at"], telemetry["finished_at"])

    def test_legacy_gateway_without_events_does_not_invent_upstream_attempts(self):
        self.session.request.return_value = envelope(payload=DATA)
        self.assertIn("Search results", self.tool.execute("query"))
        self.assertEqual(TOOL.summarize([self.directory.name])["http_attempts"], 0)

    def test_missing_tokenizer_records_null_without_network_download(self):
        self.tool.token_usage.set_tokenizer(None)
        self.session.request.return_value = self.success([event()])
        with patch.object(TOOL, "_load_tokenizer", return_value=None):
            self.assertIn("Search results", self.tool.execute("query"))
        for record in self.records():
            self.assertTrue(all(record[field] is None for field in TOOL.TOKEN_FIELDS))

    def test_logging_failure_does_not_change_search_result_or_issue_retry(self):
        self.session.request.return_value = self.success([event()])
        with patch.object(self.tool.token_usage, "record", side_effect=OSError("disk full")):
            result = self.tool.execute("query")
        self.assertIn("Search results", result)
        self.assertEqual(self.session.request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
