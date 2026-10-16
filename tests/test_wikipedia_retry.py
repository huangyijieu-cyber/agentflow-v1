"""Offline Wikipedia retries and Yibu token telemetry; no real requests or models."""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.util
import json
import multiprocessing
import os
import re
from pathlib import Path
import runpy
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[1]


def load_get(responses):
    tree = ast.parse((ROOT / 'agentflow/agentflow/tools/wikipedia_search/tool.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ('_patched_get', 'WikipediaRateLimitError')]
    ns = runpy.run_path(str(ROOT / 'agentflow/agentflow/tools/network_retry.py'))
    ns.update(_original_get=Mock(side_effect=responses),
              time=SimpleNamespace(sleep=Mock()),
              requests=SimpleNamespace(exceptions=SimpleNamespace(
                  ConnectionError=ConnectionError, Timeout=TimeoutError)), print=Mock())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<wiki-retry>', 'exec'), ns)
    return ns


def response(status, retry_after=None):
    return SimpleNamespace(status_code=status,
                           headers={} if retry_after is None else {'Retry-After': retry_after},
                           close=Mock(), raise_for_status=Mock())


class WikipediaRetryTests(unittest.TestCase):
    def test_long_retry_after_is_waited(self):
        for delay in ('30', '60', '120'):
            with self.subTest(delay=delay):
                success = response(200)
                ns = load_get([response(429, delay), success])
                self.assertIs(ns['_patched_get']('https://en.wikipedia.org/w/api.php'), success)
                ns['time'].sleep.assert_called_once_with(float(delay) + 0.5)
                self.assertEqual(ns['_original_get'].call_count, 2)

    def test_still_stops_after_three_retries(self):
        ns = load_get([response(429, '60') for _ in range(4)])
        with self.assertRaisesRegex(ns['WikipediaRateLimitError'], 'after 3 retries'):
            ns['_patched_get']('https://en.wikipedia.org/w/api.php')
        self.assertEqual(ns['_original_get'].call_count, 4)
        self.assertEqual(ns['time'].sleep.call_count, 3)
        self.assertEqual([c.args[0] for c in ns['time'].sleep.call_args_list], [60.5] * 3)

    def test_missing_header_uses_backoff(self):
        ns = load_get([response(429) for _ in range(3)] + [response(200)])
        ns['_patched_get']('https://en.wikipedia.org/w/api.php')
        for call, base in zip(ns['time'].sleep.call_args_list, (1, 2, 4)):
            self.assertGreaterEqual(call.args[0], base)
            self.assertLessEqual(call.args[0], base + 0.5)
        self.assertEqual(ns['time'].sleep.call_count, 3)


# Yibu tests share the real tool module and its built-in telemetry/summary helpers.
sys.path.insert(0, str(ROOT / "agentflow"))


class TestTokenizer:
    name_or_path = "test/character-tokenizer"

    def encode(self, text, *, add_special_tokens):
        if add_special_tokens:
            raise AssertionError("Text traffic must exclude special tokens")
        return list(text)


class ToolBase:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_tool():
    # Only stub optional framework imports; run the entire real tool module.
    base = ModuleType("agentflow.tools.base")
    base.BaseTool = ToolBase
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    spec = importlib.util.spec_from_file_location(
        "yibu_tool_under_test", ROOT / "agentflow/agentflow/tools/brave_search/tool.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, {"agentflow.tools.base": base, "dotenv": dotenv}):
        spec.loader.exec_module(module)
    return module


TOOL = load_tool()
usage = SUMMARY = TOOL


def yibu_response(status=200, data=None, text=None, headers=None):
    result = requests.Response()
    result.status_code = status
    result.url = "https://example.test/brave/v1/web/search"
    result.encoding = "utf-8"
    result._content = (text if text is not None else json.dumps(
        data or {"web": {"results": [{"title": "标题", "url": "https://example.test",
                                    "description": "搜索摘要"}]}},
        ensure_ascii=False,
    )).encode("utf-8")
    result._content_consumed = True
    result.headers.update(headers or {})
    return result


def write_process_records(directory, worker):
    recorder = usage.YibuTokenUsageRecorder(tokenizer=TestTokenizer(), log_dir=directory)
    with patch("builtins.print"):
        for index in range(4):
            recorder.record(
                call_id=f"process-{worker}-{index}", attempt=1, query="跨进程 query",
                output=f"worker={worker},index={index}\n" + "完整结果\n" * 10000,
                response=SimpleNamespace(text='{"body":"原始响应"}', status_code=200),
                status="success", elapsed_s=0.1,
            )


class YibuTokenUsageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        env = patch.dict(os.environ, {
            "BRAVE_API_KEY": "test-key-not-a-real-secret",
            "AGENTFLOW_YIBU_TOKEN_USAGE": "1",
            "AGENTFLOW_YIBU_TOKEN_USAGE_DIR": str(self.directory),
        })
        env.start()
        self.addCleanup(env.stop)
        get = patch.object(TOOL.requests, "get", side_effect=AssertionError("Unexpected request"))
        self.get = get.start()
        self.addCleanup(get.stop)
        sleep = patch.object(TOOL.time, "sleep")
        self.sleep = sleep.start()
        self.addCleanup(sleep.stop)
        output = patch("builtins.print")
        self.print = output.start()
        self.addCleanup(output.stop)
        usage._WARNED.clear()
        self.tokenizer = TestTokenizer()
        self.tool = TOOL.Brave_Search_Tool()
        self.tool.token_usage.set_tokenizer(self.tokenizer)

    def records(self):
        return [json.loads(line) for file in self.directory.glob("yibu_token.log")
                for line in file.read_text().splitlines()]

    def test_success_counts_query_formatted_result_and_raw_json_separately(self):
        raw = yibu_response()
        self.get.side_effect = [raw]
        result = self.tool.execute("中文 query", country="CN", count=1)
        expected = ("Search results for: 中文 query\n\n[1] 标题\n"
                    "URL: https://example.test\nDescription: 搜索摘要")
        self.assertEqual(result, expected)
        record, = self.records()
        self.assertEqual(record["input_tokens"], len("中文 query"))
        self.assertEqual(record["output_tokens"], len(expected))
        self.assertEqual(record["response_body_tokens"], len(raw.text))
        self.assertEqual(record["total_tokens"], len("中文 query") + len(expected))
        self.assertEqual(record["tokenizer"], self.tokenizer.name_or_path)
        self.assertEqual(record["status"], "success")
        self.assertNotIn("test-key-not-a-real-secret", json.dumps(record))
        self.assertEqual(record["input_text"], "中文 query")
        self.assertEqual(record["output_text"], expected)
        self.assertEqual(record["response_body_text"], raw.text)
        self.assertTrue(record["started_at"].endswith("+08:00"))
        self.assertTrue(record["finished_at"].endswith("+08:00"))
        self.assertLessEqual(record["started_at"], record["finished_at"])
        self.assertEqual(self.get.call_count, 1)

    def test_http_retry_keeps_each_attempt_and_counts_error_body(self):
        first = yibu_response(429, text='{"error":"限流"}')
        self.get.side_effect = [first, yibu_response()]
        result = self.tool.execute("test")
        self.assertIn("Search results for: test", result)
        records = self.records()
        self.assertEqual([r["attempt"] for r in records], [1, 2])
        self.assertEqual([r["status"] for r in records], ["http_retry", "success"])
        self.assertEqual(len({r["call_id"] for r in records}), 1)
        self.assertEqual(len({r["event_id"] for r in records}), 2)
        self.assertEqual(records[0]["response_body_tokens"], len(first.text))
        self.assertEqual(records[0]["output_tokens"], 0)
        self.sleep.assert_called_once()

    def test_timeout_is_unknown_response_not_zero(self):
        self.get.side_effect = [requests.exceptions.Timeout("timeout"), yibu_response()]
        self.tool.execute("query")
        records = self.records()
        self.assertEqual(records[0]["status"], "network_retry")
        self.assertIsNone(records[0]["response_body_tokens"])
        self.assertIsNone(records[0]["http_status"])
        self.assertEqual(records[0]["input_tokens"], 5)
        self.assertEqual(records[0]["output_tokens"], 0)
        self.assertEqual(self.get.call_count, 2)

    def test_four_failed_attempts_preserve_retry_limit_and_error_result(self):
        self.get.side_effect = [requests.exceptions.Timeout("timeout")] * 4
        result = self.tool.execute("query")
        self.assertIn("tried 4 times but failed", result)
        records = self.records()
        self.assertEqual(len(records), 4)
        self.assertEqual(records[-1]["status"], "network_error")
        self.assertEqual(records[-1]["output_text"], result)
        self.assertEqual(records[-1]["output_tokens"], len(result))
        self.assertEqual(self.sleep.call_count, 3)

    def test_nonretryable_http_error_is_recorded_once(self):
        self.get.side_effect = [yibu_response(401, text="Unauthorized")]
        result = self.tool.execute("query")
        self.assertIn("tried 1 time but failed", result)
        record, = self.records()
        self.assertEqual(record["status"], "http_error")
        self.assertEqual(record["http_status"], 401)
        self.assertEqual(record["response_body_tokens"], len("Unauthorized"))
        self.sleep.assert_not_called()

    def test_long_retry_after_still_stops_without_retry(self):
        self.get.side_effect = [yibu_response(429, text="limited", headers={"Retry-After": "120"})]
        self.tool.execute("query")
        record, = self.records()
        self.assertEqual(record["status"], "http_error")
        self.assertEqual(self.get.call_count, 1)
        self.sleep.assert_not_called()

    def test_bad_json_counts_body_and_preserves_failure(self):
        self.get.side_effect = [yibu_response(text="not json")]
        result = self.tool.execute("query")
        self.assertIn("but failed", result)
        record, = self.records()
        self.assertEqual(record["status"], "response_error")
        self.assertEqual(record["response_body_tokens"], 8)

    def test_no_results_message_is_counted_as_returned_text(self):
        self.get.side_effect = [yibu_response(data={"web": {"results": []}})]
        result = self.tool.execute("query")
        self.assertEqual(result, "No Brave search results found for query: query")
        record, = self.records()
        self.assertEqual(record["output_tokens"], len(result))

    def test_missing_tokenizer_records_null_without_guessing(self):
        self.tool.token_usage.set_tokenizer(None)
        self.get.side_effect = [yibu_response()]
        with patch.object(usage, "_load_tokenizer", return_value=None):
            self.assertIn("Search results", self.tool.execute("query"))
        record, = self.records()
        self.assertEqual(record["token_count_source"], "unavailable")
        for field in SUMMARY.TOKEN_FIELDS:
            self.assertIsNone(record[field])

    def test_standalone_tokenizer_loads_local_files_only_and_is_cached(self):
        self.tool.token_usage.set_tokenizer(None)
        transformers = ModuleType("transformers")
        loader = Mock(return_value=self.tokenizer)
        transformers.AutoTokenizer = SimpleNamespace(from_pretrained=loader)
        usage._load_tokenizer.cache_clear()
        self.addCleanup(usage._load_tokenizer.cache_clear)
        self.get.side_effect = [yibu_response(), yibu_response()]
        with patch.dict(sys.modules, {"transformers": transformers}), patch.dict(
            os.environ, {"AGENTFLOW_YIBU_TOKENIZER_PATH": "/local/test-tokenizer"},
        ):
            self.tool.execute("query")
            self.tool.execute("query")
        loader.assert_called_once_with(
            "/local/test-tokenizer", local_files_only=True, trust_remote_code=False,
        )
        self.assertTrue(all(r["input_tokens"] == 5 for r in self.records()))

    def test_broken_tokenizer_preserves_search_and_records_unknown_counts(self):
        self.tool.token_usage.set_tokenizer(SimpleNamespace(
            name_or_path="broken", encode=Mock(side_effect=ValueError("cannot encode")),
        ))
        self.get.side_effect = [yibu_response()]
        self.assertIn("Search results", self.tool.execute("query"))
        record, = self.records()
        self.assertEqual(record["tokenizer"], "broken")
        self.assertEqual(record["token_count_source"], "unavailable")
        self.assertIsNone(record["input_tokens"])

    def test_disk_failure_does_not_retry_or_discard_success(self):
        self.get.side_effect = [yibu_response()]
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            result = self.tool.execute("query")
        self.assertIn("Search results", result)
        self.assertEqual(self.get.call_count, 1)
        self.sleep.assert_not_called()

    def test_disabled_logging_skips_tokenizer(self):
        broken = SimpleNamespace(encode=Mock(side_effect=AssertionError("must not encode")))
        self.tool.token_usage.set_tokenizer(broken)
        self.get.side_effect = [yibu_response()]
        with patch.dict(os.environ, {"AGENTFLOW_YIBU_TOKEN_USAGE": "0"}):
            result = self.tool.execute("query")
        self.assertIn("Search results", result)
        broken.encode.assert_not_called()
        self.assertEqual(self.records(), [])

    def test_concurrent_writes_produce_64_complete_unique_records(self):
        def record(index):
            self.tool.token_usage.record(
                call_id=str(index), attempt=1, query="query", output="结果",
                response=yibu_response(), status="success", elapsed_s=0.1,
            )
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(record, range(64)))
        records = self.records()
        self.assertEqual(len(records), 64)
        self.assertEqual(len({r["event_id"] for r in records}), 64)
        self.assertTrue(all(r["output_tokens"] == 2 for r in records))

    def test_shared_log_path_matches_requested_training_directory(self):
        with patch.dict(os.environ, {
            "AGENTFLOW_YIBU_TOKEN_USAGE_DIR": "", "VC_TASK_INDEX": "0",
            "LOG_DIR_PREFIX": str(self.directory / "261014-143000_RL"),
        }):
            path = usage.YibuTokenUsageRecorder()._path()
        self.assertEqual(path, self.directory / "261014-143000_RL" / "task_logs" /
                         "127.0.0.1_rank0" / "yibu_token.log")

    def test_multiple_processes_preserve_long_original_texts_in_one_file(self):
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=write_process_records, args=(str(self.directory), worker))
                     for worker in range(4)]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join(timeout=15)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(len(list(self.directory.glob("yibu_token.log"))), 1)
            records = self.records()
            self.assertEqual(len(records), 16)
            self.assertEqual(len({r["event_id"] for r in records}), 16)
            self.assertTrue(all(r["output_text"].endswith("完整结果\n" * 10000)
                                for r in records))
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)

    def test_repeated_texts_reuse_counts_without_tokenizing_again(self):
        self.tokenizer.encode = Mock(wraps=self.tokenizer.encode)
        self.get.side_effect = [yibu_response(), yibu_response()]
        first_output = self.tool.execute("query")
        second_output = self.tool.execute("query")
        self.assertEqual(first_output, second_output)
        self.assertEqual(self.tokenizer.encode.call_count, 3)
        first, second = self.records()
        self.assertEqual(first["count_cache_misses"], 3)
        self.assertEqual(second["count_cache_hits"], 3)
        self.assertEqual(second["count_cache_misses"], 0)

    def test_retry_query_is_tokenized_once(self):
        self.tokenizer.encode = Mock(wraps=self.tokenizer.encode)
        self.get.side_effect = [requests.exceptions.Timeout("timeout"), yibu_response()]
        self.tool.execute("query")
        self.assertEqual(sum(call.args[0] == "query" for call in self.tokenizer.encode.call_args_list), 1)
        self.assertEqual(self.records()[1]["count_cache_hits"], 1)

    def test_switching_tokenizers_invalidates_previous_counts_even_with_same_name(self):
        self.get.side_effect = [yibu_response(), yibu_response()]
        self.tool.execute("query")
        self.tool.token_usage.set_tokenizer(SimpleNamespace(
            name_or_path=self.tokenizer.name_or_path, encode=lambda text, **kwargs: [1],
        ))
        self.tool.execute("query")
        self.assertEqual(self.records()[1]["input_tokens"], 1)

    def test_count_cache_is_bounded_and_does_not_keep_full_texts(self):
        with patch.object(usage, "_TOKEN_CACHE_SIZE", 4):
            for index in range(10):
                self.tool.token_usage._measure(f"query{index}", f"output{index}", f"raw{index}")
            cache = self.tool.token_usage._token_counts
            self.assertEqual(len(cache), 4)
            self.assertTrue(all(isinstance(key[1], bytes) and len(key[1]) == 32 for key in cache))

    def test_summary_deduplicates_events_but_keeps_retries_and_unknowns(self):
        self.get.side_effect = [requests.exceptions.Timeout("timeout"), yibu_response()]
        self.tool.execute("query")
        path, = self.directory.glob("yibu_token.log")
        duplicate = path.read_text().splitlines()[0]
        with path.open("a") as log:
            log.write(duplicate + "\n" + '{"truncated":\n')
        summary = SUMMARY.summarize([self.directory, path])
        self.assertEqual(summary["search_calls"], 1)
        self.assertEqual(summary["http_attempts"], 2)
        self.assertEqual(summary["duplicates_ignored"], 1)
        self.assertEqual(summary["malformed_lines"], 1)
        group, = summary["groups"]
        self.assertEqual(group["sums"]["input_tokens"], 10)
        self.assertEqual(group["unknown_counts"]["response_body_tokens"], 1)

    def test_summary_does_not_mix_tokenizers(self):
        self.get.side_effect = [yibu_response(), yibu_response()]
        self.tool.execute("query")
        self.tool.token_usage.set_tokenizer(SimpleNamespace(
            name_or_path="test/other", encode=lambda text, **kw: [1],
        ))
        self.tool.execute("query")
        summary = SUMMARY.summarize([self.directory])
        self.assertEqual(len(summary["groups"]), 2)
        self.assertEqual(sorted(g["sums"]["input_tokens"] for g in summary["groups"]), [1, 5])

    def test_summary_cli_needs_no_api_key_and_makes_no_search_request(self):
        self.get.side_effect = [yibu_response()]
        self.tool.execute("query")
        self.get.reset_mock()
        self.print.reset_mock()
        path = self.directory / "yibu_token.log"
        base = ModuleType("agentflow.tools.base")
        base.BaseTool = ToolBase
        dotenv = ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None
        source = ROOT / "agentflow/agentflow/tools/brave_search/tool.py"
        with patch.dict(sys.modules, {"agentflow.tools.base": base, "dotenv": dotenv}), \
                patch.dict(os.environ, {"BRAVE_API_KEY": "", "YIBU_BRAVE_API_KEY": ""}), \
                patch.object(sys, "argv", [str(source), "--summarize", str(path)]):
            runpy.run_path(str(source), run_name="__main__")
        self.get.assert_not_called()
        summary = json.loads(self.print.call_args.args[0])
        self.assertEqual(summary["search_calls"], 1)
        self.assertEqual(summary["http_attempts"], 1)

    def test_each_rollout_entrypoint_reuses_worker_tokenizer_before_search(self):
        for source in ("rollout.py",):
            with self.subTest(source=source):
                tree = ast.parse((ROOT / "train-roma" / source).read_text())
                cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                           and node.name == "RolloutAgent")
                method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef)
                              and node.name == "_solve_and_evaluate")
                # Exercise the actual InfoSeek reward/anchor helpers without
                # importing the training framework or loading a model.
                reward_helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                                  and node.name in {"_as_dict", "_training_reward",
                                                    "compute_search_subreward", "build_gigpo_anchors",
                                                    "build_turn_process_rewards"}]
                module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
                    ast.alias(name="annotations")], level=0), *reward_helpers, method], type_ignores=[])
                ns = {"asyncio": asyncio, "time": time, "datetime": datetime, "re": re,
                      "json": json, "evaluate": AsyncMock(return_value=1.0), "Rollout": SimpleNamespace}
                exec(compile(ast.fix_missing_locations(module), source, "exec"), ns)
                worker_tokenizer = TestTokenizer()
                self.tool.token_usage.set_tokenizer(None)
                self.get.side_effect = [yibu_response()]

                def solve(**kwargs):
                    self.assertIs(self.tool.token_usage.tokenizer, worker_tokenizer)
                    self.tool.execute("query")
                    return {"final_output": "<answer>answer</answer>", "planner_logs": []}, None

                rollout = SimpleNamespace(
                    solver=SimpleNamespace(executor=SimpleNamespace(
                        tool_instances_cache={"Yibu_Brave_Search_Tool": self.tool})),
                    llm_engine="test", solve=solve,
                )
                agent = SimpleNamespace(task="qa", tools=[], tokenizer=worker_tokenizer)
                result = asyncio.run(ns["_solve_and_evaluate"](
                    agent, "rollout-id", rollout, {"question": "q", "result": "answer"}, 1,
                ))
                self.assertEqual(result.final_reward, 1.0)
                self.assertEqual(self.records()[-1]["token_count_source"], "local-tokenizer")


class YibuSharedTokenUsageTests(unittest.TestCase):
    setUp = YibuTokenUsageTests.setUp
    records = YibuTokenUsageTests.records

    def event(self, attempt=1, status="success", body='{"web":{"results":[]}}'):
        return {
            "event_id": f"shared-call:{attempt}", "call_id": "shared-call", "attempt": attempt,
            "input_text": "query", "status": status, "response_body_text": body,
            "http_status": 200 if status == "success" else 503,
            "elapsed_s": 0.25, "started_at": "2026-10-16T08:00:00.000+08:00",
            "finished_at": "2026-10-16T08:00:00.250+08:00",
        }

    def shared_calls(self, responses, repeats=1):
        clients = []

        def from_env():
            session = Mock()
            session.request.return_value = responses[len(clients)]
            client = TOOL.SearchGatewayClient("http://127.0.0.1:8091", "test-token", session=session)
            clients.append(client)
            return client

        with patch.dict(os.environ, {"SEARCH_CACHE_ENABLED": "1"}), \
                patch.object(TOOL.SearchGatewayClient, "from_env", side_effect=from_env):
            results = [self.tool.execute("query") for _ in range(repeats)]
        self.get.assert_not_called()
        return results

    def envelope(self, events):
        return yibu_response(data={
            "data": {"web": {"results": [{"title": "cached title", "url": "https://example.test"}]}},
            "meta": {"yibu_usage_events": events},
        })

    def test_cache_hit_preserves_output_but_adds_no_http_attempt(self):
        results = self.shared_calls([self.envelope([self.event()]), self.envelope([])], repeats=2)
        self.assertEqual(results[0], results[1])
        records = self.records()
        self.assertEqual([r["event_type"] for r in records], ["upstream_attempt", "tool_call", "tool_call"])
        self.assertEqual(records[0]["response_body_tokens"], len(self.event()["response_body_text"]))
        self.assertEqual(records[1]["output_tokens"], len(results[0]))
        summary = SUMMARY.summarize([self.directory])
        self.assertEqual(summary["search_calls"], 2)
        self.assertEqual(summary["tool_calls"], 2)
        self.assertEqual(summary["http_attempts"], 1)

    def test_coalesced_attempt_deduplicates_across_training_logs(self):
        self.shared_calls([self.envelope([self.event()]), self.envelope([self.event()])], repeats=2)
        summary = SUMMARY.summarize([self.directory])
        self.assertEqual(summary["http_attempts"], 1)
        self.assertEqual(summary["upstream_calls"], 1)
        self.assertEqual(summary["tool_calls"], 2)
        self.assertEqual(summary["duplicates_ignored"], 1)

    def test_shared_retry_unknown_error_body_and_real_output(self):
        self.shared_calls([self.envelope([self.event(1, "http_error", None), self.event(2)])])
        first, last, logical = self.records()
        self.assertIsNone(first["response_body_tokens"])
        self.assertEqual(first["output_tokens"], 0)
        self.assertGreater(last["output_tokens"], 0)
        self.assertEqual(last["output_text"], logical["output_text"])
        self.assertEqual(SUMMARY.summarize([self.directory])["http_attempts"], 2)

    def test_shared_failure_events_are_logged_without_polluting_error_text(self):
        result, = self.shared_calls([yibu_response(502, data={
            "error": {"code": "upstream_http_error", "message": "Upstream returned HTTP 503"},
            "yibu_usage_events": [self.event(1, "http_error", "private body")],
        })])
        self.assertIn("upstream_http_error", result)
        self.assertNotIn("private body", result)
        upstream, logical = self.records()
        self.assertEqual(upstream["response_body_text"], "private body")
        self.assertEqual(logical["status"], "gateway_error")
        self.assertEqual(SUMMARY.summarize([self.directory])["http_attempts"], 1)

    def test_same_http_event_keeps_different_tokenizer_views(self):
        self.shared_calls([self.envelope([self.event()])])
        self.tool.token_usage.set_tokenizer(SimpleNamespace(
            name_or_path="test/other", encode=lambda text, **kw: [1],
        ))
        self.shared_calls([self.envelope([self.event()])])
        summary = SUMMARY.summarize([self.directory])
        self.assertEqual(summary["http_attempts"], 1)
        network = [group for group in summary["groups"] if group["event_type"] == "upstream_attempt"]
        self.assertEqual(len(network), 2)
        self.assertEqual(sorted(group["sums"]["input_tokens"] for group in network), [1, 5])


if __name__ == "__main__":
    unittest.main()
