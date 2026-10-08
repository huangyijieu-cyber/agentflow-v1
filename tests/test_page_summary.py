"""Offline behavior tests; no vLLM, embedding service or external sites needed.

Run: python3 -m unittest discover -s tests -p 'test_page_summary.py' -v
"""
import copy
import importlib
import os
from pathlib import Path
import re
import ssl
import sys
import types
import unittest
from unittest.mock import patch

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentflow"))
from agentflow.context_budget import ContextBudget, ContextBudgetError
from agentflow.models.memory import Memory, MEMORY_PLACEHOLDER
from agentflow.tools.page_summary import (
    PageSummarizer, extract_page_text, summary_cancel_event,
)
from agentflow.tools.wiki_summary import WikiResultSummarizer, WIKI_RESULT_TOKENS


class Budget:
    """Deterministic one-character tokens make boundary tests reproducible."""
    def __init__(self, limit=2600):
        self.limit = limit

    def count(self, text):
        return len(text)

    def fits(self, prompt, output_tokens):
        return len(prompt) + output_tokens <= self.limit

    def require(self, prompt, output_tokens):
        if not self.fits(prompt, output_tokens):
            raise ContextBudgetError("test context exceeded")


class EvidenceEngine:
    def __init__(self, budget=None):
        self.calls = []
        self.budget = budget or Budget()
        self._agentflow_context_budgets = {"AGENTFLOW_WEB": self.budget,
                                          "AGENTFLOW_AGENT": self.budget}

    def __call__(self, prompt, **kwargs):
        self.budget.require(prompt, kwargs.get("max_tokens", 2048))
        self.calls.append((prompt, kwargs))
        content = prompt.split("<source_material>\n", 1)[-1].split("\n</source_material>", 1)[0]
        facts = list(dict.fromkeys(re.findall(r"FACT_\d{3}", content)))
        stage = kwargs.get("usage_by", "").split()[-1]
        if stage == "extract":
            return (" ".join(facts) + " supporting detail" * 16) if facts else "NO_RELEVANT_EVIDENCE"
        return " ".join(facts) or "No supporting evidence found."


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "AGENTFLOW_WEB_SUMMARY_TOKENS": "400",
            "AGENTFLOW_WEB_EVIDENCE_TOKENS": "350",
            "AGENTFLOW_WEB_OVERLAP_TOKENS": "32",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_short_page_is_read_in_one_call(self):
        engine = EvidenceEngine()
        text = "Opening FACT_001\n\nEnd of page FACT_002"
        result = PageSummarizer(engine, engine.budget).summarize("query", "https://example.test", text)
        self.assertEqual(len(engine.calls), 1)
        self.assertIn(text, engine.calls[0][0])
        self.assertIn("FACT_002", result)
        self.assertIn("https://example.test", result)

    def test_long_page_all_segments_and_hierarchical_merge(self):
        engine = EvidenceEngine(Budget(1900))
        text = "\n\n".join(f"FACT_{i:03d} " + "x" * 220 for i in range(70))
        result = PageSummarizer(engine, engine.budget).summarize("query", "https://example.test", text)
        stages = [kwargs["usage_by"].split()[-1] for _, kwargs in engine.calls]
        self.assertIn("extract", stages)
        self.assertIn("merge", stages)
        self.assertEqual(stages[-1], "final")
        for i in range(70):
            self.assertIn(f"FACT_{i:03d}", result)
        for prompt, kwargs in engine.calls:
            self.assertTrue(engine.budget.fits(prompt, kwargs["max_tokens"]))

    def test_chinese_and_oversized_paragraph_have_no_missing_text(self):
        engine = EvidenceEngine(Budget(1800))
        reader = PageSummarizer(engine, engine.budget)
        text = "中文正文没有空格" * 1000 + "末尾证据"
        chunks = list(reader._chunks("query", "url", text, "extract", 350, overlap=False))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        overlap_chunks = list(reader._chunks("query", "url", text, "extract", 350))
        self.assertTrue(overlap_chunks[-1].endswith("末尾证据"))

    def test_html_cleanup_preserves_all_articles_and_table_rows(self):
        html = """<html><title>Title</title><body><nav>Menu</nav>
        <article><h1>Heading</h1><p>Alpha <b>Beta</b></p></article>
        <article><p>Second article evidence</p><table><caption>Results</caption>
        <tr><th>Year</th><th>Value</th></tr><tr><td>2025</td><td>42</td></tr></table></article>
        <script>bad script</script><footer>Footer</footer></body></html>"""
        text = extract_page_text(BeautifulSoup(html, "html.parser"))
        for value in ("Alpha Beta", "Second article evidence", "Year | Value", "2025 | 42"):
            self.assertIn(value, text)
        for value in ("Menu", "bad script", "Footer"):
            self.assertNotIn(value, text)

    def test_empty_page_and_irrelevant_long_page(self):
        engine = EvidenceEngine(Budget(1800))
        reader = PageSummarizer(engine, engine.budget)
        self.assertTrue(reader.summarize("q", "url", " ").startswith("Error"))
        self.assertEqual(engine.calls, [])
        result = reader.summarize("q", "url", "unrelated text " * 500)
        self.assertIn("No relevant evidence", result)
        self.assertNotIn("final", [kwargs["usage_by"].split()[-1] for _, kwargs in engine.calls])

    def test_empty_model_response_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "no text"):
            PageSummarizer(lambda *a, **k: "", Budget()).summarize("q", "url", "text")

    def test_nonshrinking_merge_fails_without_returning_partial_summary(self):
        class NonshrinkingEngine(EvidenceEngine):
            def __call__(self, prompt, **kwargs):
                if kwargs["usage_by"].endswith("merge"):
                    return prompt.split("<source_material>\n", 1)[1].split("\n</source_material>", 1)[0]
                return super().__call__(prompt, **kwargs)
        engine = NonshrinkingEngine(Budget(1900))
        text = "\n\n".join(f"FACT_{i:03d} " + "x" * 220 for i in range(70))
        with self.assertRaisesRegex(RuntimeError, "did not shrink"):
            PageSummarizer(engine, engine.budget).summarize("q", "url", text)

    def test_query_too_long_does_not_call_model(self):
        engine = EvidenceEngine(Budget(1800))
        with self.assertRaises(ContextBudgetError):
            PageSummarizer(engine, engine.budget).summarize("q" * 3000, "url", "text")
        self.assertEqual(engine.calls, [])

    def test_cancelled_work_does_not_start_another_model_call(self):
        import threading
        event = threading.Event()
        event.set()
        token = summary_cancel_event.set(event)
        try:
            engine = EvidenceEngine()
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                PageSummarizer(engine, engine.budget).summarize("q", "url", "text")
            self.assertEqual(engine.calls, [])
        finally:
            summary_cancel_event.reset(token)


class WikiSummaryTests(unittest.TestCase):
    def pages(self):
        return [{"title": f"Page {i}", "url": f"https://en.wikipedia.org/wiki/Page_{i}",
                 "retrieved_information": f"FACT_{i:03d} " + "detail " * 400}
                for i in range(1, 4)]

    def test_combines_all_pages_and_preserves_sources_in_memory(self):
        engine = EvidenceEngine(Budget(32768))
        reader = WikiResultSummarizer(engine)
        result = reader.summarize_pages("query", self.pages(), [])
        self.assertEqual(len(engine.calls), 1)
        for i in range(1, 4):
            self.assertIn(f"FACT_{i:03d}", result["summary"])
            self.assertEqual(result["sources"][i - 1]["id"], i)
        self.assertLessEqual(reader._size(result), WIKI_RESULT_TOKENS)
        self.assertLess(engine.calls[0][1]["max_tokens"], WIKI_RESULT_TOKENS)
        memory = Memory()
        memory.add_action(1, "Wikipedia_RAG_Search_Tool", "goal", "cmd", [result])
        visible = memory.get_actions()["Action Step 1"]["result"]
        self.assertEqual(visible, [result])

    def test_oversized_merge_input_considers_last_page(self):
        engine = EvidenceEngine(Budget(5000))
        reader = WikiResultSummarizer(engine)
        result = reader.summarize_pages("query", self.pages(), [])
        self.assertGreater(len(engine.calls), 1)
        self.assertIn("FACT_003", result["summary"])
        self.assertLessEqual(reader._size(result), WIKI_RESULT_TOKENS)

    def test_oversized_generated_summary_is_recompressed_once(self):
        class Engine(EvidenceEngine):
            def __call__(self, prompt, **kwargs):
                super().__call__(prompt, **kwargs)
                return "escaped \\\"中文\n" * 500 if len(self.calls) == 1 else "FACT_003"
        engine = Engine(Budget(32768))
        reader = WikiResultSummarizer(engine)
        result = reader.summarize_pages("query", self.pages(), [])
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(result["summary"], "FACT_003")
        self.assertLessEqual(reader._size(result), WIKI_RESULT_TOKENS)

    def test_backend_ignoring_limit_returns_small_error_without_partial_evidence(self):
        class Engine(EvidenceEngine):
            def __call__(self, prompt, **kwargs):
                super().__call__(prompt, **kwargs)
                return "x" * 3000
        engine = Engine(Budget(32768))
        reader = WikiResultSummarizer(engine)
        result = reader.summarize_pages("query", self.pages(), [])
        self.assertEqual(len(engine.calls), 2)
        self.assertIn("no partial summary", result["error"])
        self.assertLessEqual(reader._size(result), WIKI_RESULT_TOKENS)

    def test_empty_and_unselected_results_do_not_call_model(self):
        engine = EvidenceEngine(Budget(32768))
        reader = WikiResultSummarizer(engine)
        result = reader.summarize_pages("query", [], [{"error": "No results found for query: query"}])
        self.assertEqual(result["summary"], "No results found for query: query")
        result = reader.summarize_pages("query", [], self.pages())
        self.assertIn("Candidates only", result["summary"])
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(engine.calls, [])

    def test_large_metadata_or_query_returns_bounded_diagnostic(self):
        engine = EvidenceEngine(Budget(32768))
        reader = WikiResultSummarizer(engine)
        pages = self.pages()
        pages[0]["url"] += "中文" * 3000
        for query, selected, candidates in [("q", pages, []), ("q", [], pages),
                                            ("q" * 5000, [], [])]:
            result = reader.summarize_pages(query, selected, candidates)
            self.assertIn("error", result)
            self.assertLessEqual(reader._size(result), WIKI_RESULT_TOKENS)
        self.assertEqual(engine.calls, [])

    def test_merge_failure_returns_error_and_cancellation_still_propagates(self):
        import threading
        engine = EvidenceEngine(Budget(32768))
        reader = WikiResultSummarizer(engine)
        with patch.object(reader, "summarize", side_effect=RuntimeError("offline")):
            result = reader.summarize_pages("q", self.pages(), [])
        self.assertIn("failed", result["error"])
        self.assertLessEqual(reader._size(result), WIKI_RESULT_TOKENS)
        event = threading.Event()
        event.set()
        token = summary_cancel_event.set(event)
        try:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                reader.summarize_pages("q", self.pages(), [])
        finally:
            summary_cancel_event.reset(token)


class ContextTests(unittest.TestCase):
    def engine(self):
        return types.SimpleNamespace(model_string="qwen", base_url="http://server/v1",
                                     system_prompt="system", api_key="test")

    def test_serving_limit_caps_configuration_and_tokenizer_counts(self):
        def response(method, url, **kwargs):
            if url.endswith("/models"):
                return {"data": [{"id": "qwen", "max_model_len": 4096}]}
            self.assertEqual(url, "http://server/tokenize")
            self.assertEqual(kwargs["json"]["model"], "qwen")
            return {"count": 17}
        with patch.dict(os.environ, {"AGENTFLOW_WEB_CONTEXT_TOKENS": "8192"}), \
                patch.object(ContextBudget, "_json_request", side_effect=response):
            budget = ContextBudget(self.engine())
            self.assertEqual(budget.limit, 4096)
            self.assertEqual(budget.count("中文"), 17)
            self.assertTrue(budget.fits("prompt", 2048))
            self.assertFalse(budget.fits("prompt", 4096))

    def test_unavailable_endpoints_use_byte_bound_and_stop_retrying_tokenizer(self):
        with patch.dict(os.environ, {"AGENTFLOW_WEB_CONTEXT_TOKENS": "6000"}), \
                patch("agentflow.context_budget._load_local_tokenizer", return_value=None), \
                patch.object(ContextBudget, "_json_request", side_effect=requests.ConnectionError) as http:
            budget = ContextBudget(self.engine())
            self.assertEqual(budget.limit, 6000)
            self.assertEqual(budget.count("中文"), 6)
            self.assertEqual(budget.count("another"), 7)
            self.assertEqual(http.call_count, 3)  # Discovery once, two tokenizer routes once each.

    def test_versioned_tokenizer_route_is_remembered(self):
        def response(method, url, **kwargs):
            if url.endswith("/models"):
                return {"data": [{"id": "qwen", "max_model_len": 32768}]}
            if url == "http://server/tokenize":
                raise requests.HTTPError("404")
            self.assertEqual(url, "http://server/v1/tokenize")
            return {"count": 10}
        with patch.object(ContextBudget, "_json_request", side_effect=response) as http:
            budget = ContextBudget(self.engine())
            self.assertEqual(budget.count("first"), 10)
            self.assertEqual(budget.count("second"), 10)
            self.assertEqual(http.call_count, 4)
            self.assertEqual(budget.limit, 32768)

    def test_single_model_alias_does_not_hide_serving_limit(self):
        with patch.object(ContextBudget, "_json_request", return_value={
                "data": [{"id": "/models/qwen", "max_model_len": 32768}]}):
            self.assertEqual(ContextBudget(self.engine()).limit, 32768)

    def test_unavailable_endpoints_use_exact_local_tokenizer(self):
        tokenizer = types.SimpleNamespace(encode=lambda text, **kwargs: text.split())
        with patch.object(ContextBudget, "_json_request", side_effect=requests.ConnectionError), \
                patch("agentflow.context_budget._load_local_tokenizer", return_value=tokenizer):
            budget = ContextBudget(self.engine(), "AGENTFLOW_AGENT")
            prompt = "information " * 1000  # 12000 bytes, only 1000 toy tokens.
            self.assertTrue(budget.fits(prompt, 2048))
            self.assertEqual(budget.tokenizer_mode, "local-tokenizer")

    def test_byte_estimate_alone_does_not_abort_an_agent_rollout(self):
        with patch.object(ContextBudget, "_json_request", side_effect=requests.ConnectionError), \
                patch("agentflow.context_budget._load_local_tokenizer", return_value=None):
            budget = ContextBudget(self.engine(), "AGENTFLOW_AGENT")
            memory = Memory()
            result = "information " * 1000 + "TAIL"
            memory.add_action(1, "Web_RAG_Search_Tool", "goal", "command", [result])
            prompt = memory.render_prompt(MEMORY_PLACEHOLDER, None, budget=budget)
            self.assertIn("[truncated]", prompt)
            self.assertNotIn("TAIL", prompt)
            self.assertEqual(memory.get_all_actions()["Action Step 1"]["result"], [result])
            self.assertEqual(budget.tokenizer_mode, "utf8-upper-bound")


class MemoryTests(unittest.TestCase):
    def test_summary_tail_survives_prompt_and_raw_record_is_unchanged(self):
        memory = Memory()
        summary = "Evidence " * 100 + "TAIL_FACT"
        memory.add_action(1, "Web_RAG_Search_Tool", "goal", "command", [summary])
        raw = copy.deepcopy(memory.get_all_actions())
        prompt = memory.render_prompt("Question\n" + MEMORY_PLACEHOLDER, None,
                                      output_tokens=100, budget=Budget(10000))
        self.assertIn(summary, prompt)
        self.assertNotIn("[truncated]", prompt)
        self.assertEqual(memory.get_all_actions(), raw)

    def test_wiki_keeps_each_summary_and_url_without_candidate_noise(self):
        memory = Memory()
        result = {"query": "query", "relevant_pages (to the query)": [
            {"title": "One", "url": "url1", "abstract": "REDUNDANT",
             "retrieved_information": "A" * 250 + "TAIL_ONE"},
            {"title": "Two", "url": "url2", "abstract": "REDUNDANT",
             "retrieved_information": "B" * 250 + "TAIL_TWO"},
        ], "other_pages (may be irrelevant to the query)": [{"title": "OTHER"}]}
        memory.add_action(1, "Wikipedia_RAG_Search_Tool", "goal", "cmd", [result])
        raw = copy.deepcopy(memory.get_all_actions())
        prompt = memory.render_prompt(MEMORY_PLACEHOLDER, None, output_tokens=100, budget=Budget(10000))
        for value in ("TAIL_ONE", "TAIL_TWO", "url1", "url2"):
            self.assertIn(value, prompt)
        self.assertNotIn("REDUNDANT", prompt)
        self.assertNotIn("OTHER", prompt)
        self.assertEqual(memory.get_all_actions(), raw)

    def test_budget_discards_older_evidence_before_latest(self):
        memory = Memory()
        memory.add_action(1, "Web_RAG_Search_Tool", "old", "cmd", "OLD" * 1000)
        memory.add_action(2, "Web_RAG_Search_Tool", "new", "cmd", "LATEST" * 300)
        raw = copy.deepcopy(memory.get_all_actions())
        template = "Instructions " * 30 + MEMORY_PLACEHOLDER
        prompt = memory.render_prompt(template, None, output_tokens=300, budget=Budget(3000))
        self.assertIn("LATEST" * 300, prompt)
        self.assertNotIn("OLD" * 1000, prompt)
        self.assertTrue(Budget(3000).fits(prompt, 300))
        self.assertEqual(memory.get_all_actions(), raw)

    def test_latest_too_large_is_shortened_to_whole_prompt_budget(self):
        memory = Memory()
        memory.add_action(1, "Web_RAG_Search_Tool", "goal", "cmd", "x" * 5000)
        raw = copy.deepcopy(memory.get_all_actions())
        budget = Budget(3000)
        prompt = memory.render_prompt(MEMORY_PLACEHOLDER, None, budget=budget)
        self.assertIn("[truncated to fit context budget]", prompt)
        self.assertIn("x" * 100, prompt)
        self.assertTrue(budget.fits(prompt, 2048))
        self.assertEqual(memory.get_all_actions(), raw)

    def test_default_cap_covers_all_tools_and_preserves_raw_results(self):
        for tool in ("Web_RAG_Search_Tool", "Wikipedia_RAG_Search_Tool", "Yibu_Brave_Search_Tool"):
            with self.subTest(tool=tool):
                memory = Memory()
                result = [{"summary": "证据\\\"\n" * 1000, "sources": ["url"]}]
                memory.add_action(1, tool, "goal", "cmd", result)
                raw = copy.deepcopy(memory.get_all_actions())
                visible = memory.get_actions()["Action Step 1"]["result"]
                self.assertEqual(visible, str(result)[:2000] + "...[truncated]")
                self.assertEqual(memory.get_actions(max_result_chars=None)["Action Step 1"]["result"], result)
                self.assertEqual(memory.get_all_actions(), raw)

    def test_reported_overflow_sizes_fit_after_latest_result_shortening(self):
        class ServingBudget(Budget):
            margin = 256
            def fits(self, prompt, output_tokens):
                return self.count(prompt) + output_tokens + self.margin <= self.limit
        for input_size in (8706, 8986):
            with self.subTest(input_size=input_size):
                memory = Memory()
                memory.add_action(1, "Web_RAG_Search_Tool", "goal", "cmd", "证据" * 900)
                raw = copy.deepcopy(memory.get_all_actions())
                template = "i" * (input_size - len(str(memory.get_actions()))) + MEMORY_PLACEHOLDER
                budget = ServingBudget(10752)
                original = template.replace(MEMORY_PLACEHOLDER, str(memory.get_actions()))
                self.assertEqual(budget.count(original), input_size)
                self.assertFalse(budget.fits(original, 2048))
                prompt = memory.render_prompt(template, None, budget=budget)
                self.assertTrue(budget.fits(prompt, 2048))
                self.assertIn("truncated to fit context budget", prompt)
                self.assertEqual(memory.get_all_actions(), raw)

    def test_fixed_prompt_too_large_still_reports_configuration_problem(self):
        memory = Memory()
        memory.add_action(1, "tool", "goal", "cmd", "evidence")
        with self.assertRaisesRegex(ContextBudgetError, "question/instruction/action-metadata"):
            memory.render_prompt("fixed" * 1000 + MEMORY_PLACEHOLDER, None, budget=Budget(3000))

    def test_unknown_tokenizer_does_not_reject_oversized_fixed_byte_estimate(self):
        budget = Budget(3000)
        budget.tokenizer_mode = "utf8-upper-bound"
        memory = Memory()
        memory.add_action(1, "tool", "goal", "cmd", "x" * 5000)
        prompt = memory.render_prompt("fixed" * 1000 + MEMORY_PLACEHOLDER, None, budget=budget)
        self.assertIn("[truncated]", prompt)
        self.assertNotIn("x" * 2001, prompt)

    def test_overwritten_step_is_the_latest_action(self):
        memory = Memory()
        for i in range(1, 6):
            memory.add_action(i, "tool", "goal", "cmd", "old")
        memory.add_action(1, "tool", "goal", "cmd", "new")
        self.assertEqual(list(memory.get_actions())[-1], "Action Step 1")
        self.assertEqual(memory.get_actions()["Action Step 1"]["result"], "new")


class ToolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Stub only absent SDK imports; exercise the real Web/Wiki tool methods.
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None
        sdk = types.ModuleType("agentflow.engine.openai")
        sdk.ChatOpenAI = object
        wiki = types.ModuleType("wikipedia")
        cls.imports = patch.dict(sys.modules, {"dotenv": dotenv,
                                              "agentflow.engine.openai": sdk,
                                              "wikipedia": wiki})
        cls.imports.start()
        cls.web = importlib.import_module("agentflow.tools.web_search.tool")
        # Wikipedia's existing import patches global request functions; restore
        # those after import so offline tests do not change the runner's network state.
        with patch.object(requests.api, "request", requests.api.request), \
                patch.object(requests.api, "get", requests.api.get), \
                patch.object(requests.api, "post", requests.api.post), \
                patch.object(requests.sessions.Session, "request", requests.sessions.Session.request), \
                patch.object(requests, "get", requests.get), \
                patch.object(requests, "post", requests.post), \
                patch.object(ssl, "_create_default_https_context", ssl._create_default_https_context):
            cls.wiki = importlib.import_module("agentflow.tools.wikipedia_search.tool")

    @classmethod
    def tearDownClass(cls):
        cls.imports.stop()

    def test_web_fetch_to_summary_without_embedding(self):
        engine = EvidenceEngine(Budget(10000))
        tool = self.web.Web_Search_Tool(model_string="test", llm_engine=engine)
        with patch.object(tool, "_get_website_content", return_value="Body FACT_009"):
            result = tool.execute("query", "https://example.test")
        self.assertIn("FACT_009", result)
        self.assertEqual(len(engine.calls), 1)

    def test_wiki_selected_urls_use_same_full_page_reader(self):
        engine = EvidenceEngine(Budget(10000))
        with patch.object(self.wiki, "create_llm_engine", return_value=engine):
            tool = self.wiki.Wikipedia_Search_Tool(model_string="test")
        pages = [{"title": "Page 1", "url": "url1", "abstract": "short abstract"},
                 {"title": "Page 2", "url": "url2", "abstract": "short abstract"}]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), \
                patch.object(tool, "search_wikipedia", return_value=pages), \
                patch.object(self.wiki, "select_relevant_queries", return_value=(["Page 1", "Page 2"], [0, 1])), \
                patch.object(self.web.Web_Search_Tool, "_get_website_content",
                             side_effect=["Full page FACT_011", "Full page FACT_012"]):
            result = tool.execute("query")
        self.assertIn("FACT_011", result["summary"])
        self.assertIn("FACT_012", result["summary"])
        self.assertEqual([source["url"] for source in result["sources"]], ["url1", "url2"])
        self.assertEqual(len(engine.calls), 3)  # Two page summaries + one combined summary.
        self.assertLessEqual(WikiResultSummarizer(engine)._size(result), WIKI_RESULT_TOKENS)

    def test_planner_and_verifier_receive_capped_summary_and_raw_tail_is_kept(self):
        planner_module = importlib.import_module("agentflow.models.planner")
        verifier_module = importlib.import_module("agentflow.models.verifier")
        engine = EvidenceEngine(Budget(20000))
        with patch.object(planner_module, "create_llm_engine", return_value=engine), \
                patch.object(verifier_module, "create_llm_engine", return_value=engine):
            planner = planner_module.Planner("test", "test")
            verifier = verifier_module.Verifier("test", "test")
        memory = Memory()
        summary = "Relevant information. " * 150 + "FACT_999"
        memory.add_action(1, "Web_RAG_Search_Tool", "goal", "cmd", [summary])
        planner.query_analysis = "Initial analysis"
        trace = {}
        planner.generate_next_step("q", None, "analysis", memory, 2, 5, trace)
        planner.generate_final_output("q", None, memory)
        planner.generate_direct_output("q", None, memory)
        verifier.verificate_context("q", None, "analysis", memory, 1, trace)
        self.assertEqual(len(engine.calls), 4)
        for prompt, _ in engine.calls:
            self.assertIn(summary[:1000], prompt)
            self.assertIn("[truncated]", prompt)
            self.assertNotIn("FACT_999", prompt)
            self.assertNotIn(MEMORY_PLACEHOLDER, prompt)
        self.assertIn("[truncated]", trace["action_predictor_2_prompt"])
        self.assertIn("[truncated]", trace["verifier_1_prompt"])
        self.assertEqual(memory.get_all_actions()["Action Step 1"]["result"], [summary])


if __name__ == "__main__":
    unittest.main()
