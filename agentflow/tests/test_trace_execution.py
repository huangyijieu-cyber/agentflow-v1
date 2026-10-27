"""CPU regressions for external failures and trusted verifier outcomes."""

import ast
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "trace_execution_test", ROOT / "agentflow/agentflow/trace/execution.py")
EXECUTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXECUTION)
REWARD_SPEC = importlib.util.spec_from_file_location("trace_reward_test", ROOT / "agentflow/reward.py")
REWARD = importlib.util.module_from_spec(REWARD_SPEC)
REWARD_SPEC.loader.exec_module(REWARD)


def extract_function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    node = next(item for item in ast.walk(tree)
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def log(kind, response="answer", **kwargs):
    return {"trace_turn_type": kind, "response": response, "finish_reason": "stop", **kwargs}


class ExecutionClassificationTests(unittest.TestCase):
    def test_known_adapter_and_structured_external_errors(self):
        for result in (
            "Google Search tried 3 times but failed. Last error: ConnectionError",
            "Error generating response: APITimeoutError",
            "Yibu Brave Search gateway request failed. HTTP 503",
            {"error": "upstream unavailable"},
            {"success": False, "status_code": 429, "message": "quota exceeded"},
            {"relevant_pages (to the query)": [{"retrieved_information": "Error fetching URL: timed out"}]},
            [{"error": "Shared search request failed (api_proxy_error, phase=web.fetch, HTTP=502)"}],
        ):
            with self.subTest(result=result):
                status = EXECUTION.classify_tool_result(result)
                self.assertTrue(status["failed"])
                self.assertTrue(status["external_error"])
                self.assertEqual(status["error_source"], "external")

    def test_normal_empty_search_and_source_prose_are_not_external_failures(self):
        for result in (
            [], {"results": []}, {"error": "No results found for query: q"},
            "No search results", "A page describes timeout and ConnectionError.",
            {"content": "Error: ConnectionError is quoted in this article"},
            {"relevant_pages (to the query)": [{"retrieved_information": "Article: API timeouts can occur."}]},
        ):
            with self.subTest(result=result):
                self.assertFalse(EXECUTION.classify_tool_result(result)["failed"])

    def test_model_errors_and_ambiguous_executor_timeout_stay_separate(self):
        for result in (
            "Error in execute_tool_command: SyntaxError",
            "No result was generated because the tool was not found.",
            "Execution timed out after 3 seconds (phase=tool.execute)",
            "Error: invalid parameter: expected 500 characters",
        ):
            status = EXECUTION.classify_tool_result(result)
            self.assertTrue(status["failed"])
            self.assertFalse(status["external_error"])

    def test_successful_retry_does_not_inherit_attempt_error(self):
        result = {"success": True, "results": ["evidence"],
                  "attempts": [{"error": "ConnectionError"}]}
        self.assertFalse(EXECUTION.classify_tool_result(result)["failed"])
        partial = {"success": True, "results": [{"error": "ConnectionError"}]}
        self.assertTrue(EXECUTION.classify_tool_result(partial)["external_error"])


class OutcomeMetadataTests(unittest.TestCase):
    def annotate(self, logs, output="answer", success=True, valid=True, **kwargs):
        return EXECUTION.annotate_trace_outcome(
            logs, kwargs.get("result", {"final_output": output}), float(success),
            {"success": success, "valid": valid}, kwargs.get("task", "qa"))

    def test_real_final_and_binary_failure_remain_verified(self):
        logs = [log("analysis"), log("tool_plan"), log("final_answer")]
        result = self.annotate(logs, success=False)
        self.assertTrue(result["trace_outcome_valid"])
        self.assertEqual([item["trace_is_final"] for item in logs], [False, False, True])
        self.assertEqual([item["trace_outcome_success"] for item in logs], [False] * 3)
        self.assertEqual([item["trace_outcome_reward"] for item in logs], [0.0] * 3)

    def test_last_visible_tool_is_not_a_final(self):
        logs = [log("analysis"), log("tool_plan")]
        self.assertFalse(self.annotate(logs, result={})["trace_outcome_valid"])
        self.assertFalse(any(item["trace_is_final"] for item in logs))

    def test_evaluated_output_priority_does_not_mark_other_optional_answers(self):
        logs = [log("base", "base answer"), log("analysis"),
                log("final_answer", "final answer"), log("direct", "direct answer")]
        result = {"base_response": "base answer", "direct_output": "direct answer", "final_output": "final answer"}
        self.annotate(logs, result=result)
        self.assertEqual([item["trace_is_final"] for item in logs], [True, False, False, False])
        self.assertEqual([item["trace_outcome_valid"] for item in logs], [True, False, False, False])

    def test_truncated_final_and_invalid_verifier_do_not_supply_outcome(self):
        for logs, valid in (([log("final_answer", finish_reason="length")], True),
                            ([log("final_answer")], False)):
            self.assertFalse(self.annotate(logs, valid=valid)["trace_outcome_valid"])

    def test_unrecovered_external_failure_cannot_make_terminal_negative_label(self):
        logs = [log("tool_plan", trace_external_error=True, trace_tool_failed=True,
                    trace_action_key="search"), log("final_answer")]
        self.assertFalse(self.annotate(logs, success=False)["trace_outcome_valid"])

    def test_success_or_successful_same_action_retry_recovers_terminal_trust(self):
        for success, retry in ((True, False), (False, True)):
            logs = [log("tool_plan", trace_external_error=True, trace_tool_failed=True,
                        trace_action_key="search")]
            if retry:
                logs.append(log("tool_plan", trace_action_key="search"))
            logs.append(log("final_answer"))
            self.assertTrue(self.annotate(logs, success=success)["trace_outcome_valid"])
            self.assertTrue(logs[0]["trace_external_error"])


class VerifierDetailsTests(unittest.TestCase):
    def scorer(self, response, parsed=None):
        return extract_function("train-roma/utils.py", "compute_score", {
            "llm_scorer_engine": lambda *args, **kwargs: response,
            "AnswerVerification": object,
            "extract_json_from_markdown": lambda value: parsed,
        })

    def test_true_and_false_keep_legacy_result_and_trusted_boolean(self):
        for success in (True, False):
            score = self.scorer(NS(true_false=success))
            self.assertIs(score("q", "gt", "a"), success)
            self.assertEqual(score("q", "gt", "a", return_details=True),
                             {"valid": True, "success": success, "legacy_result": success})

    def test_parse_failure_and_nonboolean_are_not_verifier_negative_labels(self):
        for parsed, legacy in ((None, False), ({}, False), ({"true_false": "False"}, "False")):
            score = self.scorer("unparsed response", parsed)
            self.assertEqual(score("q", "gt", "a"), legacy)
            details = score("q", "gt", "a", return_details=True)
            self.assertFalse(details["valid"])
            self.assertIsNone(details["success"])
            self.assertEqual(details["legacy_result"], legacy)

    def test_both_evaluate_wrappers_still_return_float_and_collect_details(self):
        def score(*args, **kwargs):
            return {"valid": False, "success": None, "legacy_result": False} if kwargs else False
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            evaluate = extract_function(path, "evaluate", {
                "asyncio": asyncio, "compute_score": score, "Optional": Optional})
            evaluate = REWARD.reward(evaluate)
            details = {}
            self.assertEqual(asyncio.run(evaluate("q", "gt", "a", trace_outcome=details)), 0.0)
            self.assertEqual(details, {"valid": False, "success": None, "legacy_result": False})
            self.assertEqual(asyncio.run(evaluate("q", "gt", "a")), 0.0)


if __name__ == "__main__":
    unittest.main()
