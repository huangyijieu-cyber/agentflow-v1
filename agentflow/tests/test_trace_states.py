"""CPU regressions for TRACE prompt snapshots and rollout-to-batch metadata."""

import ast
import contextlib
import io
import importlib.util
import json
import logging
import os
import re
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import asyncio
from typing import Optional


ROOT = Path(__file__).resolve().parents[2]
EXECUTION_SPEC = importlib.util.spec_from_file_location(
    "trace_execution_states_test", ROOT / "agentflow/agentflow/trace/execution.py")
EXECUTION = importlib.util.module_from_spec(EXECUTION_SPEC)
EXECUTION_SPEC.loader.exec_module(EXECUTION)


def extract(path, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in ast.walk(tree)
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class FakeEngine:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.last_generation_metadata = {
            "prompt_token_ids": [101, len(self.prompts)],
            "response_token_ids": [202, 303],
            "finish_reason": "stop",
        }
        return "generated response"


class PlannerStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.planner_class = extract("agentflow/agentflow/models/planner.py", "Planner", {
            "Any": Any, "Dict": Dict, "List": List, "Tuple": Tuple, "Memory": NS,
            "os": os, "json": json, "re": re, "Image": None,
            "QueryAnalysis": type("QueryAnalysis", (), {}),
            "NextStep": type("NextStep", (), {}),
        })

    def make_planner(self, multimodal=False):
        planner = object.__new__(self.planner_class)
        planner.llm_engine = FakeEngine()
        planner.logs = []
        planner.available_tools = ["Search"]
        planner.toolbox_metadata = {"Search": "Find evidence"}
        planner.is_multimodal = multimodal
        planner.temperature = 0.7
        return planner

    def test_all_planner_turns_capture_evidence_without_current_instructions(self):
        # Words used by template delimiters are deliberately in the user body.
        question = "Question body\n\nInstructions:\nkeep this user text\nTask: keep this too"
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal), contextlib.redirect_stdout(io.StringIO()):
                planner = self.make_planner(multimodal)
                state = {"actions": "evidence before action"}
                memory = NS(get_actions=lambda: state["actions"])
                planner.generate_base_response(question, None)
                planner.analyze_query(question, None)
                planner.generate_next_step(question, None, planner.query_analysis, memory, 1, 3)
                state["actions"] = "evidence after action"
                planner.generate_final_output(question, None, memory)
                planner.generate_direct_output(question, None, memory)
                self.assertEqual([log["trace_turn_type"] for log in planner.logs],
                                 ["base", "analysis", "tool_plan", "final_answer", "direct"])
                self.assertEqual([log["trace_original_turn"] for log in planner.logs], list(range(5)))
                for log, prompt in zip(planner.logs, planner.llm_engine.prompts):
                    self.assertEqual(log["prompt"], prompt)
                    self.assertIn(question, log["trace_state_context"])
                    self.assertNotIn("Select ONE tool best suited", log["trace_state_context"])
                    self.assertNotIn("Select the **single best tool**", log["trace_state_context"])
                    self.assertEqual(log["response_token_ids"], [202, 303])
                self.assertEqual(planner.logs[0]["trace_state_context"], planner.logs[1]["trace_state_context"])
                self.assertIn("evidence before action", planner.logs[2]["trace_state_context"])
                self.assertNotIn("evidence after action", planner.logs[2]["trace_state_context"])
                self.assertIn("evidence after action", planner.logs[3]["trace_state_context"])
                self.assertNotIn("generated response", planner.logs[1]["trace_state_context"])

    def test_unchanged_evidence_has_identical_tool_and_final_score_contexts(self):
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal), contextlib.redirect_stdout(io.StringIO()):
                planner = self.make_planner(multimodal)
                planner.query_analysis = "model-only reasoning"
                memory = NS(get_actions=lambda: {"Action Step 1": {"result": "retrieved evidence"}})
                planner.generate_next_step("question", None, planner.query_analysis, memory, 2, 5)
                planner.generate_next_step("question", None, "different reasoning", memory, 3, 5)
                planner.generate_final_output("question", None, memory)
                planner.generate_direct_output("question", None, memory)
                contexts = [log["trace_state_context"] for log in planner.logs]
                self.assertEqual(len(set(contexts)), 1)
                self.assertIn("retrieved evidence", contexts[0])
                self.assertNotIn("reasoning", contexts[0])
                # Generation still sees its original stage-specific inputs.
                self.assertIn("model-only reasoning", planner.logs[0]["prompt"])
                self.assertIn("different reasoning", planner.logs[1]["prompt"])
                self.assertNotIn("model-only reasoning", planner.logs[2]["prompt"])
                self.assertIn("model-only reasoning", planner.logs[3]["prompt"])

    def test_initial_score_context_is_shared_before_any_tool_observation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            planner = self.make_planner()
            planner.generate_base_response("question", None)
            planner.analyze_query("question", None)
            planner.generate_next_step("question", None, planner.query_analysis,
                                       NS(get_actions=lambda: {}), 1, 3)
        self.assertEqual(len({log["trace_state_context"] for log in planner.logs}), 1)

    def test_snapshot_does_not_read_memory_twice_or_add_response(self):
        for turn in ("tool_plan", "final_answer", "direct"):
            with self.subTest(turn=turn):
                planner = self.make_planner()
                planner.query_analysis = "analysis"
                reads = []
                def get_actions():
                    reads.append(1)
                    return f"state snapshot {len(reads)}"
                memory = NS(get_actions=get_actions)
                if turn == "tool_plan":
                    planner.generate_next_step("question", None, "analysis", memory, 1, 3)
                elif turn == "final_answer":
                    planner.generate_final_output("question", None, memory)
                else:
                    planner.generate_direct_output("question", None, memory)
                log = planner.logs[0]
                self.assertEqual(len(reads), 1)
                self.assertIn("state snapshot 1", log["trace_state_context"])
                self.assertIn("state snapshot 1", log["prompt"])
                self.assertNotIn(log["response"], log["trace_state_context"])


class RolloutMetadataTests(unittest.TestCase):
    def test_both_encoders_preserve_native_ids_and_state_metadata(self):
        class Triplet(NS):
            pass
        log = {"prompt_token_ids": [4, 5], "response_token_ids": [6, 7],
               "finish_reason": "stop", "trace_state_context": "snapshot",
               "trace_turn_type": "analysis", "trace_original_turn": 3,
               "trace_action_key": "search-command",
               "extraction_valid": True, "trace_external_error": True,
               "trace_error_type": "connection_error", "trace_error_source": "external",
               "trace_is_final": False, "trace_outcome_valid": False,
               "trace_outcome_success": None, "trace_outcome_reward": 0.0}
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            with self.subTest(path=path):
                encode = extract(path, "encode_logs", {"Triplet": Triplet})
                triplet = encode(None, [log])[0] if "profiled" in path else encode([log])[0]
                self.assertEqual(triplet.prompt["token_ids"], [4, 5])
                self.assertEqual(triplet.response["token_ids"], [6, 7])
                self.assertEqual(triplet.metadata["trace_original_turn"], 3)
                self.assertEqual(triplet.metadata["trace_state_context"], "snapshot")
                self.assertEqual(triplet.metadata["finish_reason"], "stop")
                self.assertEqual(triplet.metadata["trace_action_key"], "search-command")
                for key in ("trace_external_error", "trace_error_type", "trace_error_source",
                            "trace_is_final", "trace_outcome_valid", "trace_outcome_success",
                            "trace_outcome_reward"):
                    self.assertEqual(triplet.metadata[key], log[key])
                missing_ids = dict(log, response_token_ids=None)
                with self.assertRaisesRegex(RuntimeError, "missing exact vLLM token ids"):
                    encode(None, [missing_ids]) if "profiled" in path else encode([missing_ids])

    def test_flags_reject_failed_invalid_and_repeated_actual_action(self):
        flags = extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        logs = [
            {"response": "first rationale", "trace_action_key": "same tool and command"},
            {"response": "different rationale", "trace_action_key": "same tool and command"},
            {"response": "unparseable", "extraction_valid": False},
            {"response": "tool decision", "trace_tool_failed": True},
            {"response": "first rationale", "trace_action_key": "different command"},
        ]
        result = flags(logs, "qa")
        self.assertEqual([{key: item[key] for key in ("valid", "dup")} for item in result], [
            {"valid": True, "dup": False}, {"valid": True, "dup": False},
            {"valid": True, "dup": False}, {"valid": True, "dup": False},
            {"valid": True, "dup": True},
        ])
        self.assertEqual([item["trace_valid"] for item in result], [True, True, False, False, True])
        self.assertEqual([item["trace_dup"] for item in result], [False, True, False, False, True])

    def test_legacy_flags_retain_task_validation_with_trace_metadata_present(self):
        flags = extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        for task, responses, expected_valid in (
                ("qa", ["answer", "", "answer"], [True, False, True]),
                ("alfworld", ["<action>look</action>", "plain text", "<action>look</action>"],
                 [True, False, True]),
                ("webshop", ["search[book]", "[ERROR] unavailable", "search[book]"],
                 [True, False, True])):
            with self.subTest(task=task):
                logs = [{"response": response, "extraction_valid": False,
                         "trace_tool_failed": True, "trace_action_key": "same-command"}
                        for response in responses]
                result = flags(logs, task)
                self.assertEqual([item["valid"] for item in result], expected_valid)
                self.assertEqual([item["dup"] for item in result], [False, False, True])
                self.assertEqual([item["trace_valid"] for item in result], [False] * 3)
                self.assertEqual([item["trace_dup"] for item in result], [False] * 3)

    def test_recovered_retry_is_not_a_trace_duplicate_of_failed_attempt(self):
        flags = extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        logs = [
            {"response": "same rationale", "trace_action_key": "search", "trace_external_error": True},
            {"response": "same rationale", "trace_action_key": "search"},
            {"response": "another rationale", "trace_action_key": "search"},
        ]
        result = flags(logs, "qa")
        self.assertEqual([item["valid"] for item in result], [True] * 3)
        self.assertEqual([item["dup"] for item in result], [False, True, False])
        self.assertEqual([item["trace_valid"] for item in result], [False, True, True])
        self.assertEqual([item["trace_dup"] for item in result], [False, False, True])

    def test_only_runner_injected_answer_instruction_is_removed_from_score_context(self):
        remove = extract("train-roma/rollout.py", "_remove_injected_qa_output_format", {})
        instruction = ("When ready, output the final answer enclosed in <answer> and </answer> tags. "
                       "Do not generate any content after the </answer> tag.")
        original = "User question. " + instruction  # User's own instruction must remain.
        generation = original + " " + instruction
        logs = [{"trace_state_context": "Query: " + generation + "\nEvidence: " + generation,
                 "prompt": generation, "prompt_token_ids": [1, 2], "response_token_ids": [3]},
                {"trace_state_context": None}]
        remove(logs, generation, original)
        self.assertEqual(logs[0]["trace_state_context"], "Query: " + original + "\nEvidence: " + generation)
        self.assertEqual(logs[0]["prompt"], generation)
        self.assertEqual(logs[0]["prompt_token_ids"], [1, 2])
        self.assertEqual(logs[0]["response_token_ids"], [3])
        self.assertIsNone(logs[1]["trace_state_context"])
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            tree = ast.parse((ROOT / path).read_text())
            method = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.AsyncFunctionDef) and node.name == "_solve_and_evaluate")
            calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "_remove_injected_qa_output_format"]
            self.assertEqual(len(calls), 1, path)
            self.assertEqual([ast.unparse(argument) for argument in calls[0].args],
                             ["planner_logs", "prompt", "task['question']"])

    def test_profiled_runner_keeps_legacy_flag_defaults(self):
        flags = extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        tree = ast.parse((ROOT / "train-roma/rollout_main_profiled.py").read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets)
                          and isinstance(node.value, ast.Dict) and any(
                              isinstance(key, ast.Constant) and key.value == "turn_flags" for key in node.value.keys))
        logs = [{"response": "first", "trace_action_key": "same"},
                {"response": "second", "trace_action_key": "same", "extraction_valid": False}]
        scope = {"planner_logs": logs, "self": NS(task="qa"), "_compute_turn_flags": flags}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), "profiled", "exec"), scope)
        self.assertEqual(scope["metadata"]["turn_flags"], [
            {"trace_valid": True, "trace_dup": False}, {"trace_valid": False, "trace_dup": True}])

    def test_failure_detection_checks_envelopes_and_known_errors_without_prose_search(self):
        failed = extract("agentflow/agentflow/solver.py", "_trace_tool_result_failed",
                         {"classify_tool_result": EXECUTION.classify_tool_result})
        for value in (None, {"error": "upstream unavailable"}, {"success": False},
                      [{"status": "timeout"}], "Error fetching URL: timed out",
                      "Error in execute_tool_command: failed", "Execution timed out after 3 seconds"):
            with self.subTest(value=value):
                self.assertTrue(failed(value))
        for value in ([], "The article discusses error correction.", "A failed search can still be useful.",
                      {"content": "Error: this is quoted source text", "error": None},
                      [{"content": "success", "success": True}], 0, False):
            with self.subTest(value=value):
                self.assertFalse(failed(value))


class DaemonMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class TensorDict(dict):
            def __init__(self, value, batch_size):
                super().__init__(value)
        class DataProto:
            def __init__(self, batch):
                self.batch = batch
                self.non_tensor_batch = {}
        scope = {"torch": torch, "np": np, "uuid": uuid,
                 "logger": logging.getLogger(__name__), "TensorDict": TensorDict,
                 "DataProto": DataProto, "List": List}
        for name in ("get_left_padded_ids_and_attention_mask", "get_right_padded_ids_and_attention_mask"):
            extract("agentflow/verl/daemon.py", name, scope)
        cls.get_batch = staticmethod(extract("agentflow/verl/daemon.py", "get_train_data_batch", scope))

    def build_batch(self, triplets, flags=None, ground_truth=None):
        rollout = NS(triplets=triplets, metadata={"turn_flags": flags or []})
        daemon = NS(is_train=True, _completed_rollouts={"rollout-0": rollout},
                    _total_tasks_queued=1, pad_token_id=0,
                    _task_id_to_original_sample={"rollout-0": {
                        "data_id": "question-0", "result": ground_truth}},
                    _fillna_reward=lambda rollout: 1.0,
                    _get_format_metrics=lambda prefix: {f"{prefix}/format": 1.0})
        return self.get_batch(daemon, 4, 4, "cpu")

    def triplet(self, index, **metadata):
        return NS(prompt={"token_ids": [1, 2]}, response={"token_ids": [3]}, metadata={
            "trace_original_turn": index, "trace_state_context": f"state-{index}",
            "trace_turn_type": "tool_plan", "finish_reason": "stop",
            "trace_action_key": f"command-{index}", **metadata})

    def test_daemon_retains_skipped_turn_gaps_and_alias_gt_as_one_object_per_row(self):
        triplets = [self.triplet(i) for i in range(3)]
        triplets[1].prompt["token_ids"] = []
        triplets[1].response["token_ids"] = []
        aliases = ["answer", "answer alias"]
        batch, metrics = self.build_batch(triplets, ground_truth=aliases)
        data = batch.non_tensor_batch
        self.assertEqual(data["turn_index_list"].tolist(), [0, 2])
        self.assertEqual(data["trace_original_turn"].tolist(), [0, 2])
        self.assertEqual(data["trace_state_context"].tolist(), ["state-0", "state-2"])
        self.assertEqual(data["trace_action_key"].tolist(), ["command-0", "command-2"])
        self.assertEqual(data["trace_gt"].shape, (2,))
        self.assertEqual(data["trace_gt"].tolist(), [aliases, aliases])
        self.assertEqual(data["trace_valid"].tolist(), [True, True])
        self.assertEqual(metrics["agent_mode/n_sample_to_train"], 2)

    def test_trace_quality_does_not_change_legacy_daemon_quality_arrays(self):
        compute_flags = extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        logs = [
            {"response": "rationale one", "trace_action_key": "command"},
            {"response": "rationale two", "trace_action_key": "command"},
            {"response": "unparseable", "extraction_valid": False},
            {"response": "failed tool", "trace_tool_failed": True},
        ]
        flags = compute_flags(logs, "qa")
        batch, _ = self.build_batch([self.triplet(i) for i in range(4)], flags, "answer")
        data = batch.non_tensor_batch
        self.assertEqual(data["turn_valid_list"].tolist(), [True] * 4)
        self.assertEqual(data["turn_dup_list"].tolist(), [False] * 4)
        self.assertEqual(data["trace_valid"].tolist(), [True, False, False, False])
        profiled_flags = [{key: item[key] for key in ("trace_valid", "trace_dup")} for item in flags]
        profiled_batch, _ = self.build_batch([self.triplet(i) for i in range(4)], profiled_flags, "answer")
        self.assertEqual(profiled_batch.non_tensor_batch["turn_valid_list"].tolist(), [True] * 4)
        self.assertEqual(profiled_batch.non_tensor_batch["turn_dup_list"].tolist(), [False] * 4)
        self.assertEqual(profiled_batch.non_tensor_batch["trace_valid"].tolist(), [True, False, False, False])

    def test_invalid_actions_keep_their_state_rows_and_are_filtered_on_both_sides(self):
        triplets = [self.triplet(i) for i in range(8)]
        triplets[1].metadata["trace_tool_failed"] = True
        triplets[2].metadata["trace_extraction_valid"] = False
        triplets[3].metadata["finish_reason"] = "length"
        triplets[4].response["token_ids"] = [3] * 5
        triplets[5].metadata["trace_state_context"] = None
        flags = [{"valid": True, "dup": False} for _ in triplets]
        flags[6]["valid"] = False
        flags[7]["dup"] = True
        batch, _ = self.build_batch(triplets, flags, "answer")
        data = batch.non_tensor_batch
        self.assertEqual(data["trace_original_turn"].tolist(), list(range(8)))
        self.assertEqual(data["trace_valid"].tolist(), [True] + [False] * 7)
        self.assertEqual(data["trace_state_context"][1], "state-1")
        self.assertEqual(data["trace_turn_type"].tolist(), ["tool_plan"] * 8)
        self.assertEqual(batch.batch["responses"].shape, (8, 4))

    def test_external_failure_and_trusted_outcome_metadata_reach_batch(self):
        triplets = [
            self.triplet(0, trace_turn_type="analysis", trace_state_context=None),
            self.triplet(1, trace_external_error=True, trace_error_type="connection_error",
                         trace_error_source="external"),
            self.triplet(2, trace_turn_type="final_answer", trace_is_final=True),
        ]
        for triplet in triplets:
            triplet.metadata.update(trace_outcome_valid=True, trace_outcome_success=True,
                                    trace_outcome_reward=1.0)
        batch, _ = self.build_batch(triplets, ground_truth="answer")
        data = batch.non_tensor_batch
        self.assertEqual(data["trace_external_error"].tolist(), [False, True, False])
        self.assertEqual(data["trace_valid"].tolist(), [True, False, True])
        self.assertEqual(data["trace_is_final"].tolist(), [False, False, True])
        self.assertEqual(data["trace_outcome_valid"].tolist(), [True] * 3)
        self.assertEqual(data["trace_outcome_success"].tolist(), [True] * 3)
        self.assertEqual(data["trace_outcome_reward"].tolist(), [1.0] * 3)
        self.assertEqual(data["trace_error_type"].tolist(), ["", "connection_error", ""])

    def test_daemon_keeps_trace_retry_eligible_despite_legacy_duplicate(self):
        compute_flags = extract("train-roma/rollout.py", "_compute_turn_flags", {"re": re})
        logs = [
            {"response": "same rationale", "trace_action_key": "search", "trace_external_error": True},
            {"response": "same rationale", "trace_action_key": "search"},
        ]
        flags = compute_flags(logs, "qa")
        triplets = [self.triplet(index, trace_action_key="search",
                                trace_external_error=index == 0) for index in range(2)]
        batch, _ = self.build_batch(triplets, flags, "answer")
        self.assertEqual(batch.non_tensor_batch["turn_dup_list"].tolist(), [False, True])
        self.assertEqual(batch.non_tensor_batch["trace_dup"].tolist(), [False, False])
        self.assertEqual(batch.non_tensor_batch["trace_valid"].tolist(), [False, True])


EXECUTION_ROOT = Path(__file__).resolve().parents[2]
CLASSIFY_SPEC = importlib.util.spec_from_file_location(
    "trace_execution_test", EXECUTION_ROOT / "agentflow/agentflow/trace/execution.py")
CLASSIFY_EXECUTION = importlib.util.module_from_spec(CLASSIFY_SPEC)
CLASSIFY_SPEC.loader.exec_module(CLASSIFY_EXECUTION)
REWARD_SPEC = importlib.util.spec_from_file_location("trace_reward_test", EXECUTION_ROOT / "agentflow/reward.py")
REWARD = importlib.util.module_from_spec(REWARD_SPEC)
REWARD_SPEC.loader.exec_module(REWARD)


def extract_function(path, name, scope):
    tree = ast.parse((EXECUTION_ROOT / path).read_text())
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
                status = CLASSIFY_EXECUTION.classify_tool_result(result)
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
                self.assertFalse(CLASSIFY_EXECUTION.classify_tool_result(result)["failed"])

    def test_model_errors_and_ambiguous_executor_timeout_stay_separate(self):
        for result in (
            "Error in execute_tool_command: SyntaxError",
            "No result was generated because the tool was not found.",
            "Execution timed out after 3 seconds (phase=tool.execute)",
            "Error: invalid parameter: expected 500 characters",
        ):
            status = CLASSIFY_EXECUTION.classify_tool_result(result)
            self.assertTrue(status["failed"])
            self.assertFalse(status["external_error"])

    def test_successful_retry_does_not_inherit_attempt_error(self):
        result = {"success": True, "results": ["evidence"],
                  "attempts": [{"error": "ConnectionError"}]}
        self.assertFalse(CLASSIFY_EXECUTION.classify_tool_result(result)["failed"])
        partial = {"success": True, "results": [{"error": "ConnectionError"}]}
        self.assertTrue(CLASSIFY_EXECUTION.classify_tool_result(partial)["external_error"])


class OutcomeMetadataTests(unittest.TestCase):
    def annotate(self, logs, output="answer", success=True, valid=True, **kwargs):
        return CLASSIFY_EXECUTION.annotate_trace_outcome(
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
    def scorer(self, response, parsed=None, engine=None):
        return extract_function("train-roma/utils.py", "compute_score", {
            "llm_scorer_engine": engine or (lambda *args, **kwargs: response),
            "AnswerVerification": object,
            "extract_json_from_markdown": lambda value: parsed,
            "re": re,
        })

    def test_true_and_false_keep_legacy_result_and_trusted_boolean(self):
        for success in (True, False):
            score = self.scorer(NS(true_false=success))
            self.assertIs(score("q", "gt", "a"), success)
            self.assertEqual(score("q", "gt", "a", return_details=True),
                             {"valid": True, "success": success, "legacy_result": success})

    def test_parse_failure_and_nonboolean_are_not_verifier_negative_labels(self):
        for parsed, legacy in ((None, False), ({}, False), ([], False), ([True], False),
                               (1, False), ({"true_false": "False"}, "False")):
            score = self.scorer("unparsed response", parsed)
            self.assertEqual(score("q", "gt", "a"), legacy)
            details = score("q", "gt", "a", return_details=True)
            self.assertFalse(details["valid"])
            self.assertIsNone(details["success"])
            self.assertEqual(details["legacy_result"], legacy)

    def test_api_error_envelopes_raise_without_becoming_rewards(self):
        for kind, expected in (("APIConnectionError", "APIConnectionError"),
                               ("rate_limit", "rate_limit"),
                               ({"code": 503}, "APIError"),
                               ("unsafe request details", "APIError")):
            response = {"error": kind, "message": "private request", "details": "private credential"}
            score = self.scorer(response)
            for detailed in (False, True):
                with self.subTest(kind=kind, detailed=detailed), \
                     contextlib.redirect_stdout(io.StringIO()) as output, \
                     self.assertRaisesRegex(RuntimeError, f"Judge API request failed \\({expected}\\)") as error:
                    score("q", "gt", "a", return_details=detailed)
                self.assertNotIn("private", str(error.exception))
                self.assertEqual(output.getvalue(), "")

    def test_nontext_responses_raise_without_becoming_rewards(self):
        for response in (None, [], 42, b"response", {}, {"true_false": True}):
            score = self.scorer(response)
            for detailed in (False, True):
                with self.subTest(response=response, detailed=detailed), \
                     self.assertRaisesRegex(RuntimeError, "Judge returned unsupported response type"):
                    score("q", "gt", "a", return_details=detailed)

    def test_raised_api_errors_preserve_failure_instead_of_returning_false(self):
        def fail(*args, **kwargs):
            raise ConnectionError("connection interrupted")
        score = self.scorer(None, engine=fail)
        with self.assertRaisesRegex(RuntimeError, "Judge API request failed \\(ConnectionError\\)") as error:
            score("q", "gt", "a", return_details=True)
        self.assertIsInstance(error.exception.__cause__, ConnectionError)

    def test_actual_text_parser_preserves_plain_and_markdown_verdicts(self):
        scope = {"json": json, "re": re}
        scope["_try_load"] = extract_function("train-roma/utils.py", "_try_load", scope)
        parser = extract_function("train-roma/utils.py", "extract_json_from_markdown", scope)
        for success in (True, False):
            payload = json.dumps({"analysis": "checked", "true_false": success})
            for response in (payload, f"```json\n{payload}\n```", "Verdict: " + payload):
                score = self.scorer(response)
                score.__globals__["extract_json_from_markdown"] = parser
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(score("q", "gt", "a", return_details=True),
                                     {"valid": True, "success": success, "legacy_result": success})

    def test_evaluate_wrappers_propagate_judge_failure_without_reward(self):
        score = self.scorer({"error": "APIConnectionError"})
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            evaluate = extract_function(path, "evaluate", {
                "asyncio": asyncio, "compute_score": score, "Optional": Optional})
            evaluate = REWARD.reward(evaluate)
            details = {}
            with self.subTest(path=path), self.assertRaisesRegex(RuntimeError, "Judge API request failed"):
                asyncio.run(evaluate("q", "gt", "a", trace_outcome=details))
            self.assertEqual(details, {})

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
