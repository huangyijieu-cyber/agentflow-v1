"""CPU regressions for TRACE prompt snapshots and rollout-to-batch metadata."""
import ast
import contextlib
import io
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

ROOT = Path(__file__).resolve().parents[2]


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

    def test_all_planner_turns_capture_original_context_without_current_instructions(self):
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
                    self.assertIn(log["trace_state_context"], prompt)
                    self.assertNotIn("Select ONE tool best suited", log["trace_state_context"])
                    self.assertNotIn("Select the **single best tool**", log["trace_state_context"])
                    self.assertEqual(log["response_token_ids"], [202, 303])
                self.assertIn("evidence before action", planner.logs[2]["trace_state_context"])
                self.assertNotIn("evidence after action", planner.logs[2]["trace_state_context"])
                self.assertIn("evidence after action", planner.logs[3]["trace_state_context"])
                self.assertNotIn("generated response", planner.logs[1]["trace_state_context"])

    def test_snapshot_does_not_read_memory_twice_or_add_response(self):
        planner = self.make_planner()
        reads = []
        def get_actions():
            reads.append(1)
            return f"state snapshot {len(reads)}"
        planner.generate_next_step("question", None, "analysis", NS(get_actions=get_actions), 1, 3)
        log = planner.logs[0]
        self.assertEqual(len(reads), 1)
        self.assertIn("state snapshot 1", log["trace_state_context"])
        self.assertNotIn(log["response"], log["trace_state_context"])


class RolloutMetadataTests(unittest.TestCase):
    def test_both_encoders_preserve_native_ids_and_state_metadata(self):
        class Triplet(NS):
            pass
        log = {"prompt_token_ids": [4, 5], "response_token_ids": [6, 7],
               "finish_reason": "stop", "trace_state_context": "snapshot",
               "trace_turn_type": "analysis", "trace_original_turn": 3,
               "trace_action_key": "search-command",
               "extraction_valid": True}
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
                self.assertEqual([item["trace_dup"] for item in result], [False, True, True])

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
        failed = extract("agentflow/agentflow/solver.py", "_trace_tool_result_failed", {})
        for value in (None, [], {"error": "upstream unavailable"}, {"success": False},
                      [{"status": "timeout"}], "Error fetching URL: timed out",
                      "Error in execute_tool_command: failed", "Execution timed out after 3 seconds"):
            with self.subTest(value=value):
                self.assertTrue(failed(value))
        for value in ("The article discusses error correction.", "A failed search can still be useful.",
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


if __name__ == "__main__":
    unittest.main()
