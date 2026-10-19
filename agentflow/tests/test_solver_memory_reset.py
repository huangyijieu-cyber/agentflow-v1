"""Offline regressions for task-local QA memory on a reused Solver."""

import ast
import json
import os
import unittest
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union


ROOT = Path(__file__).resolve().parents[2]


def load_class(relative_path, class_name, scope):
    """Execute the production class without importing model/server dependencies."""
    tree = ast.parse((ROOT / relative_path).read_text())
    node = next(item for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == class_name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative_path, "exec"), scope)
    return scope[class_name]


Memory = load_class("agentflow/agentflow/models/memory.py", "Memory", {
    "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
    "Union": Union, "os": os,
})
Solver = load_class("agentflow/agentflow/solver.py", "Solver", {
    "Optional": Optional, "deepcopy": deepcopy, "json": json,
    "time": SimpleNamespace(time=lambda: 0.0),
    "nullcontext": nullcontext, "profiling_enabled": lambda: False,
    "is_search_tool": lambda _: False,
    "log_info": lambda *_: None,
    "make_json_serializable_truncated": lambda value: value,
})


class TrackingMemory(Memory):
    def __init__(self):
        super().__init__()
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1
        super().clear()


class PlannerStub:
    def __init__(self, memory):
        self.memory = memory
        self.available_tools = ["TestTool"]
        self.toolbox_metadata = {"TestTool": {}}
        self.logs = []
        self.observations = []

    def observe(self, question, phase, step=None):
        self.observations.append({
            "question": question, "phase": phase, "step": step,
            "actions": deepcopy(self.memory.get_all_actions()),
            "prompt_actions": deepcopy(self.memory.get_actions()),
            "files": deepcopy(self.memory.get_files()),
        })

    def generate_base_response(self, question, image, max_tokens):
        self.observe(question, "base")
        return question

    def analyze_query(self, question, image, max_tokens=2048):
        self.observe(question, "analysis")
        return question

    def generate_next_step(self, question, image, analysis, memory, step,
                           max_steps, json_data, max_tokens=2048):
        self.observe(question, "next", step)
        return question, step

    @staticmethod
    def extract_context_subgoal_and_tool(next_step):
        question, step = next_step
        return question, f"{question}/{step}", "TestTool"

    def generate_final_output(self, question, image, memory, max_tokens=2048):
        self.observe(question, "final")
        return question

    def generate_direct_output(self, question, image, memory, max_tokens=2048):
        self.observe(question, "direct")
        return question


class ExecutorStub:
    def __init__(self, memory):
        self.memory = memory

    def set_query_cache_dir(self, directory):
        pass

    def generate_tool_command(self, question, image, context, sub_goal,
                              tool_name, metadata, step, json_data):
        return question, step

    @staticmethod
    def extract_explanation_and_command(command):
        return "analysis", "explanation", command

    def execute_tool_command(self, tool_name, command):
        question, step = command
        self.memory.add_file(f"{question}-{step}.txt")
        return {"question": question, "step": step}


class VerifierStub:
    def __init__(self, stop_after):
        self.stop_after = stop_after

    def verificate_context(self, question, image, analysis, memory, step, json_data):
        return step >= self.stop_after[question]

    @staticmethod
    def extract_conclusion(stop):
        return "verified", "STOP" if stop else "CONTINUE"


class SolverMemoryResetTests(unittest.TestCase):
    def make_solver(self, output_types="final,direct"):
        memory = TrackingMemory()
        planner = PlannerStub(memory)
        solver = Solver(planner, VerifierStub({"long": 20, "short": 5}),
                        memory, ExecutorStub(memory), output_types=output_types,
                        max_steps=20, max_time=600, verbose=False, task="qa")
        return solver, planner, memory

    def test_long_then_short_question_has_no_stale_actions_or_files(self):
        solver, planner, memory = self.make_solver()
        first, _ = solver.solve("long")
        self.assertEqual(first["step_count"], 20)
        self.assertEqual(len(first["memory"]), 20)
        self.assertEqual(len(memory.get_files()), 20)

        second, _ = solver.solve("short")
        self.assertEqual(second["step_count"], 5)
        observations = [item for item in planner.observations if item["question"] == "short"]
        analysis = next(item for item in observations if item["phase"] == "analysis")
        self.assertEqual(analysis["actions"], {}, "Reset before the first Planner request")
        self.assertEqual(analysis["files"], [])
        self.assertEqual(memory.clear_count, 2, "Reset once per question, not per turn")
        self.assertEqual(list(second["memory"]), [f"Action Step {step}" for step in range(1, 6)])
        self.assertEqual(len(first["memory"]), 20, "The saved first task trace is retained")
        next_steps = [item for item in observations if item["phase"] == "next"]
        self.assertEqual(next_steps[0]["prompt_actions"], {})
        self.assertEqual(list(next_steps[1]["prompt_actions"]), ["Action Step 1"])
        for item in observations:
            for action in item["actions"].values():
                self.assertEqual(action["result"]["question"], "short")
            for file in item["files"]:
                self.assertTrue(file["file_name"].startswith("short-"))

        for phase in ("final", "direct"):
            item = next(item for item in observations if item["phase"] == phase)
            self.assertEqual(list(item["prompt_actions"]), ["Action Step 3", "Action Step 4", "Action Step 5"])
            self.assertEqual(len(item["files"]), 5, "Current-task memory survives all output stages")

    def test_base_only_resets_before_the_planner_and_early_return(self):
        solver, planner, memory = self.make_solver(output_types="base")
        memory.add_action(20, "TestTool", "old", "old", "old-result")
        memory.add_file("old.txt")

        result, _ = solver.solve("short")

        self.assertEqual(result["base_response"], "short")
        self.assertEqual(planner.observations[0]["phase"], "base")
        self.assertEqual(planner.observations[0]["actions"], {})
        self.assertEqual(planner.observations[0]["files"], [])
        self.assertEqual(memory.clear_count, 1)


if __name__ == "__main__":
    unittest.main()
