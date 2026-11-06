"""Both production entrypoints retain serving tokens and isolate validation GT."""
import ast
import asyncio
import contextlib
import copy
import io
from pathlib import Path
import re
import time
from types import SimpleNamespace as NS
from typing import Optional
import unittest

import test_webshop_execution as execution_tests


ROOT = Path(__file__).resolve().parents[2]
execution = execution_tests.execution
PATHS = ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py")


def extract(path, name, scope):
    source = ROOT / path
    node = next(node for node in ast.walk(ast.parse(source.read_text()))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[name]


class WebShopRolloutTests(unittest.TestCase):
    def flags(self, logs):
        function = extract(PATHS[0], "_compute_turn_flags", {"re": re})
        return function(logs, "webshop")

    def test_duplicate_filter_uses_state_and_response_without_changing_legacy_flags(self):
        logs = [{"response": "<action>click[buy now]</action>", "trace_state_context": state,
                 "trace_action_key": key}
                for state, key in (("product A", "A:buy"), ("product B", "B:buy"), ("product A", "A:buy"))]
        flags = self.flags(logs)
        self.assertEqual([row["dup"] for row in flags], [False, True, True])
        self.assertEqual([row["trace_dup"] for row in flags], [False, False, True])
        logs[0]["trace_tool_failed"] = True
        self.assertFalse(self.flags(logs)[2]["trace_dup"])

    def run_until_solve(self, path, *, train=False, trace=True, result_present=True):
        bound, reached = [], []
        env = NS(trace_enabled=trace, trace_manifest_enabled=True, set_trace_task=bound.append)
        training_env = NS(trace_enabled=True, trace_manifest_enabled=True)
        rollout = NS(solver=NS(env=env, executor=NS(tool_instances_cache={})), solve=lambda **kwargs: None)

        class ReachedSolve(Exception):
            pass

        async def to_thread(function, **kwargs):
            reached.append((env.trace_enabled, env.trace_manifest_enabled, kwargs))
            raise ReachedSolve

        scope = {"Any": object, "AgentFlowRollout": object, "asyncio": NS(to_thread=to_thread),
                 "time": time, "profiling_enabled": lambda: False}
        function = extract(path, "_solve_and_evaluate", scope)
        owner = NS(task="webshop", training_agent=NS(solver=NS(env=training_env)))
        row = {"question": 7}
        if result_present:
            row["result"] = "" if not train else "Offline fixed candidate GT."
        with self.assertRaises(ReachedSolve):
            asyncio.run(function(owner, "rollout", rollout, row, 0, val=not train, is_train=train))
        self.assertTrue(training_env.trace_enabled)
        return reached[0], bound

    def test_validation_disables_scorer_keeps_manifest_and_needs_no_result(self):
        for path in PATHS:
            for result_present in (False, True):
                with self.subTest(path=path, result_present=result_present):
                    (enabled, manifest, kwargs), bound = self.run_until_solve(path, result_present=result_present)
                    self.assertFalse(enabled)
                    self.assertTrue(manifest)
                    self.assertEqual(bound, [])
                    self.assertIs(type(kwargs["question"]), int)
                    self.assertEqual(kwargs["question"], 7)
                    self.assertEqual(kwargs["ground_truth"], "" if result_present else None)

    def test_only_trace_training_binds_prepared_task_rows(self):
        for path in PATHS:
            with self.subTest(path=path):
                (enabled, _, _), bound = self.run_until_solve(path, train=True)
                self.assertTrue(enabled)
                self.assertEqual(bound[0]["result"], "Offline fixed candidate GT.")
                (_, _, kwargs), bound = self.run_until_solve(path, train=True, trace=False, result_present=False)
                self.assertEqual(bound, [])
                self.assertIsNone(kwargs["ground_truth"])

    def test_both_entrypoints_encode_gt_states_continuous_outcome_and_exact_tokens(self):
        log = {"response": "<action>click[buy now]</action>", "finish_reason": "stop",
               "prompt_token_ids": [91, 8, 92], "response_token_ids": [93, 5, 94],
               "trace_state_context": "actual item before", "trace_goal_check_valid_before": True,
               "trace_scoring_valid": True, "trace_gt": "Fixed offline GT.",
               "trace_answer_instruction": "Describe the current candidate.",
               "trace_goal_metadata": {"goal_sha256": "goal", "manifest_sha256": "manifest",
                                       "task_key": "task:7", "template_version": "v1"}}
        execution.annotate_webshop_step(log, "actual item before", True, {
            "trace_state_context": "actual purchased item after", "trace_env_action": "click[buy now]",
            "trace_goal_check_valid": True, "trace_scoring_valid": True,
            "task_score": .6, "terminated": True, "won": False})
        execution.annotate_trace_outcome([log], {"base_response": log["response"]}, 0., {}, "webshop")
        for path in PATHS:
            with self.subTest(path=path):
                encoder = extract(path, "encode_logs", {"Triplet": NS})
                encoded = encoder(None, [log]) if "profiled" in path else encoder([log])
                triplet = encoded[0]
                self.assertEqual(triplet.prompt, {"token_ids": [91, 8, 92]})
                self.assertEqual(triplet.response, {"token_ids": [93, 5, 94]})
                self.assertEqual(triplet.metadata["trace_gt"], "Fixed offline GT.")
                self.assertEqual(triplet.metadata["trace_task"], "webshop")
                self.assertEqual(triplet.metadata["trace_state_view"], "webshop_state")
                self.assertEqual(triplet.metadata["trace_next_state_context"], "actual purchased item after")
                self.assertEqual(triplet.metadata["trace_outcome_reward"], .6)
                self.assertEqual(triplet.metadata["trace_env_task_score"], .6)
                self.assertFalse(triplet.metadata["trace_is_final"])
                self.assertTrue(triplet.metadata["trace_is_terminal"])
                self.assertTrue(triplet.metadata["trace_outcome_valid"])

    def test_missing_native_token_ids_raise_in_both_entrypoints(self):
        for path in PATHS:
            encoder = extract(path, "encode_logs", {"Triplet": NS})
            with self.assertRaisesRegex(RuntimeError, "exact vLLM token ids"):
                encoder(None, [{}]) if "profiled" in path else encoder([{}])


class WebShopSolverTests(unittest.TestCase):
    def solver(self, *, crash=False, trace=True):
        scope = {"Optional": Optional, "deepcopy": copy.deepcopy, "time": time,
                 "log_info": lambda *args: None, "annotate_webshop_step": execution.annotate_webshop_step,
                 "annotate_alfworld_step": execution.annotate_alfworld_step}
        cls = extract("agentflow/agentflow/solver.py", "Solver", scope)

        class Env:
            trace_enabled = trace
            trace_gt = "Fixed offline candidate-state GT."
            trace_answer_instruction = "Describe the actual product state."
            trace_goal_metadata = {"goal_sha256": "goal", "manifest_sha256": "manifest",
                                   "task_key": "task:7", "template_version": "v1"}
            trace_goal_check_valid = True
            trace_scoring_valid = True
            trace_state_context = "private actual candidate state 0"

            def reset(self, *, session):
                self.session, self.index = session, 0
                return "actor visible page 0", {"anchor": "actor visible page 0"}

            def step(self, response):
                if crash:
                    raise RuntimeError("native simulator failed")
                self.index += 1
                self.trace_state_context = f"private actual candidate state {self.index}"
                done = self.index == 2
                return f"actor visible page {self.index}", 0., done, {
                    "trace_state_context": self.trace_state_context, "trace_scoring_valid": True,
                    "trace_goal_check_valid": True, "terminated": done,
                    "trace_env_action": "click[buy now]" if done else "search[cup]",
                    "task_score": .8 if done else 0., "won": False,
                    "anchor": f"actor visible page {self.index}"}

        class Planner:
            def __init__(self):
                self.logs = []

            def generate_base_response(self, question, image, max_tokens):
                response = "<action>search[cup]</action>" if not self.logs else "<action>click[buy now]</action>"
                self.logs.append({"prompt": question, "response": response, "trace_turn_type": "base",
                                  "prompt_token_ids": [100 + len(self.logs)], "response_token_ids": [200],
                                  "finish_reason": "stop"})
                return response

        solver = object.__new__(cls)
        solver.task, solver.env, solver.planner = "webshop", Env(), Planner()
        solver.output_types, solver.max_tokens, solver.verbose = ["base"], 2048, False
        return solver

    def test_solver_captures_pre_and_post_states_without_changing_actor_io(self):
        solver = self.solver()
        with contextlib.redirect_stdout(io.StringIO()):
            result, reward = solver.solve(7)
        logs = result["planner_logs"]
        self.assertEqual(reward, 0.)
        self.assertEqual(solver.env.session, 7)
        self.assertEqual([row["trace_state_context"] for row in logs],
                         ["private actual candidate state 0", "private actual candidate state 1"])
        self.assertEqual([row["trace_next_state_context"] for row in logs],
                         ["private actual candidate state 1", "private actual candidate state 2"])
        self.assertEqual([row["trace_turn_type"] for row in logs], ["tool_plan", "tool_plan"])
        self.assertEqual([row["trace_env_task_score"] for row in logs], [0., .8])
        self.assertTrue(all(row["trace_gt"] == "Fixed offline candidate-state GT." for row in logs))
        self.assertTrue(all("private" not in row["prompt"] and "offline" not in row["prompt"] for row in logs))
        self.assertEqual([row["prompt_token_ids"] for row in logs], [[100], [101]])
        self.assertEqual(solver.planner.logs, [])

    def test_native_exception_is_annotated_external_then_reraised(self):
        solver = self.solver(crash=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "native simulator"):
            solver.solve(7)
        log = solver.planner.logs[-1]
        self.assertTrue(log["trace_external_error"])
        self.assertFalse(log["trace_env_terminal"])
        self.assertFalse(log["trace_scoring_valid"])
        self.assertFalse(log["trace_next_scoring_valid"])

    def test_nontrace_webshop_keeps_original_planner_metadata(self):
        solver = self.solver(trace=False)
        with contextlib.redirect_stdout(io.StringIO()):
            result, _ = solver.solve(7)
        self.assertTrue(all(row["trace_turn_type"] == "base" for row in result["planner_logs"]))
        self.assertTrue(all("trace_gt" not in row for row in result["planner_logs"]))


if __name__ == "__main__":
    unittest.main()
