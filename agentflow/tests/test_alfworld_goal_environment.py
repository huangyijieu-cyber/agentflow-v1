"""CPU-only contracts for full-state ALFWorld scoring and actor isolation."""
import ast
import copy
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import importlib.util
import time
from typing import Optional
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock
import contextlib

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "agentflow/agentflow/envs/alfworld/alfworld_env.py"


def extract(path, name, scope):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class Memory:
    def __init__(self):
        self.items = []

    def reset(self, **kwargs):
        self.items = []

    def store(self, item):
        self.items.append(item)

    def fetch(self, length, **kwargs):
        rows = self.items[-length:]
        return [str(rows)], [len(rows)]

    def __getitem__(self, index):
        return self.items


class Tools:
    @staticmethod
    def normalize_gamefile(path):
        if isinstance(path, list):
            if len(path) != 1:
                raise ValueError("one actual game required")
            path = path[0]
        return Path(path)

    @classmethod
    def load_goal_spec(cls, path):
        return {"gamefile": str(cls.normalize_gamefile(path)), "goal_response": "A clean mug is in a cabinet.",
                "goal_source": "traj_data.json", "goal_task_type": "pick_clean_then_place_in_recep",
                "template_version": "test", "source_sha256": {"game": "abc"},
                "goal_ast": ["isclean", "mug0"], "answer_instruction": "Judge the recorded state.",
                "goal_spec_hash": "stable-spec"}

    @staticmethod
    def normalize_facts(facts):
        return sorted(facts)

    @staticmethod
    def evaluate_goal(spec, facts):
        return "(inreceptacle mug0 cabinet0)" in facts

    @staticmethod
    def render_full_state(spec, facts, *, observation=None, action_history=None):
        return f"Full-state rules and initial facts.\nCurrent facts: {facts}\nObservation: {observation}\nHistory: {action_history}"


class Simulator:
    def __init__(self, *, mismatch=False, crash=False, endless=False):
        self.mismatch, self.crash, self.endless = mismatch, crash, endless
        self.closed = 0

    def reset(self):
        return ["Visible room. Your task is to: put a clean mug in a cabinet."], {
            "extra.gamefile": ["actual/game.tw-pddl"], "admissible_commands": [["place mug"]],
            "won": [False], "facts": [["(isclean mug0)", "(holds agent0 mug0)", "(secret location0)"]],
        }

    def step(self, action):
        if self.crash:
            raise RuntimeError("simulator failed")
        facts = ["(isclean mug0)", "(holds agent0 mug0)", "(secret location0)"]
        if not self.endless:
            facts.append("(inreceptacle mug0 cabinet0)")
        return ["The mug was placed."], [float(not self.endless)], [not self.endless], {
            "extra.gamefile": ["actual/game.tw-pddl"], "admissible_commands": [["place mug"]],
            "won": [not self.endless and not self.mismatch], "facts": [facts], "last_action": ["MoveObject"],
        }

    def close(self):
        self.closed += 1


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.scope = {"os": os, "np": np, "deepcopy": copy.deepcopy, "re": re,
                      "SimpleMemory": Memory, "ALFWORLD_TRACE_INSTRUCTION": "fallback instruction",
                      "_goal_tools": lambda: Tools,
                      "ALFWORLD_TEMPLATE_NO_HIS": "ACTOR {current_observation} {admissible_actions}",
                      "ALFWORLD_TEMPLATE": "ACTOR {task_description} {action_history} {current_observation} {admissible_actions}"}
        for name in ("_environment_scalar", "_environment_flag", "parse_gamefile", "set_gamefile"):
            extract(ENV_PATH, name, self.scope)
        self.cls = extract(ENV_PATH, "AlfWorldEnv", self.scope)

    def env(self, simulator=None, *, trace=True, max_step=50):
        simulator = simulator or Simulator()
        env = self.cls(trace_enabled=trace, max_step=max_step)
        base = NS(game_files=["indexed/game.tw-pddl"], init_env=mock.Mock(return_value=simulator))
        env._env_instances[("train", "all")] = base
        env._env_gamefiles[("train", "all")] = list(base.game_files)
        self.scope["_create_trace_environment"] = mock.Mock(return_value=simulator)
        return env, base

    def reset(self, env, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return env.reset(data_type="train", task_type="all", **kwargs)

    def step(self, env, response):
        with contextlib.redirect_stdout(io.StringIO()):
            return env.step(response)

    def test_hidden_facts_only_enter_scoring_context_and_gt_stays_fixed(self):
        env, _ = self.env()
        actor, info = self.reset(env, question_id=0)
        initial = env.trace_state_context
        self.assertNotIn("secret", actor)
        self.assertIn("secret", initial)
        self.assertEqual(env.trace_gt, "A clean mug is in a cabinet.")
        self.assertEqual(env.trace_answer_instruction, "Judge the recorded state.")
        self.assertFalse(env.trace_goal_check)
        self.assertTrue(env.trace_goal_check_valid)
        actor, reward, done, info = self.step(env, "<action>place mug</action>")
        self.assertTrue(done)
        self.assertTrue(info["trace_goal_check"])
        self.assertTrue(info["trace_goal_check_valid"])
        self.assertEqual(info["trace_state_view"], "full_state")
        self.assertIn("inreceptacle", env.trace_state_context)
        self.assertNotIn("secret", actor)
        self.assertEqual(env.trace_gt, "A clean mug is in a cabinet.")

    def test_actual_saved_gamefile_overrides_question_index(self):
        env, base = self.env()
        self.reset(env, question_id=999999, gamefile=["saved/game.tw-pddl"])
        self.assertEqual(base.game_files, ["saved/game.tw-pddl"])
        # The simulator's reported loaded path remains authoritative for GT.
        self.assertEqual(env.trace_goal_metadata["gamefile"], "actual/game.tw-pddl")

    def test_rejected_action_keeps_real_facts_and_records_no_execution(self):
        env, _ = self.env()
        self.reset(env, question_id=0)
        facts = copy.deepcopy(env.trace_facts)
        _, reward, done, info = self.step(env, "<action>nonexistent action</action>")
        self.assertTrue(done)
        self.assertEqual(env.trace_facts, facts)
        self.assertTrue(info["invalid_action"])
        self.assertFalse(env.trace_action_history[-1]["executed"])
        self.assertEqual(reward, 0)

    def test_format_error_does_not_apply_effects(self):
        env, _ = self.env()
        self.reset(env, question_id=0)
        facts = copy.deepcopy(env.trace_facts)
        _, _, _, info = self.step(env, "place mug")
        self.assertEqual(env.trace_facts, facts)
        self.assertTrue(info["format_error"])
        self.assertNotIn("inreceptacle", env.trace_state_context)

    def test_external_failure_has_unknown_post_state_and_invalid_scoring(self):
        env, _ = self.env(Simulator(crash=True))
        self.reset(env, question_id=0)
        _, _, done, info = self.step(env, "<action>place mug</action>")
        self.assertTrue(done)
        self.assertTrue(info["env_error"])
        self.assertIsNone(info["trace_full_facts"])
        self.assertIsNone(info["trace_goal_check"])
        self.assertFalse(info["trace_scoring_valid"])
        self.assertIn("unavailable", env.trace_state_context)

    def test_budget_failure_preserves_real_facts_and_explicit_truncation(self):
        env, _ = self.env(Simulator(endless=True), max_step=1)
        self.reset(env, question_id=0)
        _, reward, done, info = self.step(env, "<action>place mug</action>")
        self.assertTrue(done)
        self.assertTrue(info["truncated"])
        self.assertFalse(info["terminated"])
        self.assertTrue(info["trace_goal_check_valid"])
        self.assertEqual(reward, 0)

    def test_goal_engine_disagreement_retains_diagnostic_but_invalidates_score(self):
        env, _ = self.env(Simulator(mismatch=True))
        self.reset(env, question_id=0)
        _, _, _, info = self.step(env, "<action>place mug</action>")
        self.assertTrue(info["trace_goal_check"])
        self.assertFalse(info["trace_goal_check_valid"])
        self.assertFalse(info["trace_scoring_valid"])
        self.assertIn("environment won=False", info["trace_goal_diagnostic"])

    def test_missing_requested_facts_fail_closed(self):
        env, _ = self.env()
        self.reset(env, question_id=0)
        with self.assertRaisesRegex(RuntimeError, "facts=True"):
            env._refresh_trace_state({"won": False}, "visible observation")
        with self.assertRaisesRegex(RuntimeError, "empty fact set"):
            env._refresh_trace_state({"facts": [[]], "won": False}, "visible observation")

    def test_legacy_mode_does_not_parse_rules_or_register_extra_state(self):
        env, base = self.env(trace=False)
        self.reset(env, question_id=0)
        base.init_env.assert_called_once_with(batch_size=1)
        self.scope["_create_trace_environment"].assert_not_called()
        self.assertIsNone(env.trace_goal_spec)


class RegistrationTests(unittest.TestCase):
    def test_fact_request_retains_official_wrappers_expert_and_budget(self):
        infos = mock.Mock(side_effect=lambda **kwargs: NS(**kwargs))
        gym = ModuleType("textworld.gym")
        gym.register_games = mock.Mock(return_value="registered")
        gym.make = mock.Mock(return_value="simulator")
        textworld = ModuleType("textworld")
        textworld.__path__ = []
        textworld.EnvInfos, textworld.gym = infos, gym
        official = ModuleType("alfworld.agents.environment.alfred_tw_env")
        official.AlfredDemangler = mock.Mock(return_value="demangler")
        official.AlfredInfos = "infos-wrapper"
        official.AlfredExpert = mock.Mock(return_value="expert")
        function = extract(ENV_PATH, "_create_trace_environment", {})
        base = NS(config={"env": {"type": "AlfredTWEnv", "domain_randomization": True, "expert_type": "handcoded"},
                          "general": {"training_method": "dagger"},
                          "dagger": {"training": {"max_nb_steps_per_episode": 50}}},
                  train_eval="train", game_files=["actual/game.tw-pddl"])
        with mock.patch.dict(sys.modules, {"textworld": textworld, "textworld.gym": gym,
                                          "alfworld.agents.environment.alfred_tw_env": official}):
            self.assertEqual(function(base), "simulator")
        requested = infos.call_args.kwargs
        self.assertTrue(requested["facts"])
        self.assertTrue(requested["last_action"])
        self.assertTrue(requested["won"])
        args = gym.register_games.call_args
        self.assertEqual(args.kwargs["max_episode_steps"], 50)
        self.assertEqual(args.kwargs["wrappers"], ["demangler", "infos-wrapper", "expert"])
        self.assertTrue(args.kwargs["asynchronous"])
        official.AlfredExpert.assert_called_once_with(expert_type="handcoded")


class SolverStateTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "agentflow/agentflow/trace/execution.py"
        spec = importlib.util.spec_from_file_location("alfworld_execution_env_test", path)
        execution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(execution)
        self.scope = {"Optional": Optional, "deepcopy": copy.deepcopy, "time": time,
                      "log_info": lambda *args: None,
                      "annotate_alfworld_step": execution.annotate_alfworld_step}
        self.cls = extract(ROOT / "agentflow/agentflow/solver.py", "Solver", self.scope)

    def solver(self, task="alfworld", crash=False):
        class Env:
            trace_enabled = True
            trace_gt = "A clean mug is in a cabinet."
            trace_answer_instruction = "Judge the current full state."
            trace_goal_metadata = {"template_version": "v1"}
            trace_goal_check = False
            trace_goal_check_valid = True
            trace_scoring_valid = True
            trace_state_context = "private full state 0"

            def reset(self, **kwargs):
                self.index = 0
                return "visible state 0", {"anchor": "visible state 0"}

            def step(self, response):
                if crash:
                    raise RuntimeError("environment failed")
                self.index += 1
                self.trace_state_context = f"private full state {self.index}"
                done = self.index == 2
                self.trace_goal_check = done
                return f"visible state {self.index}", float(done), done, {
                    "anchor": f"visible state {self.index}", "trace_env_action": "act",
                    "won": done, "terminated": done, "truncated": False,
                    "trace_goal_check": done, "trace_goal_check_valid": True,
                    "trace_scoring_valid": True,
                }

        class Planner:
            def __init__(self):
                self.logs = []

            def generate_base_response(self, question, image, max_tokens):
                response = "<action>act</action>"
                self.logs.append({"prompt": question, "response": response, "trace_turn_type": "base",
                                  "prompt_token_ids": [100 + len(self.logs)],
                                  "response_token_ids": [200], "finish_reason": "stop"})
                return response

        solver = object.__new__(self.cls)
        solver.task, solver.env, solver.planner = task, Env(), Planner()
        solver.output_types, solver.max_tokens, solver.verbose = ["base"], 32, False
        return solver

    def test_production_logs_snapshot_pre_action_private_state_without_actor_leak(self):
        solver = self.solver()
        with contextlib.redirect_stdout(io.StringIO()):
            result, reward = solver.solve("train_all_alfworld_0")
        logs = result["planner_logs"]
        self.assertEqual([item["trace_state_context"] for item in logs],
                         ["private full state 0", "private full state 1"])
        self.assertEqual([item["trace_turn_type"] for item in logs], ["tool_plan", "base"])
        self.assertEqual([item["trace_goal_check_before"] for item in logs], [False, False])
        self.assertEqual([item["trace_goal_check"] for item in logs], [False, True])
        self.assertEqual([item["prompt_token_ids"] for item in logs], [[100], [101]])
        self.assertTrue(all(item["trace_gt"] == "A clean mug is in a cabinet." for item in logs))
        self.assertTrue(all("private" not in item["prompt"] and "clean mug" not in item["prompt"] for item in logs))
        self.assertEqual(reward, 1)
        self.assertEqual(solver.planner.logs, [])

    def test_non_alfworld_environment_keeps_existing_base_logs(self):
        solver = self.solver(task="webshop")
        with contextlib.redirect_stdout(io.StringIO()):
            result, _ = solver.solve("session-id")
        self.assertTrue(all(item["trace_turn_type"] == "base" for item in result["planner_logs"]))
        self.assertTrue(all("trace_gt" not in item for item in result["planner_logs"]))

    def test_uncaught_environment_failure_is_not_recorded_as_success(self):
        solver = self.solver(crash=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "environment failed"):
            solver.solve("train_all_alfworld_0")
        log = solver.planner.logs[0]
        self.assertTrue(log["trace_external_error"])
        self.assertFalse(log["trace_env_terminal"])
        self.assertEqual(log["trace_state_context"], "private full state 0")


if __name__ == "__main__":
    unittest.main()
