"""CPU-only native adapter contracts, including scorer/actor and train/val isolation."""
import ast
import contextlib
import copy
import io
import os
from pathlib import Path
import re
from types import SimpleNamespace as NS
import unittest
from unittest import mock

import numpy as np

import test_webshop_goal_runtime as runtime
from agentflow.trace import webshop_goal as goals, webshop_state as states


ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / "agentflow/agentflow/envs/webshop/webshop_env.py"


def extract(path, name, scope):
    tree = ast.parse(Path(path).read_text())
    node = next(node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class NativeEnvironment:
    """Models native reset's session reuse and actual selected-option storage."""
    def __init__(self, fixture, score=.75):
        products = [{"asin": f"FIXTURE{i}", "Title": f"Actual candidate {i}",
                     "Attributes": ["insulated", "reusable"], "BulletPoints": ["steel"],
                     "Description": "A travel cup.", "options": {"color": ["blue", "black"]}}
                    for i in range(2)]
        self.server = NS(all_products=products, product_item_dict={p["asin"]: p for p in products},
                         product_prices={"FIXTURE0": 999., "FIXTURE1": 888.},
                         goals=copy.deepcopy(fixture.manifest["goals"]),
                         user_sessions={}, assigned_instruction_text="old instruction")
        self.kwargs = {"num_products": 2}
        self.file_path = fixture.root / "products.json"
        self.session, self.session_prefix = None, None
        self.kind, self.turn, self.score = "search", 0, score
        self.reset_sessions = []
        self.closed = 0
        self.crash = self.broken_state = False

    @property
    def state(self):
        if self.broken_state:
            return {}
        html = '<div id="instruction-text">Hidden task instruction.</div>'
        if self.kind == "search_results":
            html += ''.join(f'<div class="searched-product"><a class="product-link">FIXTURE{i}</a>'
                            f'<h4 class="product-title">Actual candidate {i}</h4></div>' for i in range(2))
        elif self.kind == "done":
            html += f'<p>Reward: {self.score}</p>'
        else:
            html += f'<p>Native page {self.kind}, observation {self.turn}.</p>'
        return {"url": f"http://fixture/{self.kind}/{self.session}", "html": html}

    @property
    def observation(self):
        goal = self.server.user_sessions[self.session]["goal"]
        return f'WebShop [SEP] Instruction: [SEP] {goal["instruction_text"]} [SEP] Native {self.kind} {self.turn}'

    def get_available_actions(self):
        clickables = {"search": [], "search_results": ["fixture0", "fixture1"],
                      "item_page": ["blue", "black", "buy now", "< prev"], "done": []}
        return {"has_search_bar": self.kind == "search", "clickables": clickables[self.kind]}

    def reset(self, session=None):
        self.reset_sessions.append(session)
        self.session = str(session if session is not None else 0)
        self.server.user_sessions.setdefault(self.session, {
            "goal": self.server.goals[0 if session is None else session], "done": False})
        self.server.user_sessions[self.session].update(
            {"keywords": None, "page": None, "asin": None, "options": {}})
        self.kind, self.turn = "search", 0
        return self.observation, None

    def step(self, action):
        if self.crash:
            raise RuntimeError("native simulator unavailable")
        self.turn += 1
        session = self.server.user_sessions[self.session]
        reward, done = 0., False
        if re.fullmatch(r"search\[.+\]", action):
            session.update({"keywords": [action[7:-1]], "page": 1, "asin": None, "options": {}})
            self.kind = "search_results"
        elif self.kind == "search_results" and action in ("click[fixture0]", "click[fixture1]"):
            session["asin"] = action[6:-1].upper()
            self.kind = "item_page"
        elif self.kind == "item_page" and action in ("click[blue]", "click[black]"):
            session["options"]["color"] = action[6:-1]
        elif self.kind == "item_page" and action == "click[buy now]":
            self.kind, reward, done = "done", self.score, True
            session.update({"done": True, "reward": reward})
        return self.observation, reward, done, None

    def close(self):
        self.closed += 1


class WebShopEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.fixture = runtime.FrozenRuntimeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.manifest_path = self.fixture.base / "frozen.json"
        self.manifest_path.write_text(goals.canonical_json(self.fixture.manifest))
        scope = {"os": os, "re": re, "np": np, "deepcopy": copy.deepcopy,
                 "file_path": "/unused/default-root", "WEBSHOP_TRACE_INSTRUCTION": "Describe real candidate facts.",
                 "_goal_tools": lambda: goals, "_state_tools": lambda: states}
        memory_scope = {"BaseMemory": object, "List": list, "Dict": dict, "Any": object, "Tuple": tuple}
        scope["SimpleMemory"] = extract(ROOT / "agentflow/agentflow/envs/memory/memory.py", "SimpleMemory", memory_scope)
        exec(compile(ENV_PATH.with_name("webshop_template.py").read_text(), "templates", "exec"), scope)
        extract(ENV_PATH, "_parse_trace_history_length", scope)
        self.cls = extract(ENV_PATH, "WebShopEnv", scope)
        self.factory = mock.Mock(side_effect=lambda root: NativeEnvironment(self.fixture))
        scope["_make_native_environment"] = self.factory
        self.scope = scope

    def env(self, *, trace=True, bind=True, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            env = self.cls(trace_enabled=trace, trace_manifest_path=self.manifest_path,
                           webshop_root=self.fixture.root, **kwargs)
        if bind:
            env.set_trace_task(runtime.fixture_row(self.fixture.manifest, 1))
        return env

    def reset(self, env, session=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return env.reset(session=session)

    def step(self, env, action):
        with contextlib.redirect_stdout(io.StringIO()):
            return env.step(action)

    def test_lazy_reset_restores_frozen_catalog_and_binds_without_success_label(self):
        env = self.env()
        self.factory.assert_not_called()
        actor, info = self.reset(env)
        self.factory.assert_called_once_with(str(self.fixture.root))
        self.assertEqual(env.env.server.product_prices, self.fixture.manifest["product_prices"])
        self.assertIsNone(env.env.server.assigned_instruction_text)
        self.assertTrue(info["trace_goal_check_valid"])
        self.assertTrue(info["trace_scoring_valid"])
        self.assertIsNone(env.trace_goal_check)
        self.assertEqual(env.trace_state_view, "webshop_state")
        self.assertIsNone(env.trace_snapshot["current_item"])
        self.assertNotIn(env.trace_gt, actor)
        self.assertNotIn("REFERENCE TITLE", actor)

    def test_actor_prompts_and_binary_rewards_match_without_scorer(self):
        traced, baseline = self.env(), self.env(trace=False, bind=False)
        baseline.env = NativeEnvironment(self.fixture)
        baseline.env.server.product_prices = copy.deepcopy(self.fixture.manifest["product_prices"])
        actor, _ = self.reset(traced)
        original_actor, _ = self.reset(baseline)
        self.assertEqual(actor, original_actor)
        gt = traced.trace_gt
        for action in ("search[cup]", "click[fixture1]", "click[blue]", "click[buy now]"):
            response = f"<action>{action}</action>"
            actor, reward, done, info = self.step(traced, response)
            original_actor, original_reward, original_done, _ = self.step(baseline, response)
            self.assertEqual((actor, reward, done), (original_actor, original_reward, original_done))
            self.assertEqual(traced.trace_gt, gt)
            self.assertEqual(traced.trace_full_actor_history, traced.memory[0])
        self.assertEqual(reward, 0.)
        self.assertEqual(info["task_score"], .75)
        self.assertTrue(info["terminated"])
        self.assertFalse(info["truncated_by_max_step"])
        self.assertEqual(traced.trace_terminal_snapshot["current_item"]["candidate_id"], "FIXTURE1")
        self.assertEqual(traced.trace_terminal_snapshot["selected_options"], {"color": "blue"})
        self.assertNotIn("Reward", traced.trace_state_context)
        self.assertNotIn("0.75", traced.trace_state_context)
        info["trace_terminal_snapshot"]["selected_options"]["color"] = "mutated"
        traced.env.server.user_sessions["1"]["options"]["color"] = "mutated native"
        self.assertEqual(traced.trace_terminal_snapshot["selected_options"], {"color": "blue"})

    def test_repeat_question_removes_previous_done_reward_and_native_memory(self):
        env = self.env()
        self.reset(env)
        for action in ("search[cup]", "click[fixture1]", "click[buy now]"):
            self.step(env, f"<action>{action}</action>")
        self.assertTrue(env.env.server.user_sessions["1"]["done"])
        _, info = self.reset(env)
        session = env.env.server.user_sessions["1"]
        self.assertFalse(session["done"])
        self.assertNotIn("reward", session)
        self.assertEqual(env.memory[0], [])
        self.assertIsNone(env.trace_terminal_snapshot)
        self.assertIsNone(info["trace_snapshot"]["current_item"])
        self.factory.assert_called_once()

    def test_question_is_a_strict_integer_before_native_reset(self):
        for value in (True, np.bool_(False), "1", 1., 1.5, [], None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.reset(self.env(), value)
        self.factory.assert_not_called()
        env = self.env()
        self.reset(env, np.int64(1))
        self.assertIs(type(env.env.reset_sessions[-1]), int)
        for value in (-1, 2):
            with self.assertRaises(ValueError):
                self.reset(env, value)

    def test_validation_freezes_same_prices_without_gt_and_uses_separate_tables(self):
        train = self.env()
        validation = self.env(bind=False)
        validation.trace_enabled = False
        _, train_info = self.reset(train)
        _, validation_info = self.reset(validation)
        self.assertTrue(train.trace_enabled)
        self.assertTrue(validation.trace_manifest_enabled)
        self.assertNotIn("trace_gt", validation_info)
        self.assertEqual(train.env.server.product_prices, validation.env.server.product_prices)
        self.assertIsNot(train.env.server.product_prices, validation.env.server.product_prices)
        self.assertEqual(train.env.server.goals, validation.env.server.goals)
        validation.env.server.goals[1]["attributes"].append("validation only")
        self.assertNotIn("validation only", train.env.server.goals[1]["attributes"])
        self.assertTrue(train_info["trace_scoring_valid"])
        self.assertIsNone(validation.trace_task)

    def test_nontrace_ignores_manifest_and_does_not_require_converted_rows(self):
        env = self.env(trace=False, bind=False)
        env.trace_manifest_path = "/does/not/exist.json"
        self.reset(env)
        self.assertEqual(env.env.server.product_prices["FIXTURE1"], 888.)
        self.assertIsNone(env._trace_manifest)
        with mock.patch.dict(os.environ, {"ALGORITHM": "grpo", "TRACE_ENABLED": "true"}):
            with contextlib.redirect_stdout(io.StringIO()):
                default = self.cls()
        self.assertFalse(default.trace_enabled)

    def test_scorer_history_is_native_pre_action_memory_with_independent_window(self):
        for history_length in (4, 2, 0, None):
            with self.subTest(history_length=history_length):
                env = self.env(trace_history_length=history_length)
                self.reset(env)
                for index in range(6):
                    actor, _, _, info = self.step(env, f"<action>search[cup{index}]</action>")
                self.assertEqual(env.history_length, 4)
                self.assertIn("Observation 3:", actor)
                history = env.trace_state_context.split("Recent native pre-action observations and parsed actions:")
                if history_length == 0:
                    self.assertEqual(len(history), 1)
                else:
                    self.assertIn("Observation 6:", history[-1])
                    self.assertIn(f"Observation {1 if history_length is None else 7 - history_length}:", history[-1])
                    if history_length is not None:
                        self.assertNotIn("Observation 1:", history[-1])
                self.assertEqual(info["trace_full_actor_history"], env.memory[0])
                self.assertEqual(env.memory[0][-1]["action"], "search[cup5]")
                self.assertNotIn("<action>", str(env.memory[0]))

    def test_noop_format_quality_and_budget_stop_preserve_actual_transition(self):
        env = self.env(max_step=1)
        self.reset(env)
        before = env.trace_state_context
        _, reward, done, info = self.step(env, "click[unknown]")
        self.assertTrue(done)
        self.assertEqual(reward, 0.)
        self.assertEqual(info["task_score"], 0.)
        self.assertFalse(info["terminated"])
        self.assertTrue(info["truncated_by_max_step"])
        self.assertTrue(info["format_error"])
        self.assertTrue(info["invalid_action"])
        self.assertFalse(info.get("env_error", False))
        self.assertTrue(info["trace_scoring_valid"])
        self.assertNotEqual(info["trace_state_context"], before)

    def test_native_errors_propagate_and_bad_state_never_becomes_a_zero_value(self):
        env = self.env()
        self.reset(env)
        env.env.crash = True
        with self.assertRaisesRegex(RuntimeError, "native simulator"):
            self.step(env, "<action>search[cup]</action>")
        self.assertEqual(env.memory[0], [])
        env.env.crash, env.env.broken_state = False, True
        _, reward, done, info = self.step(env, "<action>search[cup]</action>")
        self.assertFalse(done)
        self.assertEqual(reward, 0.)
        self.assertFalse(info["trace_scoring_valid"])
        self.assertIsNone(info["trace_state_context"])
        self.assertFalse(info.get("env_error", False))

    def test_binding_mismatch_is_rejected_before_actor_generation(self):
        env = self.env()
        env.trace_task["result"] = "wrong target"
        with self.assertRaisesRegex(goals.ConversionError, "result"):
            self.reset(env)


if __name__ == "__main__":
    unittest.main()
