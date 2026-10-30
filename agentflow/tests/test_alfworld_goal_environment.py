"""CPU-only contracts for full-state ALFWorld scoring and actor isolation."""
import ast
import copy
import io
import os
import pickle
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
    def read_dataset_gamefile(store_path, data_type, task_type, question_id, games_root):
        return None

    @staticmethod
    def normalize_gamefile(path):
        if isinstance(path, list):
            if len(path) != 1:
                raise ValueError("one actual game required")
            path = path[0]
        return Path(path)

    @classmethod
    def read_goal_spec(cls, store_path, path):
        return {"gamefile": str(cls.normalize_gamefile(path)), "goal_response": "A clean mug is in a cabinet.",
                "goal_source": "traj_data.json", "goal_task_type": "pick_clean_then_place_in_recep",
                "template_version": "test", "source_sha256": {"game": "abc"},
                "goal_ast": ["isclean", "mug0"], "answer_instruction": "Judge the recorded state.",
                "goal_spec_hash": "stable-spec", "goal_store_path": store_path,
                "goal_store_id": "persisted-store", "goal_store_entry": "train/actual/game.tw-pddl",
                "goal_store_spec_sha256": "persisted-spec"}

    @staticmethod
    def load_goal_spec(path):
        raise AssertionError("Training must never convert goals")

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
            "extra.trace_raw_facts": [["(isclean mug0)", "(holds agent0 mug0)", "(secret location0)"]],
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
            "extra.trace_raw_facts": [facts],
        }

    def close(self):
        self.closed += 1


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.scope = {"os": os, "np": np, "deepcopy": copy.deepcopy, "re": re,
                      "data_path": "/saved/alfworld-data",
                      "SimpleMemory": Memory, "ALFWORLD_TRACE_INSTRUCTION": "fallback instruction",
                      "_goal_tools": lambda: Tools, "_saved_goal_tools": lambda: Tools,
                      "ALFWORLD_TEMPLATE_NO_HIS": "ACTOR {current_observation} {admissible_actions}",
                      "ALFWORLD_TEMPLATE": "ACTOR {task_description} {action_history} {current_observation} {admissible_actions}"}
        for name in ("_environment_scalar", "_environment_flag", "parse_gamefile", "set_gamefile"):
            extract(ENV_PATH, name, self.scope)
        self.cls = extract(ENV_PATH, "AlfWorldEnv", self.scope)

    def env(self, simulator=None, *, trace=True, max_step=50):
        simulator = simulator or Simulator()
        env = self.cls(trace_enabled=trace, max_step=max_step, trace_goals_path="/saved/goals.sqlite")
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
        self.assertEqual(env.trace_goal_metadata["goal_store_id"], "persisted-store")
        self.assertEqual(env.trace_goal_metadata["goal_store_path"], "/saved/goals.sqlite")
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

    def test_separate_training_sessions_only_read_the_saved_store(self):
        with mock.patch.object(Tools, "read_goal_spec", wraps=Tools.read_goal_spec) as read, \
             mock.patch.object(Tools, "load_goal_spec", side_effect=AssertionError("conversion forbidden")) as convert:
            first, _ = self.env()
            self.reset(first, question_id=0)
            second, _ = self.env()
            self.reset(second, question_id=0)
        self.assertEqual(read.call_args_list, [mock.call("/saved/goals.sqlite", ["actual/game.tw-pddl"])] * 2)
        convert.assert_not_called()
        self.assertEqual(first.trace_goal_spec, second.trace_goal_spec)

    def test_restarted_environment_resets_load_the_same_persisted_file_without_conversion(self):
        import test_alfworld_goal_conversion as fixtures
        store_path = ROOT / "agentflow/agentflow/trace/alfworld_goal_store.py"
        module_spec = importlib.util.spec_from_file_location("saved_goal_environment_integration", store_path)
        store = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(store)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = fixtures.GoalTests()
            builder.root = root / "games"
            builder.root.mkdir()
            gamefile = builder.create()
            output = root / "goals.sqlite"
            store.build_goal_store(builder.root, output)
            original = output.read_bytes()
            self.scope["_saved_goal_tools"] = lambda: store
            self.scope["_goal_tools"] = lambda: fixtures.converter

            class SavedSimulator(Simulator):
                def reset(self):
                    observation, info = super().reset()
                    info["extra.gamefile"] = [str(gamefile)]
                    info["facts"] = [fixtures.INITIAL]
                    info["extra.trace_raw_facts"] = [fixtures.INITIAL]
                    return observation, info

            with mock.patch.object(store, "_converter", side_effect=AssertionError("runtime conversion forbidden")) as convert:
                first, _ = self.env(SavedSimulator())
                first.trace_goals_path = str(output)
                self.reset(first, question_id=0)
                second, _ = self.env(SavedSimulator())
                second.trace_goals_path = str(output)
                self.reset(second, question_id=0)
            convert.assert_not_called()
            self.assertEqual(first.trace_gt, second.trace_gt)
            self.assertEqual(first.trace_goal_metadata["goal_store_id"], second.trace_goal_metadata["goal_store_id"])
            self.assertTrue(first.trace_goal_check_valid)
            self.assertTrue(second.trace_goal_check_valid)
            self.assertEqual(output.read_bytes(), original)

    def test_unbound_row_and_missing_store_fail_before_environment_registration(self):
        for goal_setting in (None, "   "):
            settings = {} if goal_setting is None else {"ALFWORLD_TRACE_GOALS": goal_setting}
            with mock.patch.dict(os.environ, settings, clear=True):
                env = self.cls(trace_enabled=True)
            env._init_env = mock.Mock(side_effect=AssertionError("must fail before environment initialization"))
            with self.assertRaisesRegex(ValueError, "prepared Parquet row"):
                self.reset(env, question_id=0)
            env._init_env.assert_not_called()

    def test_prepared_row_result_loads_without_sqlite_or_actor_gt_injection(self):
        selected = Path("/saved/alfworld-data/json_2.1.1/train/row-game/game.tw-pddl")
        row = {"question": "train_all_alfworld_0", "result": "A clean mug is in a cabinet.",
               "alfworld_goal_spec": "saved-spec", "alfworld_goal_spec_sha256": "saved-hash",
               "alfworld_game_key": "train/row-game/game.tw-pddl", "alfworld_goal_row": "saved-row"}

        class RowSimulator(Simulator):
            def reset(self):
                observations, info = super().reset()
                info["extra.gamefile"] = [str(selected)]
                return observations, info

        path_reader = mock.Mock(return_value=selected)
        spec_reader = mock.Mock(side_effect=lambda task, path: Tools.read_goal_spec("parquet.result", path))
        self.scope["_parquet_goal_tools"] = lambda: NS(
            parquet_task_gamefile=path_reader, read_parquet_task_goal=spec_reader)
        env, base = self.env(RowSimulator())
        env.trace_goals_path = ""
        env.set_trace_task(row)
        row["result"] = "mutated by caller"
        with mock.patch.object(Tools, "read_goal_spec", side_effect=AssertionError("SQLite forbidden")), \
             mock.patch.object(Tools, "read_dataset_gamefile", side_effect=AssertionError("SQLite mapping forbidden")):
            # Freeze the expected spec before forbidding the legacy reader.
            spec_reader.side_effect = None
            spec_reader.return_value = {"goal_response": "A clean mug is in a cabinet.",
                                       "answer_instruction": "Judge the recorded state.", "goal_spec_hash": "row-spec"}
            actor, _ = self.reset(env, question_id=0)
        self.assertEqual(base.game_files, [str(selected)])
        self.assertEqual(env.trace_gt, "A clean mug is in a cabinet.")
        self.assertEqual(env.trace_goal_metadata["goal_source"], "parquet.result")
        self.assertEqual(env.trace_goal_metadata["parquet_goal_spec_sha256"], "saved-hash")
        self.assertNotIn(env.trace_gt, actor)
        path_reader.assert_called_once_with(env.trace_task, "/saved/alfworld-data/json_2.1.1")
        spec_reader.assert_called_once_with(env.trace_task, [str(selected)])

    def test_bound_parquet_question_must_match_all_reset_parameters(self):
        row = {"question": "train_all_alfworld_0", "result": "A clean mug is in a cabinet."}
        for changed in ({"question_id": 1}, {"data_type": "val_seen"}, {"task_type": "look"}):
            env, _ = self.env()
            env.trace_goals_path = ""
            env.set_trace_task(row)
            env._init_env = mock.Mock(side_effect=AssertionError("must reject stale row before init"))
            arguments = {"data_type": "train", "task_type": "all", "question_id": 0, **changed}
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "reset parameters differ"):
                env.reset(**arguments)
            env._init_env.assert_not_called()

    def test_prepared_parquet_survives_new_environment_sessions_without_a_store(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import test_alfworld_goal_conversion as fixtures
        module_spec = importlib.util.spec_from_file_location(
            "prepared_parquet_environment_integration", ROOT / "agentflow/agentflow/trace/alfworld_parquet.py")
        parquet = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(parquet)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            builder = fixtures.GoalTests()
            builder.root = root / "json_2.1.1"
            builder.root.mkdir()
            game = builder.create()
            source, output = root / "source.parquet", root / "prepared.parquet"
            pq.write_table(pa.table({"id": [1], "question": ["train_all_alfworld_0"], "result": [""]}), source)
            parquet.prepare_dataset_parquet(source, builder.root, output, expected_rows=1,
                                             catalogue_loader=lambda *args: [game])
            row = pq.read_table(output).to_pylist()[0]
            original = output.read_bytes()
            self.scope.update({"_parquet_goal_tools": lambda: parquet, "_goal_tools": lambda: fixtures.converter,
                               "data_path": str(root)})

            class RowSimulator(Simulator):
                def reset(self):
                    observation, info = super().reset()
                    info["extra.gamefile"] = [str(game)]
                    info["facts"] = [fixtures.INITIAL]
                    info["extra.trace_raw_facts"] = [fixtures.INITIAL]
                    return observation, info

            with mock.patch.dict(os.environ, {}, clear=True), \
                 mock.patch.object(parquet._store_tools(), "_converter", side_effect=AssertionError("conversion forbidden")), \
                 mock.patch.object(parquet._store_tools(), "read_goal_spec", side_effect=AssertionError("SQLite forbidden")):
                sessions = []
                for _ in range(2):
                    env, base = self.env(RowSimulator())
                    env.trace_goals_path = ""
                    env.set_trace_task(row)
                    env._env_gamefiles[("train", "all")] = ["reordered-machine/wrong/game.tw-pddl"]
                    actor, _ = self.reset(env, question_id=0)
                    self.assertEqual(base.game_files, [str(game)])
                    self.assertEqual(env.trace_gt, row["result"])
                    self.assertEqual(env.trace_goal_metadata["goal_source"], "parquet.result")
                    self.assertNotIn(row["result"], actor)
                    sessions.append(env.trace_goal_spec)
            self.assertEqual(sessions[0], sessions[1])
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(root.rglob("*.sqlite")), [])

    def test_prepared_row_with_empty_result_is_rejected_before_environment_init(self):
        module_spec = importlib.util.spec_from_file_location(
            "invalid_parquet_environment_integration", ROOT / "agentflow/agentflow/trace/alfworld_parquet.py")
        parquet = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(parquet)
        self.scope["_parquet_goal_tools"] = lambda: parquet
        env, _ = self.env()
        env.trace_goals_path = ""
        env.set_trace_task({"question": "train_all_alfworld_0", "result": ""})
        env._env_instances.clear()
        env._init_env = mock.Mock(side_effect=AssertionError("must reject before init"))
        with self.assertRaisesRegex(ValueError, "result is empty"):
            self.reset(env, question_id=0)
        env._init_env.assert_not_called()

    def test_configured_store_and_explicit_probe_path_are_supported(self):
        with mock.patch.dict(os.environ, {"ALFWORLD_TRACE_GOALS": "/shared/train-goals.sqlite"}, clear=True):
            from_env = self.cls(trace_enabled=True)
            explicit = self.cls(trace_enabled=True, trace_goals_path=Path("/probe/frozen-goals.sqlite"))
        self.assertEqual(from_env.trace_goals_path, "/shared/train-goals.sqlite")
        self.assertEqual(explicit.trace_goals_path, "/probe/frozen-goals.sqlite")

    def test_missing_or_stale_store_has_no_online_conversion_fallback(self):
        for failure in (FileNotFoundError("missing saved goal store"), ValueError("stale saved source hash")):
            env, _ = self.env()
            with mock.patch.object(Tools, "read_goal_spec", side_effect=failure), \
                 mock.patch.object(Tools, "load_goal_spec", side_effect=AssertionError("conversion forbidden")) as convert, \
                 self.assertRaises(type(failure)):
                self.reset(env, question_id=0)
            convert.assert_not_called()
            self.assertIsNone(env.trace_goal_spec)

    def test_trace_disabled_does_not_require_or_read_a_goal_store(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            env = self.cls(trace_enabled=False)
        self.assertEqual(env.trace_goals_path, "")
        with mock.patch.object(Tools, "read_goal_spec") as read:
            # Normal non-TRACE reset contract is independently covered below.
            env, _ = self.env(trace=False)
            self.reset(env, question_id=0)
        read.assert_not_called()

    def test_actual_saved_gamefile_overrides_question_index(self):
        env, base = self.env()
        self.reset(env, question_id=999999, gamefile=["saved/game.tw-pddl"])
        self.assertEqual(base.game_files, ["saved/game.tw-pddl"])
        # The simulator's reported loaded path remains authoritative for GT.
        self.assertEqual(env.trace_goal_metadata["gamefile"], "actual/game.tw-pddl")

    def test_saved_dataset_mapping_takes_priority_over_changed_catalogue(self):
        selected = Path("/saved/alfworld-data/json_2.1.1/train/frozen/game.tw-pddl")

        class MappedSimulator(Simulator):
            def reset(self):
                observations, info = super().reset()
                info["extra.gamefile"] = [str(selected)]
                return observations, info

        env, base = self.env(MappedSimulator())
        env._env_gamefiles[("train", "all")] = ["wrong/new-order/game.tw-pddl"]
        with mock.patch.object(Tools, "read_dataset_gamefile", return_value=selected) as read:
            self.reset(env, question_id=3304)
        self.assertEqual(base.game_files, [str(selected)])
        read.assert_called_once_with("/saved/goals.sqlite", "train", "all", 3304,
                                     "/saved/alfworld-data/json_2.1.1")
        self.assertEqual(env.trace_goal_metadata["gamefile"], str(selected))

    def test_missing_dataset_question_does_not_fall_back_to_catalogue(self):
        env, _ = self.env()
        with mock.patch.object(Tools, "read_dataset_gamefile", side_effect=ValueError("question is not selected")):
            with self.assertRaisesRegex(ValueError, "not selected"):
                self.reset(env, question_id=0)
        self.scope["_create_trace_environment"].assert_not_called()

    def test_explicit_gamefile_does_not_reinterpret_worker_local_index(self):
        env, base = self.env()
        with mock.patch.object(Tools, "read_dataset_gamefile", side_effect=AssertionError("must not look up local index")) as read:
            self.reset(env, question_id=0, gamefile="saved/actual-selection/game.tw-pddl")
        read.assert_not_called()
        self.assertEqual(base.game_files, ["saved/actual-selection/game.tw-pddl"])

    def test_dataset_mapping_requires_actual_loaded_game_to_match(self):
        env, _ = self.env()
        with mock.patch.object(Tools, "read_dataset_gamefile", return_value=Path("/saved/expected/game.tw-pddl")):
            with self.assertRaisesRegex(ValueError, "different game"):
                self.reset(env, question_id=0)

    def test_dataset_scoped_sqlite_freezes_game_across_a_reordered_runtime_catalogue(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import test_alfworld_goal_conversion as fixtures
        module_spec = importlib.util.spec_from_file_location(
            "dataset_goal_environment_integration", ROOT / "agentflow/agentflow/trace/alfworld_goal_store.py")
        store = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(store)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = fixtures.GoalTests()
            builder.root = root / "json_2.1.1"
            builder.root.mkdir()
            gamefile = builder.create()
            dataset = root / "train.parquet"
            pq.write_table(pa.table({"id": [999], "question": ["train_all_alfworld_0"]}), dataset)
            output = root / "train-goals.sqlite"
            store.build_goal_store(builder.root, output, dataset=dataset, expected_rows=1,
                                   catalogue_loader=lambda *args: [gamefile])
            self.scope.update({"_saved_goal_tools": lambda: store, "_goal_tools": lambda: fixtures.converter,
                               "data_path": str(root)})

            class SavedSimulator(Simulator):
                def reset(self):
                    observation, info = super().reset()
                    info["extra.gamefile"] = [str(gamefile)]
                    info["facts"] = [fixtures.INITIAL]
                    info["extra.trace_raw_facts"] = [fixtures.INITIAL]
                    return observation, info

            env, base = self.env(SavedSimulator())
            env.trace_goals_path = str(output)
            env._env_gamefiles[("train", "all")] = ["other-machine/reordered/game.tw-pddl"]
            with mock.patch.object(store, "_converter", side_effect=AssertionError("conversion forbidden")):
                self.reset(env, question_id=0)
            self.assertEqual(base.game_files, [str(gamefile.resolve())])
            self.assertEqual(env.trace_goal_metadata["goal_store_selection_mode"], "dataset")
            self.assertEqual(env.trace_goal_metadata["goal_store_dataset_rows"], 1)
            self.assertTrue(env.trace_goal_check_valid)

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
        # Public facts must never be used as a fallback, even if requested.
        with self.assertRaisesRegex(RuntimeError, "extra.trace_raw_facts"):
            env._refresh_trace_state({"facts": [["(isclean mug0)"]], "won": False}, "visible observation")
        with self.assertRaisesRegex(RuntimeError, "empty fact set"):
            env._refresh_trace_state({"extra.trace_raw_facts": [[]], "won": False}, "visible observation")

    def test_public_display_names_are_ignored_on_reset_and_step(self):
        class DemangledSimulator(Simulator):
            def reset(self):
                observations, info = super().reset()
                info["facts"] = [[NS(name="atlocation", arguments=[NS(name="agent1"), NS(name="loc 7")])]]
                return observations, info

            def step(self, action):
                observations, reward, done, info = super().step(action)
                info["facts"] = [[NS(name="inreceptacle", arguments=[NS(name="mug 2"), NS(name="cabinet 1")])]]
                return observations, reward, done, info

        env, _ = self.env(DemangledSimulator())
        self.reset(env, question_id=0)
        self.assertTrue(env.trace_goal_check_valid)
        self.assertNotIn("loc 7", str(env.trace_facts))
        _, _, done, info = self.step(env, "<action>place mug</action>")
        self.assertTrue(done)
        self.assertTrue(info["trace_goal_check"])
        self.assertTrue(info["trace_scoring_valid"])

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
        trace_wrapper = ModuleType("agentflow.envs.alfworld.trace_wrapper")
        trace_wrapper.AlfWorldTraceFacts = "raw-facts-wrapper"
        function = extract(ENV_PATH, "_create_trace_environment", {})
        base = NS(config={"env": {"type": "AlfredTWEnv", "domain_randomization": True, "expert_type": "handcoded"},
                          "general": {"training_method": "dagger"},
                          "dagger": {"training": {"max_nb_steps_per_episode": 50}}},
                  train_eval="train", game_files=["actual/game.tw-pddl"])
        with mock.patch.dict(sys.modules, {"textworld": textworld, "textworld.gym": gym,
                                          "agentflow.envs.alfworld.trace_wrapper": trace_wrapper,
                                          "alfworld.agents.environment.alfred_tw_env": official}):
            self.assertEqual(function(base), "simulator")
        requested = infos.call_args.kwargs
        self.assertTrue(requested["facts"])
        self.assertTrue(requested["last_action"])
        self.assertTrue(requested["won"])
        self.assertIn("trace_raw_facts", requested["extras"])
        args = gym.register_games.call_args
        self.assertEqual(args.kwargs["max_episode_steps"], 50)
        self.assertEqual(args.kwargs["wrappers"], ["demangler", "infos-wrapper", "expert", "raw-facts-wrapper"])
        self.assertTrue(args.kwargs["asynchronous"])
        official.AlfredExpert.assert_called_once_with(expert_type="handcoded")


class TraceFactsWrapperTests(unittest.TestCase):
    def wrapper(self, simulator):
        class Wrapper:
            def __init__(self, env):
                self.env = env

            def reset(self):
                return self.env.reset()

            def step(self, command):
                return self.env.step(command)

        cls = extract(ENV_PATH.with_name("trace_wrapper.py"), "AlfWorldTraceFacts", {"Wrapper": Wrapper})
        return cls(simulator)

    def test_raw_identity_survives_display_randomization_batch_and_serialization(self):
        import test_alfworld_goal_conversion as fixtures
        with tempfile.TemporaryDirectory() as directory:
            builder = fixtures.GoalTests()
            builder.root = Path(directory)
            spec = fixtures.converter.load_goal_spec(builder.create())

        def state(facts):
            raw = [NS(name=parts[0], arguments=[NS(name=arg) for arg in parts[1:]])
                   for parts in map(fixtures.converter.parse_pddl, facts)]
            return {"_facts": raw,
                    # A display label may identify different original objects
                    # across episodes. It is never the saved PDDL identity.
                    "facts": [NS(name="atlocation", arguments=[NS(name="agent1"), NS(name="loc 7")])],
                    "feedback": "You see mug 2 in cabinet 1."}

        initial = state(fixtures.INITIAL)
        success = state(fixtures.INITIAL + ["(isclean mug0)"])
        wrong_object = state(fixtures.INITIAL + ["(isclean mug1)"])
        simulator = NS(reset=lambda: initial,
                       step=mock.Mock(side_effect=[(success, 1, True), (wrong_object, 0, False)]))
        wrapper = self.wrapper(simulator)
        for result, expected in ((wrapper.reset(), False), (wrapper.step("clean mug 2")[0], True),
                                 (wrapper.step("clean mug 2")[0], False)):
            # Gym Filter keeps requested extras, batching wraps each in a list,
            # and an async process serializes that payload.
            filtered = {"extra.trace_raw_facts": [result["extra.trace_raw_facts"]]}
            batched = pickle.loads(pickle.dumps(filtered))["extra.trace_raw_facts"]
            normalized = fixtures.converter.normalize_facts(batched)
            self.assertEqual(fixtures.converter.normalize_facts(normalized), normalized)
            self.assertEqual(fixtures.converter.evaluate_goal(spec, normalized), expected)
            rendered = fixtures.converter.render_full_state(spec, normalized)
            self.assertIn("Current full environment state", rendered)
            self.assertEqual(result["facts"][0].arguments[1].name, "loc 7")
            self.assertEqual(result["feedback"], "You see mug 2 in cabinet 1.")
        self.assertIs(wrapper.reset()["_facts"], initial["_facts"])

    def test_missing_raw_facts_does_not_guess_from_display_names(self):
        wrapper = self.wrapper(NS(reset=lambda: {"facts": []}))
        with self.assertRaisesRegex(RuntimeError, "raw PddlEnv _facts"):
            wrapper.reset()


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
                    "trace_state_context": self.trace_state_context,
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
        self.assertEqual([item["trace_turn_type"] for item in logs], ["tool_plan", "tool_plan"])
        self.assertEqual([item["trace_next_state_context"] for item in logs],
                         ["private full state 1", "private full state 2"])
        self.assertTrue(all(item["trace_next_scoring_valid"] for item in logs))
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
