"""Saved ALFWorld goal artifacts reach local/external Ray actors and YAML launchers."""
import ast
import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import yaml

import test_search_cache_runtime_env as routing


ROOT = Path(__file__).resolve().parents[2]


class ArtifactRuntimeEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        routing.SearchCacheRuntimeEnvTests.setUpClass()
        cls.runner = routing.SearchCacheRuntimeEnvTests()

    def test_goal_file_is_forwarded_to_new_and_existing_ray_clusters(self):
        for initialized in (False, True):
            with self.subTest(initialized=initialized), mock.patch.dict(
                os.environ, {"ALFWORLD_TRACE_GOALS": "/shared/goals.sqlite"}, clear=True
            ):
                ray, task_runner = self.runner.run_ppo_with_fake_ray(initialized)
            expected = {"SEARCH_CACHE_ENABLED": "0", "ALFWORLD_TRACE_GOALS": "/shared/goals.sqlite"}
            task_runner.options.assert_called_once_with(runtime_env={"env_vars": expected})
            if initialized:
                ray.init.assert_not_called()
            else:
                self.assertEqual(ray.init.call_args.kwargs["runtime_env"]["env_vars"], {
                    "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", **expected,
                })

    def test_goal_artifact_does_not_change_search_cache_routing(self):
        env = {"ALFWORLD_TRACE_GOALS": "/shared/goals.sqlite", "SEARCH_CACHE_ENABLED": "1",
               "SEARCH_CACHE_BASE_URL": "http://search:8091", "SEARCH_CACHE_TOKEN": "test"}
        with mock.patch.dict(os.environ, env, clear=True):
            _, task_runner = self.runner.run_ppo_with_fake_ray(True)
        task_runner.options.assert_called_once_with(runtime_env={"env_vars": env})

    def test_missing_goal_setting_does_not_add_a_requirement_to_qa(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            _, task_runner = self.runner.run_ppo_with_fake_ray(True)
        task_runner.options.assert_called_once_with(runtime_env={"env_vars": {"SEARCH_CACHE_ENABLED": "0"}})

    def test_both_training_configs_preserve_explicit_goal_path_and_empty_default(self):
        path = ROOT / "train-roma/utils.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "load_and_set_env_from_yaml")
        scope = {"os": os, "yaml": yaml, "sys": sys}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            for goal_path in (None, "/shared/training-goals.sqlite"):
                env = {} if goal_path is None else {"ALFWORLD_TRACE_GOALS": goal_path}
                with self.subTest(config=filename, goal_path=goal_path), \
                     mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()):
                    config = scope[node.name](str(ROOT / "train-roma" / filename))
                    self.assertEqual(config["env"]["ALFWORLD_TRACE_GOALS"], goal_path or "")
                    self.assertEqual(os.environ["ALFWORLD_TRACE_GOALS"], goal_path or "")
                    self.assertEqual(config["env"]["TRACE_K"], "3")
                    self.assertEqual(config["env"]["TRACE_GAMMA"], "0.8")
                    self.assertEqual(config["env"]["TRACE_TERMINAL_FILL_SCALE"], "2.0")


if __name__ == "__main__":
    unittest.main()
