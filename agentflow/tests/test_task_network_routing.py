"""Task selection and Ray routing never start models, SSH or external services."""
import builtins
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

import test_search_cache_runtime_env as ray_fixtures


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "train-roma/task_network.py"
SPEC = importlib.util.spec_from_file_location("task_network_routing_test", MODULE)
network = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(network)
DISABLED = {"SEARCH_CACHE_ENABLED": "0", "SEARCH_PROXY_ENABLED": "0",
            "SEARCH_CACHE_AUTO_START": "0", "SEARCH_CACHE_AUTO_TUNNEL": "0"}


class TaskResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "task.yaml"

    def test_environment_task_wins_without_importing_yaml_or_reading_config(self):
        original_import = builtins.__import__

        def no_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise AssertionError("Explicit TASK must not load configuration dependencies")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=no_yaml):
            self.assertEqual(network.resolve_task(self.root / "missing", environ={"TASK": " AlFWorld "}), "alfworld")

    def test_literal_and_environment_default_configs(self):
        for value, expected in (("webshop", "webshop"), ("${env:TASK,qa}", "qa"),
                                ("${env:TRAINING_TASK,alfworld}", "alfworld")):
            self.config.write_text(f"env:\n  TASK: '{value}'\n")
            self.assertEqual(network.resolve_task(self.config, environ={}), expected)
        self.assertEqual(network.resolve_task(self.config, environ={"TRAINING_TASK": "rule"}), "rule")

    def test_invalid_or_missing_config_fails_with_bounded_error(self):
        for value in ("", "bad task", "${unknown:value}", True, None):
            with self.subTest(value=value), self.assertRaises(network.TaskNetworkError):
                network.normalize_task(value)
        with self.assertRaises(network.TaskNetworkError):
            network.resolve_task(self.config, environ={})
        self.config.write_text("env: [invalid]")
        with self.assertRaises(network.TaskNetworkError):
            network.resolve_task(self.config, environ={})

    def test_local_and_online_task_policy(self):
        for task in ("alfworld", "webshop", "rule"):
            self.assertFalse(network.task_needs_search(task))
        for task in ("qa", "new_task"):
            self.assertTrue(network.task_needs_search(task))

    def test_yaml_error_does_not_quote_unrelated_credentials(self):
        self.config.write_text("env: {TASK: qa}\nSECRET: [unit-test-sensitive-value\n")
        with self.assertRaises(network.TaskNetworkError) as caught:
            network.resolve_task(self.config, environ={})
        self.assertNotIn("unit-test-sensitive-value", str(caught.exception))

    def test_cli_uses_documented_exit_codes_and_only_reports_task(self):
        for task, expected in (("qa", 0), ("alfworld", 1), ("webshop", 1), ("rule", 1), ("", 2)):
            result = subprocess.run([sys.executable, str(MODULE), "needs-search"],
                                    env={**os.environ, "TASK": task}, capture_output=True, text=True)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertEqual(result.stdout, "")


class TaskYamlLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "train-roma/utils.py"
        tree = ast.parse(path.read_text())
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                    and item.name == "load_and_set_env_from_yaml")
        scope = {"os": os, "yaml": yaml, "sys": sys}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        cls.load_config = staticmethod(scope[node.name])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / "config.yaml"

    def test_task_loader_matches_startup_resolver_on_both_config_loads(self):
        cases = (("qa", {"TASK": "alfworld"}, "alfworld"),
                 ("qa", {"TASK": " WeBShop "}, "webshop"),
                 ("AlfWorld", {}, "alfworld"),
                 ("${env:TASK,qa}", {}, "qa"),
                 ("${env:TRAINING_TASK,qa}", {"TRAINING_TASK": "rule"}, "rule"))
        for literal, environment, expected in cases:
            with self.subTest(task=literal, environment=environment):
                self.config.write_text(yaml.safe_dump({"env": {"TASK": literal}}))
                with mock.patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(io.StringIO()):
                    startup_task = network.resolve_task(self.config)
                    first = self.load_config(self.config)   # utils import initialization
                    second = self.load_config(self.config)  # main's explicit reload
                    self.assertEqual(startup_task, expected)
                    self.assertEqual(first["env"]["TASK"], expected)
                    self.assertEqual(second["env"]["TASK"], expected)
                    self.assertEqual(os.environ["TASK"], expected)
                    self.assertEqual(network.resolve_task(self.config), expected)

    def test_task_special_case_does_not_change_other_environment_fields(self):
        self.config.write_text(yaml.safe_dump({"env": {"TASK": "qa", "OTHER": "literal",
                                                       "SCORE_MODEL_URL": "http://model:19998/v1"}}))
        with mock.patch.dict(os.environ, {"TASK": "AlfWorld", "OTHER": "old-value"}, clear=True), \
             contextlib.redirect_stdout(io.StringIO()):
            config = self.load_config(self.config)
            self.assertEqual(config["env"]["TASK"], "alfworld")
            self.assertEqual(os.environ["OTHER"], "literal")
            self.assertEqual(os.environ["SCORE_MODEL_URL"], "http://model:19998/v1")

    def test_invalid_task_is_rejected_by_both_startup_and_python_loader(self):
        for value in ("", "bad task", True, None):
            with self.subTest(value=value), mock.patch.dict(os.environ, {}, clear=True):
                self.config.write_text(yaml.safe_dump({"env": {"TASK": value}}))
                with self.assertRaises(network.TaskNetworkError):
                    network.resolve_task(self.config)
                with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                    self.load_config(self.config)


class TaskShellRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="task routing ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scripts = self.root / "train-roma"
        self.scripts.mkdir()
        self.cwd = self.root / "unrelated-working-dir"
        self.cwd.mkdir()
        for name in ("task_network.sh", "task_network.py"):
            shutil.copy2(ROOT / "train-roma" / name, self.scripts / name)
        self.helper = self.scripts / "task_network.sh"
        self.env = {key: value for key, value in os.environ.items()
                    if key not in {"TASK", "AGENTFLOW_CONFIG_FILE", "AGENTFLOW_TASK_PYTHON"}}
        self.env["AGENTFLOW_TASK_PYTHON"] = sys.executable

    def run_shell(self, body, env=None):
        return subprocess.run(["bash", "-c", 'set -eu; source "$1"; ' + body, "bash", str(self.helper)],
                              cwd=self.cwd, env={**self.env, **(env or {})}, capture_output=True, text=True)

    def test_explicit_task_does_not_start_python_and_is_normalized(self):
        result = self.run_shell('agentflow_resolve_task; test "$TASK" = alfworld; '
                                'if agentflow_task_needs_search; then exit 9; fi',
                                {"TASK": " AlFWorld ", "AGENTFLOW_TASK_PYTHON": "/missing/python"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_default_config_is_relative_to_helper_not_cwd(self):
        (self.scripts / "config.yaml").write_text("env:\n  TASK: webshop\n")
        result = self.run_shell('agentflow_resolve_task; test "$TASK" = webshop; '
                                'if agentflow_task_needs_search; then exit 9; fi')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_custom_config_and_online_policy(self):
        config = self.root / "alternate config.yaml"
        config.write_text("env:\n  TASK: qa\n")
        result = self.run_shell('agentflow_resolve_task; test "$TASK" = qa; agentflow_task_needs_search',
                                {"AGENTFLOW_CONFIG_FILE": str(config)})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_disable_only_removes_managed_proxy_and_python_hook(self):
        result = self.run_shell(
            'agentflow_disable_search_components; '
            'test "$SEARCH_CACHE_ENABLED" = 0; test "$SEARCH_PROXY_ENABLED" = 0; '
            'test "$SEARCH_CACHE_AUTO_START" = 0; test "$SEARCH_CACHE_AUTO_TUNNEL" = 0; '
            'test "$PYTHONPATH" = /before:/after; '
            'test -z "${HTTP_PROXY+x}"; test "$HTTPS_PROXY" = http://user-proxy; '
            'test "$SCORE_MODEL_URL" = http://model:19998/v1; test -z "${HF_HUB_OFFLINE+x}"',
            {"TASK": "webshop", "PROXY_PY_DIR": "/managed-hook", "PYTHONPATH": "/before:/managed-hook:/after",
             "AGENTFLOW_SEARCH_PROXY_ACTIVE": "1", "AGENTFLOW_SEARCH_PROXY_URL": "http://search-proxy",
             "HTTP_PROXY": "http://search-proxy", "HTTPS_PROXY": "http://user-proxy",
             "SCORE_MODEL_URL": "http://model:19998/v1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_disable_preserves_empty_python_path_components_and_unmanaged_proxy(self):
        result = self.run_shell('agentflow_disable_search_components; test "$PYTHONPATH" = :/other:; '
                                'test "$HTTP_PROXY" = http://user-proxy',
                                {"PROXY_PY_DIR": "/managed", "PYTHONPATH": ":/managed:/other:",
                                 "HTTP_PROXY": "http://user-proxy"})
        self.assertEqual(result.returncode, 0, result.stderr)


class TaskRayRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray_fixtures.SearchCacheRuntimeEnvTests.setUpClass()
        cls.runner = ray_fixtures.SearchCacheRuntimeEnvTests()

    def test_local_task_overrides_stale_cache_and_gateway_environment(self):
        env = {"TASK": "qa", "SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_TOKEN": "unit-test-token",
               "SEARCH_CACHE_BASE_URL": "http://search:8091", "SEARCH_GATEWAY_TOKEN": "unit-test-gateway"}
        for task in ("alfworld", "webshop", "rule"):
            for initialized in (False, True):
                with self.subTest(task=task, initialized=initialized), mock.patch.dict(os.environ, env, clear=True):
                    ray, runner = self.runner.run_ppo_with_fake_ray(initialized, SimpleNamespace(data={"task": task}))
                runner.options.assert_called_once_with(runtime_env={"env_vars": DISABLED})
                if initialized:
                    ray.init.assert_not_called()
                else:
                    self.assertEqual(ray.init.call_args.kwargs["runtime_env"]["env_vars"],
                                     {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", **DISABLED})

    def test_qa_config_preserves_existing_enabled_routing(self):
        env = {"TASK": "alfworld", "SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_TOKEN": "unit-test-token"}
        with mock.patch.dict(os.environ, env, clear=True):
            _, runner = self.runner.run_ppo_with_fake_ray(True, SimpleNamespace(data={"task": "qa"}))
        runner.options.assert_called_once_with(runtime_env={"env_vars": {
            "SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_TOKEN": "unit-test-token"}})

    def test_existing_ray_gets_only_exact_managed_proxy_and_path_overrides(self):
        env = {"TASK": "qa", "SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_TOKEN": "unit-test-token",
               "AGENTFLOW_SEARCH_PROXY_ACTIVE": "1", "AGENTFLOW_SEARCH_PROXY_URL": "http://managed-proxy",
               "PROXY_PY_DIR": "/managed-hook", "PYTHONPATH": ":/before:/managed-hook:/managed-hook-other:/after:",
               "HTTP_PROXY": "http://managed-proxy", "https_proxy": "http://managed-proxy",
               "HTTPS_PROXY": "http://ordinary-user-proxy", "http_proxy": "http://ordinary-user-proxy",
               "SCORE_MODEL_URL": "http://model:19998/v1"}
        with mock.patch.dict(os.environ, env, clear=True):
            ray, runner = self.runner.run_ppo_with_fake_ray(True, SimpleNamespace(data={"task": "alfworld"}))
        ray.init.assert_not_called()
        expected = {**DISABLED, "HTTP_PROXY": "", "https_proxy": "",
                    "PYTHONPATH": ":/before:/managed-hook-other:/after:",
                    "AGENTFLOW_SEARCH_PROXY_ACTIVE": "0", "AGENTFLOW_SEARCH_PROXY_URL": "", "PROXY_PY_DIR": ""}
        runner.options.assert_called_once_with(runtime_env={"env_vars": expected})
        self.assertNotIn("HTTPS_PROXY", expected)
        self.assertNotIn("http_proxy", expected)
        self.assertNotIn("SCORE_MODEL_URL", expected)

    def test_unmatched_managed_path_does_not_override_python_path(self):
        env = {"PROXY_PY_DIR": "/managed", "PYTHONPATH": "/managed-other:/ordinary",
               "HTTP_PROXY": "http://ordinary-proxy"}
        with mock.patch.dict(os.environ, env, clear=True):
            selected = self.runner.select_env("webshop")
        self.assertNotIn("PYTHONPATH", selected)
        self.assertNotIn("HTTP_PROXY", selected)
        self.assertEqual(selected["PROXY_PY_DIR"], "")

    def test_qa_keeps_its_managed_proxy_environment_unchanged(self):
        env = {"TASK": "webshop", "SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_TOKEN": "unit-test-token",
               "AGENTFLOW_SEARCH_PROXY_ACTIVE": "1", "AGENTFLOW_SEARCH_PROXY_URL": "http://managed-proxy",
               "PROXY_PY_DIR": "/managed", "PYTHONPATH": "/managed", "HTTP_PROXY": "http://managed-proxy"}
        with mock.patch.dict(os.environ, env, clear=True):
            selected = self.runner.select_env("qa")
        self.assertEqual(selected, {"SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_TOKEN": "unit-test-token"})


if __name__ == "__main__":
    unittest.main()
