"""Exercise actual Ray env selection without importing GPU/Ray dependencies."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


class SearchCacheRuntimeEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "verl" / "entrypoint.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_search_cache_env_vars")
        scope = {"os": os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        cls.select_env = staticmethod(scope[node.name])
        cls.ppo_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_ppo")

    def run_ppo_with_fake_ray(self, initialized):
        ray, task_runner = Mock(), Mock()
        ray.is_initialized.return_value = initialized
        scope = {"os": os, "ray": ray, "TaskRunner": task_runner, "_search_cache_env_vars": self.select_env}
        exec(compile(ast.Module(body=[self.ppo_node], type_ignores=[]), "entrypoint.py", "exec"), scope)
        scope["run_ppo"](None)
        return ray, task_runner

    def test_enabled_forwards_client_config_without_upstream_credentials(self):
        env = {"SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_BASE_URL": "http://service:8091",
               "SEARCH_CACHE_TOKEN": "shared", "SEARCH_CACHE_READ_TIMEOUT_SECONDS": "600",
               "YIBU_BRAVE_API_KEY": "upstream-only", "HTTP_PROXY": "http://egress:8080"}
        with patch.dict(os.environ, env, clear=True):
            selected = self.select_env()
        self.assertEqual(selected, {k: v for k, v in env.items() if k.startswith("SEARCH_CACHE_")})

    def test_disabled_does_not_enable_an_old_gateway(self):
        with patch.dict(os.environ, {"SEARCH_GATEWAY_BASE_URL": "http://old:8080"}, clear=True):
            self.assertEqual(self.select_env(), {"SEARCH_CACHE_ENABLED": "0"})

    def test_legacy_gateway_settings_forwarded_only_in_enabled_mode(self):
        env = {"SEARCH_CACHE_ENABLED": "true", "SEARCH_GATEWAY_BASE_URL": "http://service:8091",
               "SEARCH_GATEWAY_TOKEN": "shared"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(self.select_env(), env)

    def test_local_ray_initialization_keeps_idea_device_setting_and_cache_routing(self):
        env = {"SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_BASE_URL": "http://service:8091", "SEARCH_CACHE_TOKEN": "shared"}
        with patch.dict(os.environ, env, clear=True):
            ray, task_runner = self.run_ppo_with_fake_ray(False)
        self.assertEqual(ray.init.call_args.kwargs["runtime_env"]["env_vars"], {
            "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", **env,
        })
        task_runner.options.assert_called_once_with(runtime_env={"env_vars": env})

    def test_already_initialized_ray_still_receives_cache_configuration(self):
        env = {"SEARCH_CACHE_ENABLED": "1", "SEARCH_CACHE_BASE_URL": "http://service:8091", "SEARCH_CACHE_TOKEN": "shared"}
        with patch.dict(os.environ, env, clear=True):
            ray, task_runner = self.run_ppo_with_fake_ray(True)
        ray.init.assert_not_called()
        task_runner.options.assert_called_once_with(runtime_env={"env_vars": env})


if __name__ == "__main__":
    unittest.main()
