"""Exercise actual Ray env selection without importing GPU/Ray dependencies."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch


class SearchCacheRuntimeEnvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "verl" / "entrypoint.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_search_cache_env_vars")
        scope = {"os": os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        cls.select_env = staticmethod(scope[node.name])

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


if __name__ == "__main__":
    unittest.main()
