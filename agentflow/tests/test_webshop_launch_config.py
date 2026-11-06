"""WebShop artifact settings survive launcher expansion and existing Ray clusters."""
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


class WebshopLaunchTests(unittest.TestCase):
    def test_both_configs_select_webshop_and_preserve_explicit_artifact_paths(self):
        source = ROOT / "train-roma/utils.py"
        method = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "load_and_set_env_from_yaml")
        scope = {"os": os, "yaml": yaml, "sys": sys}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            for override in ({}, {"WEBSHOP_ROOT": "/shared/WebShop", "WEBSHOP_TRACE_MANIFEST": "/shared/frozen.json.gz",
                                   "WEBSHOP_TRACE_HISTORY_LENGTH": "0", "TRACE_MATCH_EPSILON": "0"}):
                with self.subTest(filename=filename, override=override), mock.patch.dict(os.environ, override, clear=True), \
                        contextlib.redirect_stdout(io.StringIO()):
                    config = scope[method.name](str(ROOT / "train-roma" / filename))
                    env = config["env"]
                    self.assertEqual(env["TASK"], "webshop")
                    self.assertEqual(env["OUTPUT_TYPE"], "base")
                    self.assertIn("6810_webshop_with_gt.parquet", env["TRAIN_DATA_PATH"])
                    self.assertIn("webshop_test_100_webshop.parquet", env["TEST_DATA_PATH"])
                    self.assertEqual(env["WEBSHOP_TRACE_HISTORY_LENGTH"], override.get("WEBSHOP_TRACE_HISTORY_LENGTH", "4"))
                    self.assertEqual(env["TRACE_REFERENCE_UPDATE_INTERVAL"], "None")
                    for key, value in override.items():
                        self.assertEqual(env[key], value)
                        self.assertEqual(os.environ[key], value)

    def test_manifest_root_and_history_reach_new_and_existing_ray_clusters(self):
        routing.SearchCacheRuntimeEnvTests.setUpClass()
        runner = routing.SearchCacheRuntimeEnvTests()
        env = {"WEBSHOP_ROOT": "/shared/WebShop", "WEBSHOP_TRACE_MANIFEST": "/shared/frozen.json.gz",
               "WEBSHOP_TRACE_HISTORY_LENGTH": "4"}
        for initialized in (False, True):
            with self.subTest(initialized=initialized), mock.patch.dict(os.environ, env, clear=True):
                ray, task_runner = runner.run_ppo_with_fake_ray(initialized)
            expected = {"SEARCH_CACHE_ENABLED": "0", **env}
            task_runner.options.assert_called_once_with(runtime_env={"env_vars": expected})
            if not initialized:
                self.assertEqual(ray.init.call_args.kwargs["runtime_env"]["env_vars"],
                                 {"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", **expected})


if __name__ == "__main__":
    unittest.main()
