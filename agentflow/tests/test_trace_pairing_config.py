"""The real YAML launch path carries the pairing switch into TRACE selection."""

import argparse
import ast
import contextlib
from dataclasses import asdict
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "trace_pairing_config_test_credit", ROOT / "agentflow/agentflow/trace/credit.py")
CREDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CREDIT
SPEC.loader.exec_module(CREDIT)

try:
    import yaml
except ImportError:
    yaml = None


def load_function(path, name, scope):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


@unittest.skipIf(yaml is None, "PyYAML is unavailable")
class PairingSwitchLaunchTests(unittest.TestCase):
    def launch_config(self, filename, environment=None, overrides=()):
        # Execute the production environment loader and argument builder without
        # importing their model clients or launching a distributed process.
        loader = load_function(ROOT / "train-roma/utils.py", "load_and_set_env_from_yaml",
                               {"os": os, "yaml": yaml, "sys": sys})
        utils = types.ModuleType("utils")
        utils.load_and_set_env_from_yaml = lambda _: loader(str(ROOT / "train-roma" / filename))
        runner = mock.Mock()
        main = load_function(ROOT / "train-roma/train_agent.py", "main", {
            "os": os, "sys": sys, "argparse": argparse,
            "subprocess": types.SimpleNamespace(run=runner,
                                                CalledProcessError=subprocess.CalledProcessError),
        })
        with mock.patch.dict(os.environ, environment or {}, clear=True), \
             mock.patch.dict(sys.modules, {"utils": utils}), \
             mock.patch.object(sys, "argv", ["train_agent.py", *overrides]), \
             contextlib.redirect_stdout(io.StringIO()):
            main()
        runner.assert_called_once()
        command = runner.call_args.args[0]
        self.assertEqual(command[:3], ["python", "-m", "agentflow.verl"])
        values = yaml.safe_load((ROOT / "agentflow/verl/config.yaml").read_text())["data"]["trace"]
        # Hydra receives these scalar CLI literals. Preserve CLI ordering so a
        # user override appended by train_agent.main wins over the YAML default.
        for argument in command[3:]:
            if argument.startswith("data.trace."):
                key, literal = argument.split("=", 1)
                values[key.removeprefix("data.trace.")] = yaml.safe_load(literal)
        return CREDIT.TraceConfig.from_mapping(values), command

    @staticmethod
    def pairs(config):
        rows = [dict(index=index, uid="question", turn_type="tool_plan", turn_index=0,
                     rollout_id=f"rollout-{index}", value=.2, credit=credit,
                     valid=True, length=10, old_per_token=-1., action_key=f"action-{index}")
                for index, credit in enumerate((.4, -.1, -.2))]
        return CREDIT.select_pairs(rows, config)

    def test_package_default_preserves_v2_e_all_combinations(self):
        document = yaml.safe_load((ROOT / "agentflow/verl/config.yaml").read_text())
        self.assertIs(document["data"]["trace"]["pair_round_robin"], False)
        self.assertFalse(CREDIT.TraceConfig.from_mapping(document["data"]["trace"]).pair_round_robin)

    def test_both_training_yaml_files_export_the_switch(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                document = yaml.safe_load((ROOT / "train-roma" / filename).read_text())
                self.assertEqual(document["env"]["TRACE_PAIR_ROUND_ROBIN"],
                                 "${env:TRACE_PAIR_ROUND_ROBIN,false}")
                self.assertEqual(document["python_args"]["data.trace.pair_round_robin"],
                                 "${TRACE_PAIR_ROUND_ROBIN}")

    def test_default_launch_preserves_v2_e_all_combinations(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                config, command = self.launch_config(filename)
                self.assertFalse(config.pair_round_robin)
                self.assertIn("data.trace.pair_round_robin=false", command)
                self.assertEqual(self.pairs(config), [(0, 2), (0, 1)])

    def test_env_switch_changes_selection_without_changing_other_controls(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                enabled, enabled_command = self.launch_config(filename, {"TRACE_PAIR_ROUND_ROBIN": "true"})
                disabled, disabled_command = self.launch_config(filename, {"TRACE_PAIR_ROUND_ROBIN": "false"})
                self.assertTrue(enabled.pair_round_robin)
                self.assertFalse(disabled.pair_round_robin)
                self.assertIn("data.trace.pair_round_robin=true", enabled_command)
                self.assertIn("data.trace.pair_round_robin=false", disabled_command)
                self.assertEqual(self.pairs(enabled), [(0, 2)])
                self.assertEqual(self.pairs(disabled), [(0, 2), (0, 1)])
                self.assertEqual({key: value for key, value in asdict(enabled).items()
                                  if key != "pair_round_robin"},
                                 {key: value for key, value in asdict(disabled).items()
                                  if key != "pair_round_robin"})

    def test_explicit_cli_override_wins_over_opposite_env_value(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            for env_value, override, expected_pairs in (
                    ("true", "false", [(0, 2), (0, 1)]),
                    ("false", "true", [(0, 2)])):
                with self.subTest(filename=filename, override=override):
                    config, command = self.launch_config(
                        filename, {"TRACE_PAIR_ROUND_ROBIN": env_value},
                        ["data.trace.pair_round_robin=" + override])
                    self.assertEqual(command[-1], "data.trace.pair_round_robin=" + override)
                    self.assertIs(config.pair_round_robin, override == "true")
                    self.assertEqual(self.pairs(config), expected_pairs)


if __name__ == "__main__":
    unittest.main()
