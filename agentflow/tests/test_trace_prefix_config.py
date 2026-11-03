"""Shared-prefix scorer controls validate before models or rollouts are loaded."""

from dataclasses import asdict
import importlib.util
from pathlib import Path
import re
import sys
import unittest
import warnings


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "trace_prefix_config_test_credit", ROOT / "agentflow/agentflow/trace/credit.py")
CREDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CREDIT
SPEC.loader.exec_module(CREDIT)
TraceConfig = CREDIT.TraceConfig

try:
    import yaml
except ImportError:
    yaml = None


PREFIX_SETTINGS = {
    "shared_prefix_scoring": ("TRACE_SHARED_PREFIX_SCORING", True),
    "shared_prefix_max_tokens": ("TRACE_SHARED_PREFIX_MAX_TOKENS", 8192),
    "shared_prefix_max_mask_mib": ("TRACE_SHARED_PREFIX_MAX_MASK_MIB", 256),
    "shared_prefix_verify": ("TRACE_SHARED_PREFIX_VERIFY", True),
}


class SharedPrefixConfigTests(unittest.TestCase):
    def test_library_default_keeps_original_scorer_and_normalizes_round_trip(self):
        cfg = TraceConfig.from_mapping()
        self.assertFalse(cfg.shared_prefix_scoring)
        self.assertTrue(cfg.shared_prefix_verify)
        self.assertEqual(cfg.shared_prefix_max_tokens, 8192)
        self.assertEqual(cfg.shared_prefix_max_mask_mib, 256)
        self.assertEqual(TraceConfig.from_mapping(asdict(cfg)), cfg)
        self.assertIs(TraceConfig.from_mapping(cfg), cfg)

    def test_enabled_and_explicitly_disabled_controls_are_independent_of_credit(self):
        for enabled, verify in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(enabled=enabled, verify=verify):
                cfg = TraceConfig.from_mapping({
                    "shared_prefix_scoring": enabled,
                    "shared_prefix_verify": verify,
                    "shared_prefix_max_tokens": 4096,
                    "shared_prefix_max_mask_mib": 64,
                })
                self.assertIs(cfg.shared_prefix_scoring, enabled)
                self.assertIs(cfg.shared_prefix_verify, verify)
                self.assertEqual(cfg.shared_prefix_max_tokens, 4096)
                self.assertEqual(cfg.shared_prefix_max_mask_mib, 64)
                base = TraceConfig()
                for key, value in asdict(base).items():
                    if key not in PREFIX_SETTINGS:
                        self.assertEqual(getattr(cfg, key), value)

    def test_prefix_boolean_flags_reject_strings_numbers_and_null(self):
        for key in ("shared_prefix_scoring", "shared_prefix_verify"):
            for value in ("true", "false", "1", 0, 1, None, [], {}):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, re.escape("trace." + key) + " must be a boolean"
                ):
                    TraceConfig.from_mapping({key: value})

    def test_prefix_limits_require_positive_integers_even_when_disabled(self):
        for key in ("shared_prefix_max_tokens", "shared_prefix_max_mask_mib"):
            for value in (True, False, 0, -1, 2.5, "8192", None, float("inf")):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, re.escape("trace." + key) + " must be a positive integer"
                ):
                    TraceConfig.from_mapping({key: value, "shared_prefix_scoring": False})

    def test_legacy_config_warning_and_unknown_field_rejection_are_preserved(self):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            cfg = TraceConfig.from_mapping({"pair_score_mode": "igpo", "igpo_gamma": 1.,
                                            "shared_prefix_scoring": True})
        self.assertTrue(cfg.shared_prefix_scoring)
        self.assertEqual(len(recorded), 1)
        self.assertIn("does not enable IGPO", str(recorded[0].message))
        with self.assertRaisesRegex(ValueError, "Unknown TRACE configuration fields"):
            TraceConfig.from_mapping({"shared_prefix_scorng": True})


@unittest.skipIf(yaml is None, "PyYAML is unavailable")
class SharedPrefixYamlTests(unittest.TestCase):
    def test_package_defaults_match_the_validated_library_defaults(self):
        document = yaml.safe_load((ROOT / "agentflow/verl/config.yaml").read_text())
        cfg = TraceConfig.from_mapping(document["data"]["trace"])
        self.assertEqual(cfg, TraceConfig())

    def test_training_entrypoints_enable_and_export_the_same_prefix_controls(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                document = yaml.safe_load((ROOT / "train-roma" / filename).read_text())
                for field, (variable, default) in PREFIX_SETTINGS.items():
                    default_literal = str(default).lower() if isinstance(default, bool) else str(default)
                    self.assertEqual(document["env"][variable], "${env:" + variable + "," + default_literal + "}")
                    self.assertEqual(document["python_args"]["data.trace." + field], "${" + variable + "}")

    def test_explicit_env_disable_reaches_the_original_scorer_switch(self):
        for filename in ("config.yaml", "config_grpo+slic.yaml"):
            with self.subTest(filename=filename):
                document = yaml.safe_load((ROOT / "train-roma" / filename).read_text())
                environment = {"TRACE_SHARED_PREFIX_SCORING": "false"}
                values = {}
                for field, (variable, _) in PREFIX_SETTINGS.items():
                    expression = document["env"][variable]
                    match = re.fullmatch(r"\$\{env:([A-Z_]+),([^}]+)\}", expression)
                    self.assertIsNotNone(match)
                    name, default = match.groups()
                    values[field] = yaml.safe_load(environment.get(name, default))
                cfg = TraceConfig.from_mapping(values)
                self.assertFalse(cfg.shared_prefix_scoring)
                self.assertTrue(cfg.shared_prefix_verify)
                self.assertEqual(cfg.shared_prefix_max_tokens, 8192)
                self.assertEqual(cfg.shared_prefix_max_mask_mib, 256)


if __name__ == "__main__":
    unittest.main()
