"""Exercise real reference scheduling/resume without Ray or accelerator imports."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CREDIT = load_module("reference_lifecycle_credit", ROOT / "agentflow/agentflow/trace/credit.py")
# Keep production method bodies; only replace the package import to avoid loading
# the application (and distributed optional dependencies) in this CPU test.
path = ROOT / "agentflow/verl/trace_reference.py"
tree = ast.parse(path.read_text())
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.ImportFrom) and node.module == "agentflow.trace.credit")]
scope = {"TraceConfig": CREDIT.TraceConfig}
exec(compile(tree, str(path), "exec"), scope)
Lifecycle = scope["TraceReferenceLifecycle"]


class Config(dict):
    __getattr__ = dict.__getitem__


class Worker:
    def __init__(self):
        self.calls = []
        self.fail = False

    def validate_reference_refresh(self):
        self.calls.append(("validate",))

    def refresh_reference_from_actor(self, step):
        if self.fail:
            raise RuntimeError("copy failed")
        self.calls.append(("refresh", step))

    def save_reference_checkpoint(self, path, remote, step, **kwargs):
        self.calls.append(("save", step, kwargs))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "model.pt").write_text("snapshot")

    def load_reference_checkpoint(self, path, **kwargs):
        self.calls.append(("load", kwargs))


class Base:
    def _load_checkpoint(self):
        self.global_steps = self.resume_step

    def _save_checkpoint(self):
        # The advertised actor checkpoint must never predate its reference save.
        if self._trace_reference_interval() != 0:
            path = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}" / "trace_reference"
            assert (path / "model.pt").exists()
            assert (path / "trace_reference.json").exists()
        self.base_saved = True


class Trainer(Lifecycle, Base):
    def __init__(self, root, interval=0, task="alfworld"):
        self.config = SimpleNamespace(
            data=Config(trace={"enabled": True, "reference_update_interval": interval}),
            trainer=Config(default_local_dir=root, default_hdfs_dir=None, resume_mode="auto",
                           del_local_ckpt_after_load=False),
        )
        self.algorithm = "grpo+slic"
        self.task = task
        self.global_steps = self.resume_step = 0
        self.ref_policy_wg = Worker()
        self.ref_in_actor = False
        self.use_reference_policy = True


class ReferenceConfigTests(unittest.TestCase):
    def test_values_and_roundtrip(self):
        self.assertEqual(CREDIT.TraceConfig().reference_update_interval, 0)
        for value, expected in [(None, None), ("None", None), (" null ", None), (0, 0), (1, 1), (10, 10)]:
            self.assertEqual(CREDIT.TraceConfig.from_mapping({"reference_update_interval": value}).reference_update_interval, expected)
        for invalid in [True, False, -1, 1.5, "10", "bad", [], {}]:
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "reference_update_interval"):
                CREDIT.TraceConfig.from_mapping({"reference_update_interval": invalid})

    def test_yaml_routes_interval_to_training(self):
        import yaml
        for file in ["config.yaml", "config_grpo+slic.yaml"]:
            config = yaml.safe_load((ROOT / "train-roma" / file).read_text())
            self.assertEqual(config["env"]["TRACE_REFERENCE_UPDATE_INTERVAL"], "${env:TRACE_REFERENCE_UPDATE_INTERVAL,0}")
            self.assertEqual(config["python_args"]["data.trace.reference_update_interval"], "${TRACE_REFERENCE_UPDATE_INTERVAL}")
        config = yaml.safe_load((ROOT / "agentflow/verl/config.yaml").read_text())
        self.assertEqual(config["data"]["trace"]["reference_update_interval"], 0)


class ReferenceLifecycleTests(unittest.TestCase):
    def test_same_schedule_for_qa_alfworld_and_both_algorithms(self):
        for task in ("qa", "alfworld"):
            for algorithm in ("grpo+slic", "gigpo+slic"):
                trainer = Trainer("unused", 3, task)
                trainer.algorithm = algorithm
                trainer._load_checkpoint()
                for step in range(1, 8):
                    trainer.global_steps = step
                    metrics = trainer._refresh_trace_reference_after_step()
                    self.assertEqual(metrics["trace/reference_step"], (step - 1) // 3 * 3)
                    self.assertEqual(metrics["trace/reference_step_next"], step // 3 * 3)
                self.assertEqual(trainer.ref_policy_wg.calls, [("validate",), ("refresh", 3), ("refresh", 6)])

    def test_none_is_latest_previous_completed_step(self):
        trainer = Trainer("unused", None)
        trainer._load_checkpoint()
        for step in range(1, 4):
            trainer.global_steps = step
            metrics = trainer._refresh_trace_reference_after_step()
            self.assertEqual(metrics["trace/reference_step"], step - 1)
            self.assertEqual(metrics["trace/reference_step_next"], step)

    def test_fixed_and_non_trace_do_not_touch_workers(self):
        for interval, algorithm, enabled in [(0, "grpo+slic", True), (None, "grpo", True), (2, "gigpo+slic", False)]:
            trainer = Trainer("unused", interval)
            trainer.algorithm = algorithm
            trainer.config.data.trace["enabled"] = enabled
            trainer._load_checkpoint()
            trainer.global_steps = 12
            trainer._refresh_trace_reference_after_step()
            trainer._save_checkpoint()
            self.assertFalse(trainer.ref_policy_wg.calls)

    def test_failed_copy_does_not_advance_version(self):
        trainer = Trainer("unused", None)
        trainer._load_checkpoint()
        trainer.global_steps = 1
        trainer.ref_policy_wg.fail = True
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            trainer._refresh_trace_reference_after_step()
        self.assertEqual(trainer._trace_reference_step, 0)

    def test_snapshot_restores_nonboundary_reference(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = Trainer(root, 3)
            trainer._load_checkpoint()
            trainer.global_steps = 3
            trainer._refresh_trace_reference_after_step()
            trainer.global_steps = 5
            trainer._save_checkpoint()
            restored = Trainer(root, 3)
            restored.resume_step = 5
            restored._load_checkpoint()
            self.assertEqual(restored._trace_reference_step, 3)
            self.assertEqual(restored.ref_policy_wg.calls[-1], ("load", {"del_local_after_load": False, "reference_step": 3}))
            restored.global_steps = 6
            restored._refresh_trace_reference_after_step()
            self.assertEqual(restored._trace_reference_step, 6)

    def test_explicit_resume_path_is_used(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = Trainer(root, 5)
            trainer._load_checkpoint()
            trainer.global_steps = 2
            trainer._save_checkpoint()
            restored = Trainer("not-the-checkpoint-root", 5)
            restored.config.trainer.update(resume_mode="resume_path", resume_from_path=str(Path(root) / "global_step_2"))
            restored.resume_step = 2
            restored._load_checkpoint()
            self.assertEqual(restored._trace_reference_step, 0)

    def test_old_save_can_be_adopted_only_at_refresh_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            for interval in [None, 1, 3]:
                trainer = Trainer(root, interval)
                trainer.resume_step = 6
                trainer._load_checkpoint()
                self.assertEqual(trainer.ref_policy_wg.calls[-1], ("refresh", 6))
            trainer = Trainer(root, 3)
            trainer.resume_step = 5
            with self.assertRaisesRegex(ValueError, "older reference snapshot"):
                trainer._load_checkpoint()

    def test_incompatible_saved_interval_fails(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = Trainer(root, 3)
            trainer._load_checkpoint()
            trainer.global_steps = 2
            trainer._save_checkpoint()
            restored = Trainer(root, None)
            restored.resume_step = 2
            with self.assertRaisesRegex(ValueError, "configured interval/step"):
                restored._load_checkpoint()

    def test_refresh_is_after_updates_and_stale_early_return(self):
        source = (ROOT / "agentflow/verl/trainer.py").read_text()
        tree = ast.parse(source)
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentFlowTrainer")
        step = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_train_step")
        calls = {name: [n.lineno for n in ast.walk(step) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name]
                 for name in ("update_actor", "update_actor_by_dpo", "_refresh_trace_reference_after_step")}
        self.assertEqual(len(calls["_refresh_trace_reference_after_step"]), 1)
        refresh_line = calls["_refresh_trace_reference_after_step"][0]
        self.assertGreater(refresh_line, max(calls["update_actor"] + calls["update_actor_by_dpo"]))
        early = [n.lineno for n in ast.walk(step) if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant) and n.value.value is None]
        self.assertTrue(early)
        self.assertTrue(all(line < refresh_line for line in early))


if __name__ == "__main__":
    unittest.main()
