"""Exercise the real task-aware entry point without running training/services."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TaskNetworkStartTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="task-aware startup ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repo = self.root / "repo with spaces"
        self.scripts = self.repo / "train-roma"
        self.scripts.mkdir(parents=True)
        for name in ("start.sh", "task_network.sh", "task_network.py"):
            shutil.copy2(ROOT / "train-roma" / name, self.scripts / name)
        (self.scripts / "config.yaml").write_text("env:\n  TASK: alfworld\n")
        self.record = self.root / "training-record.txt"
        self.proxy = self.root / "proxy-record.txt"
        (self.scripts / "run_train.sh").write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\n" "$TASK" "${SEARCH_CACHE_ENABLED-unset}" '
            '"${SEARCH_PROXY_ENABLED-unset}" "$PWD" "$@" > "$TEST_TRAIN_RECORD"\n'
        )
        (self.scripts / "enable_search_proxy.sh").write_text(
            'printf "%s\\n" called > "$TEST_PROXY_RECORD"\n'
            'export SEARCH_PROXY_ENABLED=1\n'
        )
        self.env = {name: value for name, value in os.environ.items()
                    if not name.startswith(("TASK", "SEARCH_", "PROXY_", "AGENTFLOW_"))}
        self.env.update(TEST_TRAIN_RECORD=str(self.record), TEST_PROXY_RECORD=str(self.proxy))

    def launch(self, **env):
        return subprocess.run(["bash", str(self.scripts / "start.sh"), "forwarded-argument"],
                              cwd=self.root, env={**self.env, **env},
                              capture_output=True, text=True, timeout=10)

    def test_literal_yaml_local_task_runs_without_proxy_credentials_from_other_directory(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = self.record.read_text().splitlines()
        self.assertEqual(lines[:3], ["alfworld", "0", "0"])
        self.assertEqual(Path(lines[3]).resolve(), self.repo.resolve())
        self.assertEqual(lines[4], "forwarded-argument")
        self.assertFalse(self.proxy.exists())

    def test_local_task_overrides_stale_enabled_proxy_and_cache(self):
        for task in ("alfworld", "webshop", "rule"):
            with self.subTest(task=task):
                result = self.launch(TASK=task, SEARCH_PROXY_ENABLED="1", SEARCH_CACHE_ENABLED="1")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.record.read_text().splitlines()[:3], [task, "0", "0"])
                self.assertFalse(self.proxy.exists())

    def test_qa_can_start_without_optional_proxy_credentials(self):
        result = self.launch(TASK="qa")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.proxy.exists())
        self.assertEqual(self.record.read_text().splitlines()[0], "qa")

    def test_qa_auto_proxy_uses_configured_credentials(self):
        result = self.launch(TASK="qa", PROXY_HOST="proxy.test", PROXY_TOKEN="test-placeholder")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.proxy.exists())
        self.assertNotIn("test-placeholder", result.stdout + result.stderr)

    def test_qa_explicit_proxy_disable_is_preserved(self):
        result = self.launch(TASK="qa", SEARCH_PROXY_ENABLED="0",
                             PROXY_HOST="proxy.test", PROXY_TOKEN="test-placeholder")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.proxy.exists())

    def test_invalid_task_stops_before_components_and_training(self):
        result = self.launch(TASK="bad task")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.proxy.exists())
        self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
