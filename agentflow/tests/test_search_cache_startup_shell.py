"""Test startup configuration without launching training, SSH or model servers."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class StartupShellTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="search startup shell ")
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name) / "repo"
        self.scripts = self.repo / "train-roma"
        self.scripts.mkdir(parents=True)
        self.helper = self.scripts / "enable_search_cache.sh"
        shutil.copy2(ROOT / "train-roma/enable_search_cache.sh", self.helper)
        for filename in ("task_network.sh", "task_network.py", "enable_search_proxy.sh"):
            shutil.copy2(ROOT / "train-roma" / filename, self.scripts / filename)
        (self.scripts / "config.yaml").write_text("env:\n  TASK: qa\n")
        self.env_file = self.scripts / "search-cache.local.env"
        self.record = self.repo / "bootstrap.json"
        self.runner = self.repo / "fake-python"
        self.runner.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib\n"
            "pathlib.Path(os.environ['TEST_BOOTSTRAP_RECORD']).write_text(json.dumps({\n"
            "    key: value for key, value in os.environ.items()\n"
            "    if key.startswith(('SEARCH_CACHE_', 'SEARCH_SERVICE_'))\n"
            "}))\n"
        )
        self.runner.chmod(0o700)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("SEARCH_CACHE_", "SEARCH_SERVICE_"))}
        self.env.update(TASK="qa", TMPDIR=str(self.repo), SEARCH_CACHE_PYTHON=str(self.runner), TEST_BOOTSTRAP_RECORD=str(self.record))

    def source(self, *, env=None, prefix="set -eu;", suffix=""):
        return subprocess.run(
            ["bash", "-c", prefix + ' source "$1"; ' + suffix, "bash", str(self.helper)],
            env={**self.env, **(env or {})}, capture_output=True, text=True, timeout=5,
        )

    def captured(self):
        return json.loads(self.record.read_text())

    def test_explicit_disable_overrides_enabled_file_without_bootstrap(self):
        self.env_file.write_text('SEARCH_CACHE_ENABLED=1\nSEARCH_CACHE_TOKEN=file-token\n')
        result = self.source(env={"SEARCH_CACHE_ENABLED": "0"}, suffix='test "$SEARCH_CACHE_ENABLED" = 0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.record.exists())

    def test_platform_configuration_wins_over_file(self):
        self.env_file.write_text(
            'SEARCH_CACHE_TOKEN=file-token\nSEARCH_CACHE_BASE_URL=http://file:1234\n'
            'SEARCH_CACHE_AUTO_START=1\nSEARCH_CACHE_AUTO_TUNNEL=1\nSEARCH_SERVICE_PORT=8091\n'
            'SEARCH_CACHE_SSH_IDENTITY_FILE=pem/file.pem\n'
            'SEARCH_CACHE_SSH_AUTO_PREPARE=1\n'
        )
        overrides = {
            "SEARCH_CACHE_TOKEN": "platform-token", "SEARCH_CACHE_BASE_URL": "http://127.0.0.1:9001",
            "SEARCH_CACHE_AUTO_START": "0", "SEARCH_CACHE_AUTO_TUNNEL": "0", "SEARCH_SERVICE_PORT": "9002",
            "SEARCH_CACHE_SSH_IDENTITY_FILE": "pem/platform key.pem",
            "SEARCH_CACHE_SSH_AUTO_PREPARE": "0",
        }
        result = self.source(env=overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name, value in overrides.items():
            self.assertEqual(self.captured()[name], value)
        self.assertNotIn("platform-token", result.stdout + result.stderr)

    def test_file_configuration_is_exported(self):
        self.env_file.write_text('SEARCH_CACHE_TOKEN=file-token\nSEARCH_CACHE_AUTO_START=0\n')
        result = self.source(suffix='case "$-" in *a*) exit 7;; esac')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.captured()["SEARCH_CACHE_TOKEN"], "file-token")
        self.assertEqual(self.captured()["SEARCH_CACHE_AUTO_START"], "0")
        self.assertEqual(self.captured()["SEARCH_CACHE_ENABLED"], "1")

    def test_default_url_uses_configured_local_port(self):
        result = self.source(env={"SEARCH_CACHE_TOKEN": "test-token", "SEARCH_CACHE_LOCAL_PORT": "9123"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.captured()["SEARCH_CACHE_BASE_URL"], "http://127.0.0.1:9123")
        self.assertEqual(self.captured()["SEARCH_CACHE_SSH_AUTO_PREPARE"], "1")
        self.assertEqual(self.captured()["SEARCH_CACHE_SSH_IDENTITY_FILE"], "pem/h50065774.pem")

    def test_explicit_empty_identity_keeps_ssh_agent_available(self):
        result = self.source(env={"SEARCH_CACHE_TOKEN": "test-token", "SEARCH_CACHE_SSH_IDENTITY_FILE": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.captured()["SEARCH_CACHE_SSH_IDENTITY_FILE"], "")

    def test_missing_token_fails_before_bootstrap(self):
        result = self.source()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set SEARCH_CACHE_TOKEN", result.stderr)
        self.assertFalse(self.record.exists())

    def test_allexport_caller_option_is_preserved(self):
        self.env_file.write_text('SEARCH_CACHE_TOKEN=file-token\n')
        result = self.source(prefix="set -eua;", suffix='case "$-" in *a*) :;; *) exit 7;; esac')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_custom_environment_file(self):
        custom = self.repo / "platform configuration.env"
        custom.write_text('SEARCH_CACHE_TOKEN=custom-token\nSEARCH_CACHE_AUTO_TUNNEL=0\n')
        result = self.source(env={"SEARCH_CACHE_ENV_FILE": str(custom)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.captured()["SEARCH_CACHE_TOKEN"], "custom-token")

    def test_launchers_respect_explicit_disable_and_detect_local_config(self):
        # Execute only the safe activation block, never the full launchers'
        # process cleanup, dependency installation or training commands.
        self.env_file.write_text('SEARCH_CACHE_TOKEN=test-token\n')
        for filename in ("run_train.sh", "run_distribute_train.sh", "run_train_forever.sh"):
            script = (ROOT / "train-roma" / filename).read_text()
            marker = 'if agentflow_task_needs_search && {'
            block = script[script.index(marker):].split("\nfi", 1)[0] + "\nfi\n"
            for enabled in ("0", "false", None, "1"):
                with self.subTest(filename=filename, enabled=enabled):
                    self.record.unlink(missing_ok=True)
                    env = {**self.env, "ROOT_PATH": str(self.repo)}
                    if enabled is not None:
                        env["SEARCH_CACHE_ENABLED"] = enabled
                    result = subprocess.run(["bash", "-c", 'set -eu; source "${ROOT_PATH}/train-roma/task_network.sh"; agentflow_resolve_task;\n' + block], env=env,
                                            capture_output=True, text=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(self.record.exists(), enabled in (None, "1"))


    def test_local_tasks_skip_enabled_env_file_and_missing_key_without_sourcing_it(self):
        marker = self.repo / "env-was-sourced"
        self.env_file.write_text('SEARCH_CACHE_ENABLED=1\n'
                                 + f'touch "{marker}"\n'
                                 + 'SEARCH_CACHE_SSH_IDENTITY_FILE=/missing/private-key\n')
        for task in ("alfworld", "webshop", "rule"):
            with self.subTest(task=task):
                result = self.source(env={"TASK": task, "SEARCH_CACHE_ENABLED": "1"},
                                     suffix='test "$SEARCH_CACHE_ENABLED" = 0')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(marker.exists())
                self.assertFalse(self.record.exists())

    def test_local_task_from_config_is_resolved_before_cache_environment_file(self):
        (self.scripts / "config.yaml").write_text("env:\n  TASK: alfworld\n")
        self.env_file.write_text('exit 99\n')
        env = dict(self.env)
        env.pop("TASK", None)
        result = subprocess.run(["bash", "-c", 'set -eu; source "$1"; test "$TASK" = alfworld; test "$SEARCH_CACHE_ENABLED" = 0',
                                 "bash", str(self.helper)], env=env, cwd=self.repo.parent,
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.record.exists())

    def test_proxy_skips_local_tasks_before_credentials_or_python_patch(self):
        proxy = self.scripts / "enable_search_proxy.sh"
        env = {key: value for key, value in self.env.items() if key not in ("PROXY_HOST", "PROXY_TOKEN", "PROXY_PY_DIR")}
        env.update(HTTP_PROXY="http://user-proxy.invalid:1234", PYTHONPATH="/user/python/path")
        for task in ("alfworld", "webshop", "rule"):
            with self.subTest(task=task):
                result = subprocess.run(["bash", "-c", 'set -eu; source "$1"; test "$SEARCH_CACHE_ENABLED" = 0; '
                                         'test "$HTTP_PROXY" = http://user-proxy.invalid:1234; '
                                         'test "$PYTHONPATH" = /user/python/path; test -z "${PROXY_PY_DIR:-}"',
                                         "bash", str(proxy)], env={**env, "TASK": task},
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("Search Proxy 已启用", result.stdout)

    def test_managed_proxy_hook_is_removed_when_switching_to_local_task(self):
        proxy = self.scripts / "enable_search_proxy.sh"
        result = subprocess.run(["bash", "-c", 'set -eu; source "$1"; '
                                 'test "$AGENTFLOW_SEARCH_PROXY_ACTIVE" = 1; '
                                 'test -f "$PROXY_PY_DIR/sitecustomize.py"; '
                                 'TASK=alfworld; source "$2"; '
                                 'test "$SEARCH_CACHE_ENABLED" = 0; '
                                 'test "$PYTHONPATH" = /user/python/path; '
                                 'test -z "${HTTP_PROXY:-}"; test -z "${HTTPS_PROXY:-}"',
                                 "bash", str(proxy), str(self.helper)],
                                env={**self.env, "TASK": "qa", "PROXY_HOST": "proxy.invalid", "PROXY_TOKEN": "test-only",
                                     "PYTHONPATH": "/user/python/path"}, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repeated_proxy_loading_reuses_one_managed_python_hook(self):
        proxy = self.scripts / "enable_search_proxy.sh"
        result = subprocess.run(["bash", "-c", 'set -eu; source "$1"; first_hook="$PROXY_PY_DIR"; '
                                 'first_path="$PYTHONPATH"; source "$1"; '
                                 'test "$PROXY_PY_DIR" = "$first_hook"; test "$PYTHONPATH" = "$first_path"; '
                                 'TASK=webshop; source "$2"; test "$PYTHONPATH" = /user/python/path',
                                 "bash", str(proxy), str(self.helper)],
                                env={**self.env, "TASK": "qa", "PROXY_HOST": "proxy.invalid", "PROXY_TOKEN": "test-only",
                                     "PYTHONPATH": "/user/python/path"}, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_search_service_proxy_ignores_training_task(self):
        proxy = self.scripts / "enable_search_proxy.sh"
        result = subprocess.run(["bash", "-c", 'set -eu; source "$1" --service; '
                                 'test "$AGENTFLOW_SEARCH_PROXY_ACTIVE" = 1; test -f "$PROXY_PY_DIR/sitecustomize.py"',
                                 "bash", str(proxy)], env={**self.env, "TASK": "alfworld", "PROXY_HOST": "proxy.invalid", "PROXY_TOKEN": "test-only"},
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_launcher_headers_resolve_task_and_root_from_another_directory(self):
        (self.scripts / "config.yaml").write_text("env:\n  TASK: alfworld\n")
        for filename in ("run_train.sh", "run_distribute_train.sh", "run_train_forever.sh"):
            script = (ROOT / "train-roma" / filename).read_text()
            # Run only the task/root header, before cleanup, installation or services.
            header = script[:script.index("\nfi") + len("\nfi")]
            path = self.scripts / filename
            path.write_text(header + '\nprintf "%s\\n" "$TASK" "$ROOT_PATH" "$SEARCH_CACHE_ENABLED"\n')
            for task in ("webshop", None):
                with self.subTest(filename=filename, task=task):
                    env = {**self.env, "SEARCH_CACHE_ENABLED": "1"}
                    if task is None:
                        env.pop("TASK", None)
                    else:
                        env["TASK"] = task
                    result = subprocess.run(["bash", str(path)], env=env, cwd=self.repo.parent,
                                            capture_output=True, text=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.splitlines()[-3:], [task or "alfworld", str(self.repo), "0"])
                    self.assertFalse(self.record.exists())

    def test_launchers_disable_search_for_local_tasks_even_with_stale_enabled_config(self):
        self.env_file.write_text('SEARCH_CACHE_ENABLED=1\nexit 99\n')
        for filename in ("run_train.sh", "run_distribute_train.sh", "run_train_forever.sh"):
            script = (ROOT / "train-roma" / filename).read_text()
            marker = 'if agentflow_task_needs_search && {'
            block = script[script.index(marker):].split("\nfi", 1)[0] + "\nfi\n"
            for task in ("alfworld", "webshop"):
                with self.subTest(filename=filename, task=task):
                    result = subprocess.run(["bash", "-c", 'set -eu; source "${ROOT_PATH}/train-roma/task_network.sh"; '
                                             'agentflow_resolve_task;\n' + block],
                                            env={**self.env, "TASK": task, "ROOT_PATH": str(self.repo), "SEARCH_CACHE_ENABLED": "1"},
                                            capture_output=True, text=True, timeout=5)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
