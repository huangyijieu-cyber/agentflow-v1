"""Safely test the distributed launcher header/cache gates without executing Ray or training."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'train-roma' / 'run_distribute_train.sh'


class DistributedNetworkRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='network-gate ')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.repo = root / 'repository with spaces'
        self.scripts = self.repo / 'train-roma'
        self.scripts.mkdir(parents=True)
        self.other_cwd = root / 'elsewhere'
        self.other_cwd.mkdir()
        self.record = root / 'network-record'
        self.config = self.scripts / 'config.yaml'
        self.config.write_text('env:\n  TASK: ${env:TASK,qa}\n')
        # Source stubs only. They must not perform any networking.
        (self.scripts / 'enable_search_proxy.sh').write_text(
            'echo proxy >> "${TEST_NETWORK_RECORD}"\n'
        )
        (self.scripts / 'enable_search_cache.sh').write_text(
            'echo cache >> "${TEST_NETWORK_RECORD}"\n'
        )
        script = LAUNCHER.read_text(encoding='utf-8')
        self.assertNotIn('\r', script)
        header_end = script.index('#source train-roma/con_to_web.sh')
        cache_begin = script.index('if [[ "${NEED_SEARCH_NETWORK}" == 1 ]]; then', header_end)
        cache_end = script.index('\n# ==========================================', cache_begin)
        self.header = script[:header_end]
        self.cache = script[cache_begin:cache_end]
        self.env = {
            key: value for key, value in os.environ.items()
            if key not in {'TASK', 'TRAINING_TASK', 'SEARCH_CACHE_ENABLED', 'SEARCH_PROXY_ENABLED',
                           'SEARCH_CACHE_AUTO_START', 'SEARCH_CACHE_AUTO_TUNNEL',
                           'SEARCH_CACHE_ENV_FILE', 'AGENTFLOW_TASK_PYTHON'}
        }
        self.env['AGENTFLOW_TASK_PYTHON'] = sys.executable
        self.env['TEST_NETWORK_RECORD'] = str(self.record)

    def run_header(self, *, task=None, cache=False, full=False, **extra):
        env = {**self.env, **extra}
        if task is not None:
            env['TASK'] = task
        if cache:
            env['SEARCH_CACHE_ENABLED'] = '1'
        cmd = self.header + (self.cache if full else '') + (
            '\nprintf "result:%s|%s|%s|%s\\n" "$TASK" "$ROOT_PATH" '
            '"${SEARCH_CACHE_ENABLED-unset}" "${SEARCH_PROXY_ENABLED-unset}"\n'
        )
        # The saved script is intentionally limited to the safe portion.
        temp_script = self.scripts / 'header-only.sh'
        temp_script.write_text(cmd, encoding='utf-8')
        return subprocess.run(['bash', str(temp_script)], env=env,
                              cwd=self.other_cwd, capture_output=True,
                              text=True, timeout=8)

    def test_local_tasks_never_load_search_components(self):
        (self.scripts / 'search-cache.local.env').write_text('exit 99\n')
        for task in ('alfworld', 'webshop', 'rule'):
            with self.subTest(task=task):
                self.record.unlink(missing_ok=True)
                result = self.run_header(task=task, cache=True, full=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'result:{task}|{self.repo}|0|0', result.stdout)
                self.assertFalse(self.record.exists())

    def test_yaml_literal_local_task_and_case_normalization(self):
        self.config.write_text('env:\n  TASK: "  ALFWorld  "\n')
        result = self.run_header(full=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('result:alfworld|', result.stdout)
        self.assertFalse(self.record.exists())

    def test_env_task_overrides_yaml_task(self):
        self.config.write_text('env:\n  TASK: qa\n')
        result = self.run_header(task=' WeBShOp ', cache=True, full=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('result:webshop|', result.stdout)
        self.assertFalse(self.record.exists())

    def test_default_qa_loads_proxy_and_optional_cache(self):
        result = self.run_header(full=True, cache=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('result:qa|', result.stdout)
        self.assertEqual(self.record.read_text().splitlines(), ['proxy', 'cache'])

    def test_qa_auto_loads_cache_when_config_file_is_present(self):
        (self.scripts / 'search-cache.local.env').write_text('SEARCH_CACHE_ENABLED=1\n')
        result = self.run_header(full=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.record.read_text().splitlines(), ['proxy', 'cache'])

    def test_qa_can_explicitly_disable_cache_only(self):
        (self.scripts / 'search-cache.local.env').write_text('SEARCH_CACHE_ENABLED=1\n')
        result = self.run_header(full=True, SEARCH_CACHE_ENABLED='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.record.read_text().splitlines(), ['proxy'])

    def test_launch_without_pyyaml_or_pre_activated_conda(self):
        # -S disables site-packages: PyYAML is deliberately unavailable here.
        # This mimics a ModelArts launcher before pip install or conda activation.
        wrapper = self.scripts / 'python-stdlib-only'
        wrapper.write_text(
            '#!/bin/sh\nexec "' + sys.executable + '" -S "$@"\n'
        )
        wrapper.chmod(0o755)
        self.config.write_text('env:\n  TASK: "  ALFWorld  " # comment\n')
        result = self.run_header(full=True, AGENTFLOW_TASK_PYTHON=str(wrapper))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('result:alfworld|', result.stdout)
        self.assertFalse(self.record.exists())

    def test_one_click_yaml_resolution_and_override_without_pyyaml(self):
        wrapper = self.scripts / 'python-stdlib-only'
        wrapper.write_text('#!/bin/sh\nexec "' + sys.executable + '" -S "$@"\n')
        wrapper.chmod(0o755)
        self.config.write_text('env:\n  TASK: ${env:TRAINING_TASK,qa} # inline comment\n')
        cases = (
            ({}, 'qa', ['proxy']),
            ({'TRAINING_TASK': ' webshop '}, 'webshop', []),
            ({'TASK': ' ALFWORLD '}, 'alfworld', []),
        )
        for env, task, calls in cases:
            with self.subTest(env=env):
                self.record.unlink(missing_ok=True)
                result = self.run_header(full=True, AGENTFLOW_TASK_PYTHON=str(wrapper), **env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'result:{task}|', result.stdout)
                self.assertEqual(self.record.read_text().splitlines() if self.record.exists() else [], calls)

    def test_yaml_default_env_fallback(self):
        self.config.write_text('env:\n  TASK: ${env:TRAINING_TASK,qa}\n')
        result = self.run_header(full=True, TRAINING_TASK=' rule ')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('result:rule|', result.stdout)
        self.assertFalse(self.record.exists())

    def test_invalid_tasks_or_yaml_fail_before_network(self):
        for task in ('', 'a bad task', 'qa;id'):
            with self.subTest(task=task):
                result = self.run_header(task=task, full=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.record.exists())
        for yaml_text in ('env: [not a mapping]\n', 'env: {TASK: qa\n', 'env:\n  TASK: true\n'):
            with self.subTest(yaml_text=yaml_text):
                self.config.write_text(yaml_text)
                result = self.run_header(full=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.record.exists())

    def test_missing_configuration_fails_even_if_task_specified(self):
        self.config.unlink()
        result = self.run_header(task='alfworld', full=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.record.exists())


if __name__ == '__main__':
    unittest.main()
