"""Remote helper tests using temporary local daemons, never SSH or the internet."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor


HELPER = Path(__file__).resolve().parents[2] / "train-roma/ensure_search_cache_service.sh"
FAKE_DAEMON = r'''
import json
import os
from pathlib import Path
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

with open(os.environ['TEST_STARTS_FILE'], 'a') as stream:
    stream.write(str(os.getpid()) + '\n')
delay = float(os.environ.get('TEST_BOOT_DELAY', '0'))
time.sleep(delay)
if os.environ.get('TEST_EXIT_EARLY') == '1':
    raise SystemExit(7)
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass
    def do_GET(self):
        forced = os.environ.get('TEST_HEALTH_STATUS')
        status = int(forced) if forced else 200 if self.headers.get('Authorization') == 'Bearer ' + os.environ['SEARCH_SERVICE_TOKEN'] else 401
        body = json.dumps({'status': 'ok'}).encode()
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
ThreadingHTTPServer((os.environ.get('SEARCH_SERVICE_HOST', '127.0.0.1'), int(os.environ['SEARCH_SERVICE_PORT'])), Handler).serve_forever()
'''


class AutostartHelperTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="cache helper test ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repo = self.root / "repository with spaces"
        self.scripts = self.repo / "train-roma"
        self.scripts.mkdir(parents=True)
        self.helper = self.scripts / "ensure_search_cache_service.sh"
        shutil.copy2(HELPER, self.helper)
        module = self.repo / "agentflow/agentflow/search_service/server.py"
        module.parent.mkdir(parents=True)
        module.write_text("# Fake source marker for bootstrap tests\n")
        self.fake_daemon = self.root / "fake daemon.py"
        self.fake_daemon.write_text(FAKE_DAEMON)
        (self.scripts / "run_search_cache.sh").write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\nexec "${SEARCH_SERVICE_PYTHON}" "${TEST_FAKE_DAEMON}"\n'
        )
        self.cache_dir = self.root / "persistent cache"
        self.cache_dir.mkdir()
        self.env_file = self.cache_dir / "search-service.env"
        self.env_file.write_text('SEARCH_SERVICE_TOKEN="unit-test-token"\nSEARCH_SERVICE_USE_PROXY=0\n')
        self.starts = self.root / "starts.txt"
        with socket.socket() as candidate:
            candidate.bind(("127.0.0.1", 0))
            self.port = candidate.getsockname()[1]
        self.env = os.environ.copy()
        for name in ("SEARCH_SERVICE_TOKEN", "SEARCH_CACHE_TOKEN"):
            self.env.pop(name, None)
        self.env.update(
            SEARCH_CACHE_DIR=str(self.cache_dir), SEARCH_SERVICE_ENV_FILE=str(self.env_file),
            SEARCH_SERVICE_HOST="127.0.0.1", SEARCH_SERVICE_PORT=str(self.port),
            SEARCH_SERVICE_PYTHON=sys.executable, SEARCH_SERVICE_STARTUP_TIMEOUT="3",
            SEARCH_SERVICE_AUTO_INSTALL_DEPS="0", TEST_FAKE_DAEMON=str(self.fake_daemon),
            TEST_STARTS_FILE=str(self.starts), HTTP_PROXY="http://127.0.0.1:1", HTTPS_PROXY="http://127.0.0.1:1",
        )
        self.addCleanup(self.stop_test_daemons)

    def stop_test_daemons(self):
        # Only tests terminate their own temporary daemons. The helper never stops one.
        if not self.starts.exists():
            return
        for line in self.starts.read_text().splitlines():
            try:
                os.kill(int(line), signal.SIGTERM)
            except ProcessLookupError:
                pass

    def invoke(self, **overrides):
        env = dict(self.env)
        env.update(overrides)
        return subprocess.run(["bash", str(self.helper)], env=env, capture_output=True, text=True, timeout=6)

    def success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("unit-test-token", result.stdout + result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["port"], self.port)
        return data

    def test_cold_start_detaches_and_later_invocations_reuse(self):
        first = self.success(self.invoke())
        second = self.success(self.invoke())
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(len(self.starts.read_text().splitlines()), 1)
        pid = int((self.cache_dir / "service.pid").read_text())
        os.kill(pid, 0)  # Helper exited; daemon still runs.
        self.assertTrue((self.cache_dir / "service.log").exists())

    def test_simultaneous_cold_invocations_start_exactly_one_daemon(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            replies = list(executor.map(lambda _: self.invoke(TEST_BOOT_DELAY="0.2"), range(2)))
        data = [self.success(result) for result in replies]
        self.assertEqual(sorted(item["reused"] for item in data), [False, True])
        self.assertEqual(len(self.starts.read_text().splitlines()), 1)

    def test_explicit_port_cache_and_python_override_environment_file_defaults(self):
        self.env_file.write_text('SEARCH_SERVICE_TOKEN="unit-test-token"\nSEARCH_SERVICE_PORT=1\nSEARCH_CACHE_DIR="/not/a/cache"\nSEARCH_SERVICE_PYTHON="/missing/python"\nSEARCH_SERVICE_STARTUP_TIMEOUT="nan"\n')
        self.success(self.invoke())
        self.assertTrue((self.cache_dir / "service.pid").exists())

    def test_bad_token_does_not_restart_or_terminate_existing_service(self):
        self.success(self.invoke())
        pid = int((self.cache_dir / "service.pid").read_text())
        self.env_file.write_text('SEARCH_SERVICE_TOKEN="wrong-test-token"\n')
        reply = self.invoke()
        self.assertNotEqual(reply.returncode, 0)
        self.assertIn("authentication rejected", reply.stderr)
        self.assertEqual(len(self.starts.read_text().splitlines()), 1)
        os.kill(pid, 0)
        self.assertNotIn("wrong-test-token", reply.stdout + reply.stderr)

    def test_missing_environment_credentials_fail_without_starting(self):
        self.env_file.unlink()
        reply = self.invoke()
        self.assertNotEqual(reply.returncode, 0)
        self.assertIn("environment file is missing", reply.stderr)
        self.assertFalse(self.starts.exists())

    def test_missing_source_fails_without_starting(self):
        (self.repo / "agentflow/agentflow/search_service/server.py").unlink()
        reply = self.invoke()
        self.assertNotEqual(reply.returncode, 0)
        self.assertIn("synchronize the cache branch", reply.stderr)
        self.assertFalse(self.starts.exists())

    def test_existing_live_pid_is_never_replaced_or_killed_when_unhealthy(self):
        (self.cache_dir / "service.pid").write_text(str(os.getpid()))
        started = time.monotonic()
        reply = self.invoke(SEARCH_SERVICE_STARTUP_TIMEOUT="0.2")
        self.assertNotEqual(reply.returncode, 0)
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertIn("no process was stopped", reply.stderr)
        self.assertFalse(self.starts.exists())
        self.assertEqual(int((self.cache_dir / "service.pid").read_text()), os.getpid())

    def test_startup_failure_reports_log_without_printing_contents(self):
        reply = self.invoke(TEST_EXIT_EARLY="1")
        self.assertNotEqual(reply.returncode, 0)
        self.assertIn("status=7", reply.stderr)
        self.assertIn("service.log", reply.stderr)
        self.assertNotIn("unit-test-token", reply.stdout + reply.stderr)

    def test_timeout_leaves_daemon_available_for_later_training(self):
        reply = self.invoke(TEST_BOOT_DELAY="0.5", SEARCH_SERVICE_STARTUP_TIMEOUT="0.1")
        self.assertNotEqual(reply.returncode, 0)
        pid = int((self.cache_dir / "service.pid").read_text())
        os.kill(pid, 0)
        time.sleep(0.6)
        response = self.success(self.invoke())
        self.assertTrue(response["reused"])
        self.assertEqual(len(self.starts.read_text().splitlines()), 1)

    def test_startup_lock_wait_is_in_the_same_bounded_deadline(self):
        with (self.cache_dir / "bootstrap.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            started = time.monotonic()
            reply = self.invoke(SEARCH_SERVICE_STARTUP_TIMEOUT="0.2")
            self.assertNotEqual(reply.returncode, 0)
            self.assertIn("startup lock", reply.stderr)
            self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(self.starts.exists())

    def test_missing_dependencies_explain_one_time_setup_without_installing(self):
        wrapper = self.root / "python without service dependencies"
        code = (
            "import importlib.util,sys\n"
            "original=importlib.util.find_spec\n"
            "importlib.util.find_spec=lambda name: None if name in {'requests','bs4'} else original(name)\n"
            "sys.argv=sys.argv[1:]\n"
            "exec(compile(sys.stdin.read(),'<bootstrap-test>','exec'))\n"
        )
        wrapper.write_text("#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable) + " -c " + shlex.quote(code) + ' "$@"\n')
        wrapper.chmod(0o700)
        reply = self.invoke(SEARCH_SERVICE_PYTHON=str(wrapper))
        self.assertNotEqual(reply.returncode, 0)
        self.assertIn("missing Python dependencies", reply.stderr)
        self.assertIn("pip install requests beautifulsoup4", reply.stderr)
        self.assertFalse(self.starts.exists())

    def test_nonfinite_timeout_is_rejected_without_starting(self):
        reply = self.invoke(SEARCH_SERVICE_STARTUP_TIMEOUT="nan")
        self.assertNotEqual(reply.returncode, 0)
        self.assertIn("finite and positive", reply.stderr)
        self.assertFalse(self.starts.exists())


if __name__ == "__main__":
    unittest.main()
