"""No SSH/network operations: exercise the actual training bootstrap with stubs."""
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cache_bootstrap_under_test", ROOT / "train-roma/search_cache_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bootstrap
spec.loader.exec_module(bootstrap)


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="search-boot-", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.env = {
            "SEARCH_CACHE_BASE_URL": "http://127.0.0.1:8091",
            "SEARCH_CACHE_TOKEN": "private-shared-token",
            "SEARCH_CACHE_RUNTIME_DIR": self.directory.name,
        }
        self.config = bootstrap.Config.from_env(self.env)

    def test_default_configuration_and_explicit_flags_are_independent(self):
        self.assertTrue(self.config.auto_start)
        self.assertTrue(self.config.auto_tunnel)
        self.assertEqual((self.config.ssh_target, self.config.ssh_port), ("ma-user@7.150.11.99", 31753))
        self.assertEqual(self.config.local_port, 8091)
        direct = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_BASE_URL="http://development.test:8092"))
        self.assertFalse(direct.auto_tunnel)
        tunnel_only = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_AUTO_START="0", SEARCH_CACHE_AUTO_TUNNEL="1"))
        self.assertFalse(tunnel_only.auto_start)
        self.assertTrue(tunnel_only.auto_tunnel)

    def test_forward_uses_exact_loopback_and_remote_bind_interface(self):
        for base, service_host, expected in [
            ("http://127.0.0.2:8091", "10.2.3.4", "127.0.0.2:8091:10.2.3.4:8091"),
            ("http://[::1]:8091", "::", "[::1]:8091:[::1]:8091"),
            ("http://localhost:8091", "0.0.0.0", "127.0.0.1:8091:127.0.0.1:8091"),
        ]:
            config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_BASE_URL=base,
                                                    SEARCH_SERVICE_HOST=service_host,
                                                    SEARCH_CACHE_AUTO_START="0"))
            healthy, calls = [False], []
            def runner(command, *, timeout):
                calls.append(command)
                healthy[0] = True
                return completed()
            bootstrap.bootstrap(config, probe=lambda *_a, **_k: healthy[0], runner=runner)
            command = calls[0]
            self.assertEqual(command[command.index("-L") + 1], expected)

    def test_warm_service_reuse_has_no_remote_start_or_forward(self):
        runner = Mock(side_effect=AssertionError("warm cache must be reused"))
        result = bootstrap.bootstrap(self.config, probe=Mock(return_value=True), runner=runner)
        self.assertTrue(result["reused"])
        runner.assert_not_called()

    def test_explicit_identity_applies_to_remote_start_tunnel_and_control_commands(self):
        key = Path(self.directory.name) / "dummy identity.pem"
        key.write_text("dummy fixture, not a private key")
        key.chmod(0o600)
        config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_SSH_IDENTITY_FILE=str(key)))
        bootstrap._socket_path(config, Path(self.directory.name)).touch()
        healthy, calls = [False], []
        def runner(command, *, timeout):
            calls.append(command)
            if "-O" in command:
                return completed(returncode=255)
            if "-fN" in command:
                healthy[0] = True
                return completed()
            return completed('{"status":"ok","port":8091,"reused":true}')
        bootstrap.bootstrap(config, runner=runner, probe=lambda *_a, **_k: healthy[0])
        self.assertEqual(len(calls), 3)
        for command in calls:
            self.assertEqual(command[command.index("-i") + 1], str(key.resolve()))
            self.assertIn("IdentitiesOnly=yes", command)
            self.assertIn("BatchMode=yes", command)
            self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)

    def test_identity_relative_paths_are_repo_rooted_and_tilde_expands(self):
        key = Path(self.directory.name) / "pem" / "dummy with spaces.pem"
        key.parent.mkdir()
        key.write_text("dummy fixture")
        key.chmod(0o400)
        config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_SSH_IDENTITY_FILE="pem/dummy with spaces.pem"))
        with patch.object(bootstrap, "REPO_ROOT", Path(self.directory.name)), patch.dict(
            os.environ, {"HOME": self.directory.name}
        ):
            options = bootstrap.ssh_options(config)
            self.assertEqual(options[options.index("-i") + 1], str(key.resolve()))
            self.assertEqual(bootstrap.validate_ssh_identity_file("~/pem/dummy with spaces.pem"), str(key.resolve()))
        self.assertEqual(key.stat().st_mode & 0o777, 0o400)

    def test_missing_unreadable_directory_or_permissive_identity_fails_before_ssh(self):
        key = Path(self.directory.name) / "dummy.pem"
        key.write_text("dummy fixture")
        runner = Mock(side_effect=AssertionError("invalid key must not invoke SSH"))
        for path in (str(Path(self.directory.name) / "missing.pem"), self.directory.name):
            config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_SSH_IDENTITY_FILE=path))
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.bootstrap(config, runner=runner, probe=Mock(return_value=False))
        key.chmod(0o644)
        config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_SSH_IDENTITY_FILE=str(key)))
        with self.assertRaisesRegex(bootstrap.BootstrapError, "chmod 600"):
            bootstrap.bootstrap(config, runner=runner, probe=Mock(return_value=False))
        self.assertEqual(key.stat().st_mode & 0o777, 0o644)
        key.chmod(0o600)
        with patch.object(bootstrap.os, "access", return_value=False), self.assertRaisesRegex(
            bootstrap.BootstrapError, "readable regular file"
        ):
            bootstrap.bootstrap(config, runner=runner, probe=Mock(return_value=False))
        runner.assert_not_called()

    def test_warm_reuse_does_not_require_configured_key_to_exist(self):
        config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_SSH_IDENTITY_FILE="pem/no-longer-present.pem"))
        runner = Mock(side_effect=AssertionError("warm reuse must not invoke SSH"))
        result = bootstrap.bootstrap(config, runner=runner, probe=Mock(return_value=True))
        self.assertTrue(result["reused"])
        runner.assert_not_called()

    def test_unconfigured_identity_preserves_ssh_agent_and_default_key_selection(self):
        options = bootstrap.ssh_options(self.config)
        self.assertNotIn("-i", options)
        self.assertNotIn("IdentitiesOnly=yes", options)

    def test_cold_service_starts_remote_then_forward_with_bounded_ssh(self):
        calls, healthy = [], [False]
        def runner(command, *, timeout):
            calls.append((command, timeout))
            if "-fN" in command:
                healthy[0] = True
                return completed()
            return completed(json.dumps({"status": "ok", "port": 8091, "reused": False}))
        result = bootstrap.bootstrap(self.config, runner=runner, probe=lambda *_args, **_kw: healthy[0])
        self.assertFalse(result["reused"])
        self.assertEqual(len(calls), 2)
        ensure, tunnel = calls[0][0], calls[1][0]
        self.assertIn("ensure_search_cache_service.sh", ensure[-1])
        self.assertIn("-fN", tunnel)
        self.assertIn("ExitOnForwardFailure=yes", tunnel)
        self.assertIn("StrictHostKeyChecking=yes", tunnel)
        self.assertEqual(tunnel[tunnel.index("-L") + 1], "127.0.0.1:8091:127.0.0.1:8091")
        self.assertTrue(all(0 < timeout <= self.config.bootstrap_timeout for _, timeout in calls))
        self.assertNotIn(self.config.token, " ".join(ensure + tunnel))
        self.assertEqual(os.stat(self.directory.name).st_mode & 0o777, 0o700)

    def test_start_disabled_can_create_tunnel_without_running_remote_helper(self):
        config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_AUTO_START="0"))
        healthy, calls = [False], []
        def runner(command, *, timeout):
            calls.append(command)
            healthy[0] = True
            return completed()
        result = bootstrap.bootstrap(config, probe=lambda *_a, **_k: healthy[0], runner=runner)
        self.assertTrue(result["reused"])
        self.assertEqual(len(calls), 1)
        self.assertIn("-fN", calls[0])
        self.assertNotIn("ensure_search_cache_service.sh", " ".join(calls[0]))

    def test_both_disabled_only_checks_health_and_fails_if_unreachable(self):
        config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_AUTO_START="0", SEARCH_CACHE_AUTO_TUNNEL="0"))
        runner = Mock(side_effect=AssertionError("disabled modes must not run ssh"))
        with self.assertRaisesRegex(bootstrap.BootstrapError, "disabled"):
            bootstrap.bootstrap(config, probe=Mock(return_value=False), runner=runner)
        runner.assert_not_called()

    def test_direct_service_autostart_does_not_create_tunnel(self):
        config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_BASE_URL="http://development.test:8091"))
        healthy, calls = [False], []
        def runner(command, *, timeout):
            calls.append(command)
            healthy[0] = True
            return completed('{"status":"ok","port":8091,"reused":true}')
        result = bootstrap.bootstrap(config, probe=lambda *_a, **_k: healthy[0], runner=runner)
        self.assertTrue(result["reused"])
        self.assertFalse(result["tunnel"])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("-fN", calls[0])

    def test_concurrent_training_starts_reuse_single_remote_ensure_and_tunnel(self):
        healthy, calls = [False], []
        def runner(command, *, timeout):
            calls.append(command)
            time.sleep(0.03)
            if "-fN" in command:
                healthy[0] = True
                return completed()
            return completed('{"status":"ok","port":8091,"reused":false}')
        results, errors = [], []
        def task():
            try:
                results.append(bootstrap.bootstrap(self.config, runner=runner,
                                                   probe=lambda *_a, **_k: healthy[0]))
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=task) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 4)
        self.assertEqual(sum("-fN" in command for command in calls), 1)
        self.assertEqual(len(calls), 2)

    def test_remote_command_quotes_spaces_and_shell_metacharacters(self):
        path = "/home/user/a b;$(touch unwanted)'quoted"
        config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_REMOTE_REPO_DIR=path,
                                                SEARCH_CACHE_DIR=path + "/cache"))
        command = bootstrap.remote_command(config, 90)
        args = shlex.split(command)
        self.assertEqual(args[0], "env")
        self.assertIn("SEARCH_CACHE_DIR=" + path + "/cache", args)
        self.assertEqual(args[-2:], ["bash", path + "/train-roma/ensure_search_cache_service.sh"])
        self.assertNotIn(config.token, command)
        self.assertNotIn("TOKEN=", command)

    def test_wrong_token_never_restarts_the_service(self):
        runner = Mock()
        probe = Mock(side_effect=bootstrap.BootstrapError("authentication failed"))
        with self.assertRaisesRegex(bootstrap.BootstrapError, "authentication"):
            bootstrap.bootstrap(self.config, runner=runner, probe=probe)
        runner.assert_not_called()

    def test_remote_ssh_failure_is_explicit_and_does_not_echo_output(self):
        runner = Mock(return_value=completed("private-shared-token", 255, "private-shared-token"))
        with self.assertRaises(bootstrap.BootstrapError) as caught:
            bootstrap.bootstrap(self.config, probe=Mock(return_value=False), runner=runner)
        self.assertIn("SSH exit 255", str(caught.exception))
        self.assertNotIn(self.config.token, str(caught.exception))
        self.assertEqual(runner.call_count, 1)

    def test_remote_helper_wrong_port_or_json_fails_before_tunnel(self):
        for result in ("not json", '{"status":"ok","port":8000,"reused":false}', '{"status":"ok","port":8091,"reused":"true"}'):
            runner = Mock(return_value=completed(result))
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.bootstrap(self.config, probe=Mock(return_value=False), runner=runner)
            self.assertEqual(runner.call_count, 1)

    def test_stale_control_socket_is_checked_then_replaced(self):
        control_path = bootstrap._socket_path(self.config, Path(self.directory.name))
        control_path.touch()
        healthy, calls = [False], []
        def runner(command, *, timeout):
            calls.append(command)
            if "-O" in command:
                return completed(returncode=255)
            if "-fN" in command:
                self.assertFalse(control_path.exists())
                healthy[0] = True
                return completed()
            return completed('{"status":"ok","port":8091,"reused":true}')
        bootstrap.bootstrap(self.config, runner=runner, probe=lambda *_a, **_k: healthy[0])
        self.assertEqual(len(calls), 3)
        self.assertIn("check", calls[1])

    def test_running_control_master_with_unreachable_backend_is_not_duplicated(self):
        bootstrap._socket_path(self.config, Path(self.directory.name)).touch()
        calls = []
        def runner(command, *, timeout):
            calls.append(command)
            return completed() if "-O" in command else completed('{"status":"ok","port":8091,"reused":true}')
        with self.assertRaisesRegex(bootstrap.BootstrapError, "Existing SSH"):
            bootstrap.bootstrap(self.config, runner=runner, probe=Mock(return_value=False))
        self.assertFalse(any("-fN" in command for command in calls))

    def test_invalid_configuration_is_bounded_and_hides_token(self):
        for value in ("inf", "nan", "0", "bad"):
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_BOOTSTRAP_TIMEOUT_SECONDS=value))
        for target in ("-oProxyCommand=bad", "user@host; touch bad"):
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_SSH_TARGET=target))
        self.assertNotIn(self.config.token, repr(self.config))

    def test_unreachable_service_and_lock_wait_have_total_deadlines(self):
        config = bootstrap.Config.from_env(dict(self.env, SEARCH_CACHE_BOOTSTRAP_TIMEOUT_SECONDS="0.03"))
        runner = Mock(return_value=completed('{"status":"ok","port":8091,"reused":true}'))
        started = time.monotonic()
        with self.assertRaises(bootstrap.BootstrapError):
            bootstrap.bootstrap(config, probe=Mock(return_value=False), runner=runner)
        self.assertLess(time.monotonic() - started, 0.3)
        errors = []
        with bootstrap.node_lock(config, time.monotonic() + 1):
            def blocked_start():
                try:
                    bootstrap.bootstrap(config, probe=Mock(return_value=False), runner=runner)
                except bootstrap.BootstrapError as error:
                    errors.append(error)
            thread = threading.Thread(target=blocked_start)
            thread.start()
            thread.join(timeout=0.3)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)


class HTTPAndProcessTests(unittest.TestCase):
    def setUp(self):
        self.config = bootstrap.Config.from_env({
            "SEARCH_CACHE_BASE_URL": "http://127.0.0.1:8091", "SEARCH_CACHE_TOKEN": "secret-token",
        })

    def test_health_uses_direct_connection_authorization_and_no_redirect(self):
        opener = Mock()
        opener.open.return_value = io.BytesIO(b'{"status":"ok"}')
        with patch.object(bootstrap, "build_opener", return_value=opener) as build:
            self.assertTrue(bootstrap.probe_health(self.config, timeout=2))
        handlers = build.call_args.args
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], bootstrap._NoRedirect)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8091/healthz")
        self.assertEqual(request.headers["Authorization"], "Bearer secret-token")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 2)

    def test_health_auth_http_and_bad_json_fail_without_exposing_response(self):
        for status in (401, 403, 404):
            opener = Mock()
            opener.open.side_effect = HTTPError(self.config.base_url, status, "secret-token", {}, None)
            with patch.object(bootstrap, "build_opener", return_value=opener), self.assertRaises(bootstrap.BootstrapError) as caught:
                bootstrap.probe_health(self.config, timeout=1)
            self.assertNotIn(self.config.token, str(caught.exception))
        opener = Mock()
        opener.open.return_value = io.BytesIO(b"secret-token bad JSON")
        with patch.object(bootstrap, "build_opener", return_value=opener), self.assertRaises(bootstrap.BootstrapError) as caught:
            bootstrap.probe_health(self.config, timeout=1)
        self.assertNotIn(self.config.token, str(caught.exception))

    def test_unreachable_health_returns_false(self):
        opener = Mock()
        opener.open.side_effect = URLError("connection refused")
        with patch.object(bootstrap, "build_opener", return_value=opener):
            self.assertFalse(bootstrap.probe_health(self.config, timeout=1))

    def test_ssh_has_no_secret_argv_environment_or_stdin_and_has_timeout(self):
        with patch.dict(os.environ, {"SEARCH_CACHE_TOKEN": "secret-token", "YIBU_BRAVE_API_KEY": "upstream-key"}), patch.object(
            bootstrap.subprocess, "run", return_value=completed()
        ) as run:
            bootstrap.run_command(["ssh", "host", "true"], timeout=20)
        self.assertNotIn("secret-token", " ".join(run.call_args.args[0]))
        self.assertNotIn("SEARCH_CACHE_TOKEN", run.call_args.kwargs["env"])
        self.assertNotIn("YIBU_BRAVE_API_KEY", run.call_args.kwargs["env"])
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs["timeout"], 20)

    def test_tunnel_and_control_commands_do_not_capture_persistent_process_pipes(self):
        for command in (["ssh", "-fN", "host"], ["ssh", "-O", "check", "host"]):
            with patch.object(bootstrap.subprocess, "run", return_value=completed()) as run:
                bootstrap.run_command(command, timeout=1)
            self.assertEqual(run.call_args.kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(run.call_args.kwargs["stderr"], subprocess.DEVNULL)

    @unittest.skipUnless(hasattr(os, "fork"), "POSIX background-process descriptor check")
    def test_real_background_child_with_inherited_stdout_does_not_block_tunnel_start(self):
        # Reproduce ssh -fN: parent exits, a detached child keeps stdout/stderr.
        # With PIPE this would wait 0.8s for EOF and hit the 0.3s run timeout.
        code = "import os,time; pid=os.fork(); time.sleep(0.8) if pid==0 else None; os._exit(0)"
        started = time.monotonic()
        result = bootstrap.run_command([sys.executable, "-c", code, "-fN"], timeout=0.3)
        self.assertEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started, 0.3)

    def test_missing_ssh_and_process_timeout_have_clean_diagnostics(self):
        for error in (FileNotFoundError("secret-token"), subprocess.TimeoutExpired(["ssh"], 1, output="secret-token")):
            with patch.object(bootstrap.subprocess, "run", side_effect=error), self.assertRaises(bootstrap.BootstrapError) as caught:
                bootstrap.run_command(["ssh"], timeout=1)
            self.assertNotIn(self.config.token, str(caught.exception))

    def test_real_local_http_health_reuses_service_and_rejects_wrong_token(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                authenticated = self.headers.get("Authorization") == "Bearer secret-token"
                self.send_response(200 if authenticated else 401)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}' if authenticated else b'{"error":"unauthorized"}')
            def log_message(self, *_args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        runner = Mock(side_effect=AssertionError("local healthy service must not use SSH"))
        try:
            env = {"SEARCH_CACHE_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                   "SEARCH_CACHE_TOKEN": "secret-token"}
            result = bootstrap.bootstrap(bootstrap.Config.from_env(env), runner=runner)
            self.assertTrue(result["reused"])
            env["SEARCH_CACHE_TOKEN"] = "wrong-token"
            with self.assertRaisesRegex(bootstrap.BootstrapError, "authentication failed"):
                bootstrap.bootstrap(bootstrap.Config.from_env(env), runner=runner)
            runner.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
