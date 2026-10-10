#!/usr/bin/env bash
# Run on the development server via SSH. Reuse or detach one shared service.
# Credentials stay in the server environment file, never in SSH arguments.
set -euo pipefail

_search_ensure_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
_search_ensure_cache_set="${SEARCH_CACHE_DIR+x}"
_search_ensure_cache="${SEARCH_CACHE_DIR:-/home/ma-user/work/code-rl/cache}"
_search_ensure_host_set="${SEARCH_SERVICE_HOST+x}"
_search_ensure_host="${SEARCH_SERVICE_HOST:-127.0.0.1}"
_search_ensure_port_set="${SEARCH_SERVICE_PORT+x}"
_search_ensure_port="${SEARCH_SERVICE_PORT:-8091}"
_search_ensure_python_set="${SEARCH_SERVICE_PYTHON+x}"
_search_ensure_python="${SEARCH_SERVICE_PYTHON:-python3}"
_search_ensure_timeout_set="${SEARCH_SERVICE_STARTUP_TIMEOUT+x}"
_search_ensure_timeout="${SEARCH_SERVICE_STARTUP_TIMEOUT:-90}"
_search_ensure_env="${SEARCH_SERVICE_ENV_FILE:-${_search_ensure_cache}/search-service.env}"

if [[ -f "${_search_ensure_env}" ]]; then
    set -a
    source "${_search_ensure_env}"
    set +a
elif [[ -z "${SEARCH_SERVICE_TOKEN:-${SEARCH_CACHE_TOKEN:-}}" ]]; then
    echo "Search service environment file is missing: ${_search_ensure_env}. Prepare it with SEARCH_SERVICE_TOKEN once on the development server." >&2
    exit 1
fi

# Explicit SSH configuration must identify the same port/cache used by the
# training client's tunnel, even when the environment file has old defaults.
if [[ -n "${_search_ensure_cache_set}" ]]; then SEARCH_CACHE_DIR="${_search_ensure_cache}"; fi
if [[ -n "${_search_ensure_host_set}" ]]; then SEARCH_SERVICE_HOST="${_search_ensure_host}"; fi
if [[ -n "${_search_ensure_port_set}" ]]; then SEARCH_SERVICE_PORT="${_search_ensure_port}"; fi
if [[ -n "${_search_ensure_python_set}" ]]; then SEARCH_SERVICE_PYTHON="${_search_ensure_python}"; fi
if [[ -n "${_search_ensure_timeout_set}" ]]; then SEARCH_SERVICE_STARTUP_TIMEOUT="${_search_ensure_timeout}"; fi
export SEARCH_CACHE_DIR="${SEARCH_CACHE_DIR:-${_search_ensure_cache}}"
export SEARCH_SERVICE_HOST="${SEARCH_SERVICE_HOST:-${_search_ensure_host}}"
export SEARCH_SERVICE_PORT="${SEARCH_SERVICE_PORT:-${_search_ensure_port}}"
export SEARCH_SERVICE_PYTHON="${SEARCH_SERVICE_PYTHON:-${_search_ensure_python}}"
export SEARCH_SERVICE_ENV_FILE="${_search_ensure_env}"
export SEARCH_SERVICE_ENV_LOADED=1
export SEARCH_SERVICE_STARTUP_TIMEOUT="${SEARCH_SERVICE_STARTUP_TIMEOUT:-90}"
export SEARCH_SERVICE_AUTO_INSTALL_DEPS="${SEARCH_SERVICE_AUTO_INSTALL_DEPS:-0}"

exec "${SEARCH_SERVICE_PYTHON}" - "${_search_ensure_repo}" <<'PY'
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request


def fail(message):
    print("Search service startup failed: " + message, file=sys.stderr)
    raise SystemExit(1)


repo = Path(sys.argv[1]).resolve()
cache_dir = Path(os.environ["SEARCH_CACHE_DIR"]).expanduser().resolve()
run_script = repo / "train-roma/run_search_cache.sh"
module_file = repo / "agentflow/agentflow/search_service/server.py"
token = os.environ.get("SEARCH_SERVICE_TOKEN") or os.environ.get("SEARCH_CACHE_TOKEN", "")
if not token or token != token.strip() or any(ch.isspace() for ch in token):
    fail("SEARCH_SERVICE_TOKEN is missing or invalid in the development server environment file")
try:
    port = int(os.environ["SEARCH_SERVICE_PORT"])
    timeout = float(os.environ["SEARCH_SERVICE_STARTUP_TIMEOUT"])
except (ValueError, TypeError):
    fail("SEARCH_SERVICE_PORT and SEARCH_SERVICE_STARTUP_TIMEOUT must be valid numbers")
if not 1 <= port <= 65535 or not math.isfinite(timeout) or timeout <= 0:
    fail("port must be 1..65535 and startup timeout must be finite and positive")
if not run_script.is_file() or not module_file.is_file():
    fail("cache service source is missing; synchronize the cache branch to the development repository once")
host = os.environ["SEARCH_SERVICE_HOST"]
if host in {"127.0.0.1", "0.0.0.0", "localhost"}:
    health_host = "127.0.0.1"
elif host in {"::", "::1"}:
    health_host = "[::1]"
else:
    # A shared service can bind a specific private interface as well.
    health_host = "[" + host + "]" if ":" in host else host
health_url = "http://" + health_host + ":" + str(port) + "/healthz"
deadline = time.monotonic() + timeout
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def remaining():
    return max(0.0, deadline - time.monotonic())


def healthy():
    if remaining() <= 0:
        return False
    request = urllib.request.Request(health_url, headers={"Authorization": "Bearer " + token})
    try:
        with opener.open(request, timeout=max(0.05, min(2.0, remaining()))) as response:
            if response.status != 200:
                fail("health endpoint returned HTTP " + str(response.status) + "; refusing to start another service")
            raw = response.read(16385)
            if len(raw) > 16384:
                fail("health response exceeds limit; check the configured service port")
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeError):
                fail("health endpoint did not return JSON; check the configured service port")
            if not isinstance(body, dict) or body.get("status") != "ok":
                fail("health endpoint is not the search service; check the configured service port")
            return True
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            fail("health authentication rejected (HTTP " + str(exc.code) + "); correct the server token, no process was restarted")
        if exc.code in {429, 503}:
            return False
        fail("health endpoint returned HTTP " + str(exc.code) + "; refusing to start another service")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return False


def alive(pid):
    if pid is None or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Never replace an uninspectable process or terminate a reused PID.
        return True


def read_pid(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def ready(reused):
    print(json.dumps({"status": "ok", "port": port, "reused": reused}, separators=(",", ":")))


try:
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = (cache_dir / "bootstrap.lock").open("a+")
except OSError:
    fail("cannot create cache directory/startup lock: " + str(cache_dir))
with lock:
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if remaining() <= 0:
                fail("timed out waiting for the shared service startup lock")
            time.sleep(min(0.1, remaining()))
    if healthy():
        ready(True)
        raise SystemExit(0)
    pid_path = cache_dir / "service.pid"
    log_path = cache_dir / "service.log"
    existing_pid = read_pid(pid_path)
    child = None
    reused = alive(existing_pid)
    if not reused:
        missing = [name for name in ("requests", "bs4") if importlib.util.find_spec(name) is None]
        if missing:
            fail("missing Python dependencies; prepare the configured Python environment once with: "
                 "python -m pip install requests beautifulsoup4 (SEARCH_SERVICE_PYTHON selects that environment)")
        if os.environ.get("SEARCH_SERVICE_AUTO_INSTALL_DEPS", "0") not in {"0", "false", "False"}:
            fail("automatic dependency installation is disabled; prepare the server Python environment once")
        try:
            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "ab", buffering=0) as log:
                child = subprocess.Popen(["bash", str(run_script)], cwd=repo, env=os.environ.copy(),
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                         start_new_session=True, close_fds=True)
            temporary_pid = pid_path.with_name(pid_path.name + ".tmp." + str(os.getpid()))
            fd = os.open(temporary_pid, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as pid_file:
                pid_file.write(str(child.pid) + "\n")
            os.replace(temporary_pid, pid_path)
        except OSError:
            fail("could not detach the search service; inspect permissions for " + str(cache_dir))
    while remaining() > 0:
        if healthy():
            ready(reused)
            raise SystemExit(0)
        if child is not None and child.poll() is not None:
            fail("service exited during startup (status=" + str(child.returncode) + "); inspect " + str(log_path))
        if child is None and not alive(existing_pid):
            fail("existing service exited before becoming healthy; inspect " + str(log_path))
        time.sleep(min(0.2, remaining()))
    # Don't stop the daemon: another training task can still use it when ready.
    fail("service did not become healthy before the startup deadline; inspect " + str(log_path)
         + "; no process was stopped")
PY
