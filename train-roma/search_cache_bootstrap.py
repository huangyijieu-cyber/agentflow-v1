#!/usr/bin/env python3
"""Reuse/start the development search service and a node-local SSH forward.

Only standard-library dependencies are needed on the training node. The shared
service and SSH master outlive training; this program never stops either one.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import shlex
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class BootstrapError(RuntimeError):
    """A bounded diagnostic which never includes credentials or command output."""


def _flag(env, name, default):
    value = str(env.get(name, "1" if default else "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise BootstrapError(f"{name} must be 0 or 1")


def _number(env, name, default, *, port=False):
    try:
        value = int(env.get(name, default)) if port else float(env.get(name, default))
    except (TypeError, ValueError):
        raise BootstrapError(f"{name} must be a valid {'port' if port else 'number'}") from None
    if not math.isfinite(value) or value <= 0 or (port and value > 65535):
        raise BootstrapError(f"{name} must be {'between 1 and 65535' if port else 'positive and finite'}")
    return value


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class Config:
    base_url: str
    token: str = field(repr=False)
    auto_start: bool = True
    auto_tunnel: bool = True
    ssh_target: str = "ma-user@7.150.11.99"
    ssh_port: int = 31753
    remote_repo: str = "/home/ma-user/work/code-rl"
    cache_dir: str = "/home/ma-user/work/code-rl/cache"
    remote_port: int = 8091
    local_port: int = 8091
    local_host: str = "127.0.0.1"
    service_host: str = "127.0.0.1"
    service_env_file: str = ""
    service_python: str = "python3"
    service_startup_timeout: float = 90.0
    auto_install_deps: bool = False
    connect_timeout: float = 10.0
    health_timeout: float = 3.0
    bootstrap_timeout: float = 180.0
    runtime_dir: str = ""

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        base = str(env.get("SEARCH_CACHE_BASE_URL") or env.get("SEARCH_GATEWAY_BASE_URL", "")).strip().rstrip("/")
        target = urlsplit(base)
        try:
            local_port = target.port or (443 if target.scheme == "https" else 80)
        except ValueError:
            raise BootstrapError("SEARCH_CACHE_BASE_URL contains an invalid port") from None
        if (target.scheme not in {"http", "https"} or not target.hostname
                or target.username is not None or target.password is not None
                or target.query or target.fragment):
            raise BootstrapError("SEARCH_CACHE_BASE_URL must be an absolute HTTP(S) URL without credentials")
        token = env.get("SEARCH_CACHE_TOKEN") or env.get("GATEWAY_TOKEN") or env.get("SEARCH_GATEWAY_TOKEN", "")
        if not token or any(ord(ch) < 33 or ord(ch) > 126 for ch in token):
            raise BootstrapError("SEARCH_CACHE_TOKEN must contain printable ASCII without whitespace")
        loopback = _loopback(target.hostname)
        auto_tunnel = _flag(env, "SEARCH_CACHE_AUTO_TUNNEL", loopback)
        if auto_tunnel and not loopback:
            raise BootstrapError("Automatic SSH forwarding requires a loopback SEARCH_CACHE_BASE_URL")
        ssh_target = str(env.get("SEARCH_CACHE_SSH_TARGET", "ma-user@7.150.11.99"))
        if (not ssh_target or ssh_target.startswith("-")
                or any(not (ch.isalnum() or ch in "@._:-[]") for ch in ssh_target)):
            raise BootstrapError("SEARCH_CACHE_SSH_TARGET must be an SSH hostname or user@host")
        remote_repo = str(env.get("SEARCH_CACHE_REMOTE_REPO_DIR", "/home/ma-user/work/code-rl"))
        cache_dir = str(env.get("SEARCH_CACHE_DIR", "/home/ma-user/work/code-rl/cache"))
        env_file = str(env.get("SEARCH_SERVICE_ENV_FILE", str(Path(cache_dir) / "search-service.env")))
        if any(not path.startswith("/") or "\x00" in path for path in (remote_repo, cache_dir, env_file)):
            raise BootstrapError("Remote repository, cache directory and service env file must be absolute paths")
        service_host = str(env.get("SEARCH_SERVICE_HOST", "127.0.0.1"))
        if not service_host or any(not (ch.isalnum() or ch in ".:-") for ch in service_host):
            raise BootstrapError("SEARCH_SERVICE_HOST must be a hostname or IP address")
        # OpenSSH Unix sockets have a short path limit; macOS's TMPDIR can be long.
        runtime_root = Path(tempfile.gettempdir())
        if len(os.fsencode(runtime_root)) > 35:
            runtime_root = Path("/tmp")
        runtime_dir = str(env.get("SEARCH_CACHE_RUNTIME_DIR") or runtime_root / f"agentflow-search-cache-{os.getuid()}")
        return cls(
            base_url=base, token=token,
            auto_start=_flag(env, "SEARCH_CACHE_AUTO_START", True), auto_tunnel=auto_tunnel,
            ssh_target=ssh_target, ssh_port=_number(env, "SEARCH_CACHE_SSH_PORT", 31753, port=True),
            remote_repo=remote_repo, cache_dir=cache_dir,
            remote_port=_number(env, "SEARCH_SERVICE_PORT", 8091, port=True), local_port=local_port,
            local_host="127.0.0.1" if target.hostname == "localhost" else target.hostname,
            service_host=service_host, service_env_file=env_file,
            service_python=str(env.get("SEARCH_SERVICE_PYTHON", "python3")),
            service_startup_timeout=_number(env, "SEARCH_SERVICE_STARTUP_TIMEOUT", 90),
            auto_install_deps=_flag(env, "SEARCH_SERVICE_AUTO_INSTALL_DEPS", False),
            connect_timeout=_number(env, "SEARCH_CACHE_SSH_CONNECT_TIMEOUT_SECONDS", 10),
            health_timeout=_number(env, "SEARCH_CACHE_HEALTH_TIMEOUT_SECONDS", 3),
            bootstrap_timeout=_number(env, "SEARCH_CACHE_BOOTSTRAP_TIMEOUT_SECONDS", 180),
            runtime_dir=runtime_dir,
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def probe_health(config, *, timeout):
    """One authenticated health request. Unreachable is false; wrong identity fails."""
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    request = Request(config.base_url + "/healthz", headers={
        "Authorization": f"Bearer {config.token}", "Accept": "application/json",
        "Accept-Encoding": "identity",
    })
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise BootstrapError("Search cache health response exceeds the size limit")
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeError):
                raise BootstrapError("Search cache endpoint returned invalid health JSON") from None
            if not isinstance(result, dict) or result.get("status") != "ok":
                raise BootstrapError("Search cache endpoint is reachable but not healthy")
            return True
    except HTTPError as error:
        error.close()
        if error.code in {401, 403}:
            raise BootstrapError("Search cache authentication failed; check the shared token (service will not be restarted)") from None
        raise BootstrapError(f"Search cache health request returned HTTP {error.code}; check the service URL") from None
    except URLError as error:
        if isinstance(error.reason, ssl.SSLCertVerificationError):
            raise BootstrapError("Search cache TLS certificate verification failed") from None
        return False
    except (TimeoutError, socket.timeout, ConnectionError):
        return False


def run_command(command, *, timeout):
    # SSH never receives service/upstream credentials through argv or SendEnv.
    env = {key: value for key, value in os.environ.items() if key not in {
        "SEARCH_CACHE_TOKEN", "SEARCH_SERVICE_TOKEN", "SEARCH_GATEWAY_TOKEN", "GATEWAY_TOKEN",
        "BRAVE_API_KEY", "YIBU_BRAVE_API_KEY", "OPENAI_API_KEY",
    }}
    # A background OpenSSH master inherits its parent's descriptors. PIPE would
    # leave communicate() waiting for EOF until that persistent master exits.
    capture = "-fN" not in command and "-O" not in command
    try:
        return subprocess.run(command, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                              stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
                              text=True, timeout=timeout,
                              check=False, env=env)
    except FileNotFoundError:
        raise BootstrapError("OpenSSH client is unavailable; install ssh on the training node") from None
    except subprocess.TimeoutExpired:
        raise BootstrapError("SSH search cache bootstrap exceeded its deadline") from None
    except OSError:
        raise BootstrapError("SSH search cache bootstrap could not start") from None


def ssh_options(config):
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "SendEnv=-*", "-o", f"ConnectTimeout={max(1, math.ceil(config.connect_timeout))}",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            "-p", str(config.ssh_port)]


def remote_command(config, startup_timeout):
    values = {
        "SEARCH_CACHE_DIR": config.cache_dir,
        "SEARCH_SERVICE_ENV_FILE": config.service_env_file,
        "SEARCH_SERVICE_HOST": config.service_host,
        "SEARCH_SERVICE_PORT": str(config.remote_port),
        "SEARCH_SERVICE_PYTHON": config.service_python,
        "SEARCH_SERVICE_STARTUP_TIMEOUT": str(startup_timeout),
        "SEARCH_SERVICE_AUTO_INSTALL_DEPS": "1" if config.auto_install_deps else "0",
    }
    helper = str(Path(config.remote_repo) / "train-roma/ensure_search_cache_service.sh")
    return " ".join(["env", *(shlex.quote(f"{key}={value}") for key, value in values.items()),
                     "bash", shlex.quote(helper)])


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BootstrapError("Search cache bootstrap exceeded its deadline")
    return remaining


def _runtime_directory(config):
    path = Path(config.runtime_dir)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if path.is_symlink() or not path.is_dir() or info.st_uid != os.getuid():
        raise BootstrapError("Search cache runtime directory must be owned by the training user")
    path.chmod(0o700)
    return path


@contextlib.contextmanager
def node_lock(config, deadline):
    directory = _runtime_directory(config)
    lock_path = directory / f"port-{config.local_port}.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(min(0.1, _remaining(deadline)))
        yield directory
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _socket_path(config, directory):
    identity = (f"{config.ssh_target}:{config.ssh_port}:{config.local_host}:"
                f"{config.local_port}:{config.service_host}:{config.remote_port}")
    path = directory / ("ssh-" + hashlib.sha256(identity.encode()).hexdigest()[:20])
    if len(os.fsencode(path)) >= 100:
        raise BootstrapError("SSH control socket path is too long; use a shorter SEARCH_CACHE_RUNTIME_DIR")
    return path


def _forward_destination(config):
    # Wildcard bind addresses aren't meaningful remote destinations.
    host = {"0.0.0.0": "127.0.0.1", "::": "::1", "localhost": "127.0.0.1"}.get(
        config.service_host, config.service_host
    )
    return f"[{host}]" if ":" in host else host


def bootstrap(config, *, runner=run_command, probe=probe_health):
    deadline = time.monotonic() + config.bootstrap_timeout

    def ready():
        return probe(config, timeout=min(config.health_timeout, _remaining(deadline)))

    if ready():
        return {"status": "ok", "reused": True, "tunnel": config.auto_tunnel}
    if not config.auto_start and not config.auto_tunnel:
        raise BootstrapError("Search cache is unreachable; automatic service start and SSH tunnel are disabled")

    with node_lock(config, deadline) as directory:
        if ready():
            return {"status": "ok", "reused": True, "tunnel": config.auto_tunnel}
        metadata = {"reused": True}
        if config.auto_start:
            startup_timeout = min(config.service_startup_timeout, max(1, _remaining(deadline) - config.connect_timeout))
            command = ssh_options(config) + ["-o", "ControlMaster=no", "-o", "ControlPath=none",
                                              config.ssh_target, remote_command(config, startup_timeout)]
            result = runner(command, timeout=min(_remaining(deadline), startup_timeout + config.connect_timeout + 5))
            if result.returncode != 0:
                raise BootstrapError(
                    f"Development search service ensure failed (SSH exit {result.returncode}); "
                    "check the SSH key, known_hosts and deployed ensure_search_cache_service.sh"
                )
            try:
                metadata = json.loads(result.stdout)
            except (ValueError, TypeError):
                raise BootstrapError("Development search service helper returned invalid readiness JSON") from None
            if (not isinstance(metadata, dict) or metadata.get("status") != "ok"
                    or type(metadata.get("port")) is not int or metadata["port"] != config.remote_port
                    or type(metadata.get("reused")) is not bool):
                raise BootstrapError("Development search service helper returned an unexpected status or port")

        if ready():
            return {"status": "ok", "reused": metadata["reused"], "tunnel": config.auto_tunnel}
        if config.auto_tunnel:
            control_path = _socket_path(config, directory)
            if control_path.exists():
                check = runner(ssh_options(config) + ["-S", str(control_path), "-O", "check", config.ssh_target],
                               timeout=min(config.connect_timeout + 2, _remaining(deadline)))
                if check.returncode == 0:
                    raise BootstrapError("Existing SSH cache tunnel is running but the service is unreachable; check forwarding and service binding")
                control_path.unlink(missing_ok=True)
            bind_host = f"[{config.local_host}]" if ":" in config.local_host else config.local_host
            forward = f"{bind_host}:{config.local_port}:{_forward_destination(config)}:{config.remote_port}"
            command = ssh_options(config) + ["-M", "-S", str(control_path), "-fN",
                                              "-o", "ControlPersist=yes", "-o", "ExitOnForwardFailure=yes",
                                              "-L", forward, config.ssh_target]
            result = runner(command, timeout=min(config.connect_timeout + 10, _remaining(deadline)))
            if result.returncode != 0:
                raise BootstrapError(f"SSH cache tunnel failed (exit {result.returncode}); check local port, SSH key and known_hosts")

        # A successful -fN has bound the port; allow a short, bounded readiness wait.
        readiness_deadline = min(deadline, time.monotonic() + 15)
        while time.monotonic() < readiness_deadline:
            if ready():
                return {"status": "ok", "reused": metadata["reused"], "tunnel": config.auto_tunnel}
            time.sleep(min(0.5, _remaining(readiness_deadline)))
        raise BootstrapError("Search cache did not become reachable; check HTTP routing and service binding")


def main():
    try:
        result = bootstrap(Config.from_env())
    except (BootstrapError, OSError) as error:
        # OSError may include a path; tokens and subprocess output are never shown.
        message = str(error) if isinstance(error, BootstrapError) else "Cannot access the local SSH runtime directory"
        print(f"Search cache bootstrap failed: {message}", file=sys.stderr)
        return 1
    state = "reused" if result["reused"] else "started"
    print(f"[OK] Shared search cache ready ({state}); training will use the service.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
