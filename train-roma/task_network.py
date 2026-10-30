"""Resolve a training task before loading optional external-search components.

Local task interaction still uses its normal environment/model HTTP services.
This module decides only whether external web-search plumbing is applicable.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import re
import sys


LOCAL_TASKS = frozenset({"alfworld", "webshop", "rule"})
_TASK = re.compile(r"[a-z][a-z0-9_-]*")
_ENV_DEFAULT = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*),(.*?)\}")


class TaskNetworkError(ValueError):
    """The selected task cannot be determined without guessing."""


def normalize_task(value):
    if not isinstance(value, str):
        raise TaskNetworkError("TASK must be a nonempty task name")
    task = value.strip().lower()
    if _TASK.fullmatch(task) is None:
        raise TaskNetworkError("TASK must be a nonempty task name containing letters, digits, '_' or '-'")
    return task


def task_needs_search(task):
    """Unknown tasks retain the existing online startup behavior."""
    return normalize_task(task) not in LOCAL_TASKS


def resolve_task(config_path=None, *, environ=None):
    """Prefer explicit TASK; otherwise read env.TASK from the launch config."""
    environ = os.environ if environ is None else environ
    if "TASK" in environ:
        return normalize_task(environ["TASK"])
    configured_path = config_path or environ.get("AGENTFLOW_CONFIG_FILE")
    path = (Path(configured_path).expanduser() if configured_path else Path(__file__).with_name("config.yaml"))
    try:
        import yaml
    except ImportError as error:
        raise TaskNetworkError("Reading the task from config requires PyYAML; activate the training Python environment or set TASK") from error
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        # YAML diagnostics may quote unrelated credential lines. Report only
        # the selected config path, never configuration values or raw errors.
        raise TaskNetworkError(f"Cannot read task configuration {path}") from None
    if not isinstance(config, Mapping) or not isinstance(config.get("env", {}), Mapping):
        raise TaskNetworkError("Task configuration must contain an env mapping")
    value = config.get("env", {}).get("TASK", "qa")
    if isinstance(value, str):
        match = _ENV_DEFAULT.fullmatch(value.strip())
        if match is not None:
            name, default = match.groups()
            value = environ.get(name, default)
    return normalize_task(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("task", "needs-search"))
    parser.add_argument("--config", help="Training YAML; defaults to config.yaml beside this script")
    args = parser.parse_args(argv)
    try:
        task = resolve_task(args.config)
    except TaskNetworkError as error:
        print(f"Task network routing failed: {error}", file=sys.stderr)
        return 2
    if args.command == "task":
        print(task)
        return 0
    return 0 if task_needs_search(task) else 1


if __name__ == "__main__":
    raise SystemExit(main())
