#!/usr/bin/env python3
"""Sample one ALFWorld task once; edit a probe and rescore saved trajectories.

Run without a subcommand for the local editor, or use sample / score / run.
Heavy environment/model imports happen only in the operation that needs them.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from datetime import datetime, timezone
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import traceback
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import uuid


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET = "data/train/alfworld/train_all_alfworld_3305.parquet"
DEFAULT_CONFIG = {
    "dataset": {"path": DEFAULT_DATASET, "row_index": 0},
    "sampling": {
        "trajectories": 4, "workers": 4, "max_steps": 50, "history_length": 4,
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 512, "seed": 42,
        "base_url": "http://127.0.0.1:19998/v1",
        "model": "Qwen3-30B-A3B-Instruct-2507", "api_key_env": "OPENAI_API_KEY",
        "timeout": 180, "return_token_ids": True,
        "alfworld_root": "", "alfworld_data": "",
    },
    "scorer": {
        "backend": "vllm", "base_url": "http://127.0.0.1:19998/v1",
        "model": "Qwen3-30B-A3B-Instruct-2507",
        "tokenizer": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "api_key_env": "OPENAI_API_KEY", "timeout": 180,
        "max_context_tokens": 16384, "device": "auto",
    },
    "probe": {
        "prompt": (
            "Based on the applicable environment rules, initial state, current "
            "full state and recorded action history above, describe the current "
            "state in relation to the task's success conditions. Desired goal "
            "conditions are not evidence that they already hold."
        ),
        "response": "{goal_response}", "state_view": "full_state",
        "epsilon": 0.1, "k": 3, "gamma": 0.8, "max_response_tokens": 256,
        "terminal_fill_enabled": True, "terminal_fill_scale": 2.0,
        "norm_adv_by_std_in_grpo": False, "outcome_success_threshold": 0.5,
        "outcome_match_value": False, "match_epsilon": 0.05,
        "positive_threshold": 0.01, "negative_threshold": 0.01,
        "gap_epsilon": 0.02, "length_ratio_max": 1.8,
        "k_pairs_per_group": 2, "max_pairs_per_step": 256,
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def compact_env_info(info):
    """Drop repeated heavyweight simulator facts already recorded in states[].facts."""
    if not isinstance(info, dict):
        return info
    return {key: value for key, value in info.items()
            if key not in ("facts", "trace_full_facts", "trace_goal_metadata")}


def atomic_json(path, value):
    """Durable replace with disk verification and a last-known-good trajectory backup.

    A large full-state snapshot can exceed tens of MB. The old implementation
    replaced it without syncing or verifying the temporary file. All writes
    now go through a verified temporary file, with a backup of the last valid
    merged trajectory snapshot. Checkpoints still remain independent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact encoding significantly reduces repeated multi-MB checkpoint I/O.
    payload = (json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    protect = path.name == "trajectories.json"
    backup = path.with_name(path.name + ".lastgood.bak")
    backup_temp = backup.with_name(backup.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("xb") as stream:
            written = stream.write(payload)
            if written != len(payload):
                raise OSError(f"Short JSON write: {written}/{len(payload)} bytes to {temp}")
            stream.flush()
            os.fsync(stream.fileno())
        if temp.stat().st_size != len(payload):
            raise OSError(
                f"Snapshot truncated during write: expected {len(payload)} bytes, "
                f"got {temp.stat().st_size}, file={temp}"
            )
        # The expensive JSON readback is confined to merged snapshots, which
        # are written only as rollouts finish; per-turn worker checkpoints use
        # fsync + length verification to avoid an extra large read every turn.
        if protect:
            try:
                with temp.open("r", encoding="utf-8") as reader:
                    json.load(reader)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise OSError(f"Temporary snapshot failed JSON validation: {temp}: {exc}") from exc
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8") as reader:
                        json.load(reader)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Preserve the broken original, not overwrite a good backup.
                    raise RuntimeError(
                        f"Existing snapshot is invalid: {path}. Use the recovery "
                        "button before sampling again; original file was not overwritten."
                    )
                # Copy/link the last complete version before promoting the new
                # snapshot. The backup name is stable to avoid unbounded growth.
                try:
                    os.link(path, backup_temp)
                except OSError:
                    shutil.copyfile(path, backup_temp)
                    with backup_temp.open("rb") as stream:
                        os.fsync(stream.fileno())
                os.replace(backup_temp, backup)
        os.replace(temp, path)
        # Durability of the rename itself requires syncing the directory.
        try:
            dirfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except OSError:
            # Some mounted storage cannot fsync directories. File-level fsync
            # and the last-good backup still provide protection.
            pass
    except Exception as exc:
        if isinstance(exc, OSError):
            raise OSError(
                f"Failed to save {path} ({len(payload):,} bytes): {exc}. "
                "Check filesystem free space, inode/quota limits and mount type."
            ) from exc
        raise
    finally:
        temp.unlink(missing_ok=True)
        backup_temp.unlink(missing_ok=True)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def scalar(value):
    value = jsonable(value)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def flag(value):
    value = scalar(value)
    return bool(value) if isinstance(value, (bool, int, float)) and value in (0, 1) else None


def deep_update(target, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def resolve_dataset(raw_path):
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path,
                                                    REPO_ROOT.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset not found: {raw_path}. Set --dataset to its absolute path.")


def select_row(raw_path, index):
    """Read only as far as the requested row; task index comes from question."""
    if type(index) is not int or index < 0:
        raise ValueError("row_index must be a nonnegative integer")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Reading the dataset requires pyarrow (pip install pyarrow).") from error
    path = resolve_dataset(raw_path)
    parquet = pq.ParquetFile(path)
    if index >= parquet.metadata.num_rows:
        raise IndexError(f"Row {index} exceeds dataset size {parquet.metadata.num_rows}")
    offset = 0
    for batch in parquet.iter_batches(batch_size=256):
        if offset + batch.num_rows > index:
            row = batch.slice(index - offset, 1).to_pylist()[0]
            break
        offset += batch.num_rows
    question = row.get("question")
    match = re.fullmatch(r"(train|val_seen|val_unseen)_(.+)_alfworld_(\d+)", str(question))
    if not match:
        raise ValueError(f"Unsupported ALFWorld task identifier: {question!r}")
    return {
        "path": str(path), "row_index": index, "num_rows": parquet.metadata.num_rows,
        "question": question, "source_row": jsonable(row),
        "reset": {"data_type": match[1], "task_type": match[2], "question_id": int(match[3])},
    }


def make_environment(sampling):
    source = sampling.get("alfworld_root")
    if source:
        sys.path.insert(0, str(Path(source).expanduser().resolve()))
    # Import the inner, lightweight agentflow package, avoiding the RL trainer.
    sys.path.insert(0, str(REPO_ROOT / "agentflow"))
    try:
        module = importlib.import_module("agentflow.envs.alfworld.alfworld_env")
    except ImportError as error:
        raise RuntimeError(
            f"Cannot load ALFWorld: {error}. Run in the existing ALFWorld environment; "
            "use --alfworld-root for its source directory and --alfworld-data for "
            "the directory containing json_2.1.1/ and logic/. Scoring saved trajectories "
            "does not import or require ALFWorld."
        ) from error
    if sampling.get("alfworld_data"):
        module.data_path = str(Path(sampling["alfworld_data"]).expanduser().resolve())
    return module.AlfWorldEnv(history_length=sampling["history_length"],
                              max_step=sampling["max_steps"], trace_enabled=True)


def request_action(prompt, sampling, trajectory_index, turn_index):
    """One genuine policy completion; scoring never calls this function."""
    payload = {
        "model": sampling["model"], "messages": [{"role": "user", "content": prompt}],
        "temperature": sampling["temperature"], "top_p": sampling["top_p"],
        "max_tokens": sampling["max_tokens"], "stream": False,
        "seed": sampling["seed"] + 100000 * trajectory_index + turn_index,
    }
    if sampling.get("return_token_ids"):
        payload["return_token_ids"] = True
    key = os.environ.get(sampling.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
    request = Request(sampling["base_url"].rstrip("/") + "/chat/completions",
                      data=json.dumps(payload).encode(), method="POST",
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urlopen(request, timeout=sampling["timeout"]) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"Policy HTTP {error.code}: {detail}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"Policy request failed: {error}") from error
    choice = result["choices"][0]
    text = choice.get("message", {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Policy returned no action text; trajectory checkpoint was preserved.")
    return {"response": text, "finish_reason": choice.get("finish_reason"),
            "prompt_token_ids": result.get("prompt_token_ids"),
            "response_token_ids": choice.get("token_ids"), "usage": result.get("usage")}


def _goal_module():
    name = "_alfworld_probe_goal_conversion"
    if name not in sys.modules:
        path = REPO_ROOT / "agentflow/agentflow/trace/alfworld.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def goal_metadata(spec):
    keys = ("gamefile", "goal_source", "goal_task_type", "task_type", "template_version",
            "source_sha256", "goal_spec_hash", "goal_ast", "goal_validation_status")
    return {key: jsonable(spec[key]) for key in keys if key in spec}


def _save_goal_fields(trajectory, spec):
    if not isinstance(spec, dict) or not isinstance(spec.get("goal_response"), str):
        raise ValueError("Environment goal specification has no deterministic goal_response")
    trajectory.update({"goal_response": spec["goal_response"], "goal_spec": jsonable(spec),
                       "goal_metadata": goal_metadata(spec), "goal_source": spec.get("goal_source"),
                       "goal_task_type": spec.get("goal_task_type", spec.get("task_type")),
                       "goal_spec_hash": spec.get("goal_spec_hash"),
                       "state_view": "full_state"})


def _full_facts(env, info):
    facts = info.get("trace_full_facts", getattr(env, "trace_facts", None))
    if facts is None:
        return None
    return _goal_module().normalize_facts(facts)


def state_snapshot(env, actor_prompt, info, index, sampling, *, prior_turns=(), done=False):
    """Snapshot before the next action; error branches cannot reuse stale context."""
    raw_info = info or {}
    facts = _full_facts(env, raw_info)
    info = jsonable(raw_info)
    observation = scalar(info.get("anchor"))
    if not isinstance(observation, str) or observation.startswith("["):
        observation = scalar(info.get("observation_text"))
    if not isinstance(observation, str):
        observation = getattr(env, "prev_text_obs", "") or actor_prompt
    actions = info.get("admissible_commands", getattr(env, "prev_admissible_command", []))
    if isinstance(actions, list) and len(actions) == 1 and isinstance(actions[0], list):
        actions = actions[0]
    is_error = any(flag(info.get(k)) is True for k in ("env_error", "format_error", "invalid_action"))
    # Locally rejected actions leave verified simulator facts unchanged. Their
    # full-state view is still usable; external crashes cannot reuse old facts.
    context = (getattr(env, "trace_state_context", None)
               if not is_error or (facts is not None and flag(info.get("env_error")) is not True)
               else None)
    if not isinstance(context, str) or not context.strip():
        recent = list(prior_turns)[-sampling["history_length"]:] if sampling["history_length"] else []
        history = [{"action": t.get("action"), "observation": t.get("observation_before")}
                   for t in recent]
        context = (f"Task: {getattr(env, 'task', '')}\n\n"
                   f"Recent observations and actions: {json.dumps(history, ensure_ascii=False)}\n\n"
                   f"Current observation: {observation}\n\n"
                   f"Available actions: {json.dumps(actions, ensure_ascii=False)}")
    return {"state_index": index, "state_context": context, "observation": observation,
            "available_actions": actions, "remaining_steps": max(0, sampling["max_steps"] - index),
            "is_post_state": index > 0, "is_terminal_state": bool(done),
            "env_info": compact_env_info(info), "facts": facts,
            "state_view": "full_state" if facts is not None else "legacy_observed",
            "goal_check": info.get("trace_goal_check"),
            "goal_check_valid": info.get("trace_goal_check_valid"),
            "scoring_valid": flag(info.get("env_error")) is not True
            and info.get("trace_scoring_valid", True) is not False
            and (info.get("trace_goal_check_valid") is True if facts is not None
                 else info.get("trace_goal_check_valid", True) is not False)}


def summarize_outcome(info, reward, done):
    info = info or {}
    external = flag(info.get("env_error")) is True
    truncated = flag(info.get("truncated")) is True or flag(info.get("truncated_by_max_step")) is True
    terminated = flag(info.get("terminated"))
    if terminated is None:
        terminated = bool(done and not truncated and not external)
    success = flag(info.get("won"))
    if success is None and done and not external:
        # The wrapper's normal reward is binary; do not infer from positive format bonuses.
        raw = scalar(reward)
        success = raw == 1.0 if isinstance(raw, (int, float)) and raw in (0, 1) else None
    reason = ("environment_error" if external else "format_error" if flag(info.get("format_error"))
              else "invalid_action" if flag(info.get("invalid_action")) else "max_steps" if truncated
              else "success" if success else "task_failure" if done else "incomplete")
    goal_check_valid = info.get("trace_goal_check_valid", True) is not False
    return {"reward": scalar(reward), "success": success,
            "valid": bool(done and not external and success is not None and goal_check_valid),
            "terminated": terminated, "truncated": truncated, "reason": reason}


def capture_trajectories(snapshot, sampling, save, *, env_factory=make_environment,
                         policy=request_action, progress=None, trajectory_offset=0):
    """Resume incomplete capture by replaying saved actions, never resampling them."""
    progress = progress or (lambda message: None)
    if snapshot.get("complete"):
        progress("Saved trajectories already complete; reusing without policy requests.")
        return snapshot
    if snapshot["sampling"] != sampling:
        raise ValueError("Sampling settings differ from the saved session. Use the saved settings or a new work-dir.")
    count = sampling["trajectories"]
    # Keep the ALFWorld wrapper and its game-file cache across rollouts.
    env = None
    for trajectory_index in range(count):
        trajectories = snapshot["trajectories"]
        trajectory = trajectories[trajectory_index] if trajectory_index < len(trajectories) else None
        if trajectory and trajectory.get("status") in ("completed", "external_error"):
            continue
        if env is None:
            env = env_factory(sampling)
        try:
            actor_prompt, info = env.reset(**snapshot["dataset"]["reset"])
            initial = state_snapshot(env, actor_prompt, info, 0, sampling)
            if trajectory is None:
                if trajectories:
                    first = trajectories[0]
                    same_game = first.get("initial_gamefile") == jsonable(info.get("extra.gamefile"))
                    same_state = first["states"][0]["state_context"] == initial["state_context"]
                    if not (same_game and same_state):
                        raise RuntimeError("The selected task reset changed across rollouts; refusing to mix initial states.")
                trajectory = {"trajectory_id": f"traj_{trajectory_index + trajectory_offset + 1:02d}",
                              "goal": getattr(env, "task", ""),
                              "initial_gamefile": jsonable(info.get("extra.gamefile")),
                              "status": "running", "outcome": None,
                              "states": [initial], "turns": []}
                spec = getattr(env, "trace_goal_spec", None)
                if spec is not None:
                    _save_goal_fields(trajectory, spec)
                trajectories.append(trajectory)
                save(snapshot)
            elif trajectory["states"][0]["state_context"] != initial["state_context"]:
                raise RuntimeError("Reset changed the saved initial state; refusing to mix trajectories.")
            done = False
            reward = 0.0
            # An interrupted request may be retried, but every already saved action is replayed.
            for turn in trajectory["turns"]:
                if actor_prompt != turn["action_prompt"]:
                    raise RuntimeError("Environment replay diverged before a saved action.")
                actor_prompt, reward, done, info = env.step(turn["action_response"])
                observed = state_snapshot(env, actor_prompt, info, turn["turn_index"] + 1,
                                          sampling, prior_turns=trajectory["turns"][:turn["turn_index"] + 1],
                                          done=done)
                expected = trajectory["states"][turn["turn_index"] + 1]
                if observed["state_context"] != expected["state_context"]:
                    raise RuntimeError("Environment replay diverged after a saved action.")
            while not done and len(trajectory["turns"]) < sampling["max_steps"]:
                index = len(trajectory["turns"])
                progress(f"Sampling trajectory {trajectory_index + trajectory_offset + 1}, turn {index + 1}; saved actions are retained.")
                generated = policy(actor_prompt, sampling, trajectory_index + trajectory_offset, index)
                response = generated["response"]
                match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
                turn = {"turn_index": index, "action_prompt": actor_prompt,
                        "action_response": response, "action": match[1].strip() if match else None,
                        "observation_before": trajectory["states"][-1]["observation"],
                        "finish_reason": generated.get("finish_reason"),
                        "prompt_token_ids": generated.get("prompt_token_ids"),
                        "response_token_ids": generated.get("response_token_ids"),
                        "usage": generated.get("usage")}
                try:
                    next_prompt, reward, done, info = env.step(response)
                except Exception as error:
                    next_prompt = "Environment execution failed."
                    reward, done = None, True
                    info = {"env_error": True, "env_error_type": type(error).__name__,
                            "observation_text": str(error), "terminated": False, "truncated": False}
                info = jsonable(info or {})
                if index + 1 >= sampling["max_steps"] and not done:
                    done = True
                    info.update({"truncated": True, "truncated_by_max_step": True})
                turn.update({"reward": scalar(reward), "done": bool(done),
                             "action": info.get("trace_env_action", turn["action"]),
                             "external_error": flag(info.get("env_error")) is True,
                             "format_error": flag(info.get("format_error")) is True,
                             "invalid_action": flag(info.get("invalid_action")) is True,
                             "env_info": compact_env_info(info)})
                trajectory["turns"].append(turn)
                trajectory["states"].append(state_snapshot(
                    env, next_prompt, info, index + 1, sampling,
                    prior_turns=trajectory["turns"], done=done))
                actor_prompt = next_prompt
                if done:
                    trajectory["outcome"] = summarize_outcome(info, reward, done)
                    trajectory["status"] = "external_error" if turn["external_error"] else "completed"
                save(snapshot)
            if trajectory["outcome"] is None:
                trajectory["outcome"] = summarize_outcome(info, reward, True)
                trajectory["status"] = "completed"
            save(snapshot)
        finally:
            env.close()
    snapshot["complete"] = True
    snapshot["completed_at"] = utc_now()
    save(snapshot)
    return snapshot



# The catalogue is immutable: it can be shared safely between processes and
# saved samples. The actual game environment is mutable and MUST remain private
# to each rollout worker.
_CATALOG_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
_PERSISTENT_POOL = None
_PERSISTENT_POOL_WORKERS = None
_POOL_LOCK = threading.Lock()
_PROCESS_ENV = None
_PROCESS_ENV_KEY = None


def _catalogue_for_sample(dataset, sampling, directory, progress):
    """Scan the ALL-task game list once per data split, cache it for future samples.

    Filtering the ordered ALL list reproduces the order of each task-specific
    AlfredTWEnv.collect_game_files() configuration without rescanning.
    """
    reset = dataset["reset"]
    split = reset["data_type"]
    if split not in ("train", "val_seen", "val_unseen"):
        raise ValueError(f"Unsupported ALFWorld split: {split}")
    root = (Path(sampling["alfworld_data"]).expanduser().resolve()
            if sampling.get("alfworld_data")
            else (REPO_ROOT / "data" / "alfworld-master" / "json_data").resolve())
    workspace = Path(directory)
    if workspace.parent.name == "samples":
        workspace = workspace.parent.parent
    key = {"root": str(root), "split": split}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    catalogue_file = workspace / "game_catalog" / f"{split}-{digest}.json"
    stored = read_json(catalogue_file)
    if stored is None or stored.get("key") != key:
        progress(f"Indexing ALFWorld games once for {split}; future samples will reuse the catalogue.")
        wrapper = make_environment(sampling)
        base = wrapper._init_env(data_type=split, task_type="all")
        records = []
        for filename in base.game_files:
            gamefile = str(Path(filename).resolve())
            with Path(gamefile).with_name("traj_data.json").open(encoding="utf-8") as f:
                details = json.load(f)
            records.append({"gamefile": gamefile, "task_type": details["task_type"]})
        stored = {"key": key, "records": records, "created_at": utc_now()}
        atomic_json(catalogue_file, stored)
        progress(f"Cached {len(records)} ALFWorld games at {catalogue_file}")
    else:
        progress(f"Reusing the cached ALFWorld game catalogue ({len(stored['records'])} games).")
    config_path = (REPO_ROOT / "agentflow" / "agentflow" / "envs" /
                   "alfworld" / "config" / f"config_tw_{reset['task_type']}.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"Unsupported ALFWorld task configuration: {config_path}")
    import yaml
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    allowed_types = {_CATALOG_TYPES[i] for i in config["env"]["task_types"]}
    candidates = [r["gamefile"] for r in stored["records"] if r["task_type"] in allowed_types]
    data_config = config.get("dataset") or {}
    limit = data_config.get("num_train_games" if split == "train" else "num_eval_games", -1)
    if isinstance(limit, int) and limit > 0:
        candidates = candidates[:limit]
    selected_id = reset["question_id"]
    if selected_id < 0 or selected_id >= len(candidates):
        raise IndexError(
            f"Task question_id {selected_id} exceeds catalogue size {len(candidates)} "
            f"for {split}/{reset['task_type']}"
        )
    selected = candidates[selected_id]
    if not Path(selected).is_file():
        raise FileNotFoundError(f"Cached ALFWorld game has moved: {selected}")
    progress(f"Selected game {selected_id}/{len(candidates)}: {Path(selected).parent.name}")
    return selected


def _persistent_pool(workers):
    global _PERSISTENT_POOL, _PERSISTENT_POOL_WORKERS
    with _POOL_LOCK:
        if _PERSISTENT_POOL is None or _PERSISTENT_POOL_WORKERS != workers:
            if _PERSISTENT_POOL is not None:
                _PERSISTENT_POOL.shutdown(wait=True)
            _PERSISTENT_POOL = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
            _PERSISTENT_POOL_WORKERS = workers
    return _PERSISTENT_POOL


def _shutdown_persistent_pool():
    global _PERSISTENT_POOL, _PERSISTENT_POOL_WORKERS
    with _POOL_LOCK:
        if _PERSISTENT_POOL is not None:
            _PERSISTENT_POOL.shutdown(wait=True)
        _PERSISTENT_POOL = None
        _PERSISTENT_POOL_WORKERS = None


def _capture_parallel_worker(trajectory_index, dataset, sampling, snapshot_id, directory, original,
                             selected_gamefile):
    """Reuse this process's environment, without rescanning the game directory."""
    global _PROCESS_ENV, _PROCESS_ENV_KEY
    checkpoint = Path(directory) / "partial_trajectories" / f"traj_{trajectory_index + 1:02d}.json"
    worker_sampling = {**sampling, "trajectories": 1}
    saved = read_json(checkpoint)
    if saved is not None:
        if saved.get("snapshot_id") != snapshot_id or saved.get("sampling") != worker_sampling:
            raise RuntimeError(
                f"Checkpoint belongs to another sampling session: {checkpoint}. "
                "Use a fresh --work-dir for another task or sampling configuration."
            )
        candidate = (saved.get("trajectories") or [None])[0]
        if candidate and (original is None or len(candidate.get("turns", [])) >= len(original.get("turns", []))):
            original = candidate

    # Worker sees exactly one curated game, so its local question_id is zero.
    # Keep the original parquet row and task identifier in the parent snapshot.
    worker_dataset = copy.deepcopy(dataset)
    worker_dataset["reset"]["question_id"] = 0
    worker_dataset["reset"]["gamefile"] = selected_gamefile
    worker_snapshot = {
        "schema_version": 1, "snapshot_id": snapshot_id,
        "dataset": worker_dataset, "sampling": worker_sampling,
        "complete": False, "trajectories": [original] if original else [],
    }

    env_key = (
        sampling.get("alfworld_root"), sampling.get("alfworld_data"),
        sampling["history_length"], sampling["max_steps"],
    )
    if _PROCESS_ENV is None or _PROCESS_ENV_KEY != env_key:
        if _PROCESS_ENV is not None:
            _PROCESS_ENV.close()
        _PROCESS_ENV = make_environment(sampling)
        _PROCESS_ENV_KEY = env_key
    env = _PROCESS_ENV

    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    original_collect = AlfredTWEnv.collect_game_files

    def collect_selected(self, verbose=False):
        # Bypass the expensive recursive game-file scan in every worker.
        self.game_files = [selected_gamefile]
        self.num_games = 1

    # The worker executes one submitted trajectory at a time. This temporary
    # override cannot affect the parent or any other spawned process.
    AlfredTWEnv.collect_game_files = collect_selected
    try:
        env_key = (worker_dataset["reset"]["data_type"], worker_dataset["reset"]["task_type"])
        if env_key in env._env_instances:
            env._env_gamefiles[env_key] = [selected_gamefile]
        capture_trajectories(
            worker_snapshot, worker_sampling, lambda value: atomic_json(checkpoint, value),
            env_factory=lambda _: env,
            progress=lambda message: print(f"[worker {trajectory_index + 1}] {message}", flush=True),
            trajectory_offset=trajectory_index,
        )
    finally:
        AlfredTWEnv.collect_game_files = original_collect
    result = worker_snapshot["trajectories"][0]
    if result.get("trajectory_id") != f"traj_{trajectory_index + 1:02d}":
        raise RuntimeError("Parallel worker returned a mismatched trajectory ID.")
    if scalar(result.get("initial_gamefile")) != selected_gamefile:
        raise RuntimeError("Worker loaded a different game than the shared catalogue selection.")
    return result


def capture_parallel_trajectories(snapshot, sampling, save, directory, progress=None):
    """Scan one immutable catalogue and reuse persistent isolated game workers."""
    progress = progress or (lambda message: None)
    if snapshot.get("complete"):
        return snapshot
    if snapshot["sampling"] != sampling:
        raise ValueError("Sampling settings differ from saved session; use a new --work-dir.")
    count = sampling["trajectories"]
    workers = min(sampling["workers"], count)
    indexed = {t["trajectory_id"]: t for t in snapshot["trajectories"]}
    expected_ids = {f"traj_{i + 1:02d}" for i in range(count)}
    if set(indexed) - expected_ids:
        raise RuntimeError("Saved snapshot contains unexpected trajectory IDs.")
    selected_gamefile = _catalogue_for_sample(snapshot["dataset"], sampling, directory, progress)
    save(snapshot)
    progress(
        f"Using {workers} reusable independent ALFWorld workers for {count} trajectories; "
        "no redundant directory scans."
    )
    executor = _persistent_pool(workers)
    futures = {
        executor.submit(
            _capture_parallel_worker, i, snapshot["dataset"], sampling,
            snapshot["snapshot_id"], str(directory), indexed.get(f"traj_{i + 1:02d}"),
            selected_gamefile,
        ): i for i in range(count)
        if indexed.get(f"traj_{i + 1:02d}", {}).get("status") not in ("completed", "external_error")
    }
    for future in as_completed(futures):
        i = futures[future]
        trajectory = future.result()
        indexed[trajectory["trajectory_id"]] = trajectory
        snapshot["trajectories"] = [
            indexed[f"traj_{j + 1:02d}"]
            for j in range(count) if f"traj_{j + 1:02d}" in indexed
        ]
        first = snapshot["trajectories"][0]
        for other in snapshot["trajectories"][1:]:
            if (other["initial_gamefile"] != first["initial_gamefile"]
                    or other["states"][0]["state_context"] != first["states"][0]["state_context"]):
                raise RuntimeError("Parallel rollouts reset to different initial games/states.")
        save(snapshot)
        progress(f"Finished trajectory {i + 1}/{count}; {len(indexed)}/{count} collected.")
    if len(indexed) != count or any(
        indexed[f"traj_{i + 1:02d}"].get("status") not in ("completed", "external_error")
        for i in range(count)
    ):
        raise RuntimeError("Not all parallel trajectories finished; resume with the same work directory.")
    snapshot["trajectories"] = [indexed[f"traj_{i + 1:02d}"] for i in range(count)]
    snapshot["complete"] = True
    snapshot["completed_at"] = utc_now()
    save(snapshot)
    return snapshot


def enrich_snapshot(snapshot, *, replay=False, env_factory=make_environment, progress=None):
    """Add official goal metadata/full states in memory; never generate actions."""
    if not snapshot.get("complete"):
        raise ValueError("Finish the initial sampling before enriching its cache")
    progress = progress or (lambda message: None)
    converted = copy.deepcopy(snapshot)
    module = _goal_module()
    specs = {}
    for trajectory in converted.get("trajectories", []):
        gamefile = module.normalize_gamefile(trajectory.get("initial_gamefile"))
        key = str(gamefile)
        if key not in specs:
            specs[key] = module.load_goal_spec(gamefile)
        spec = specs[key]
        _save_goal_fields(trajectory, spec)
        if replay:
            progress(f"Enriching {trajectory['trajectory_id']} by replay only; no policy requests")
            env = env_factory(converted["sampling"])
            try:
                reset = dict(converted["dataset"]["reset"])
                actor_prompt, info = env.reset(**reset, gamefile=str(gamefile))
                actual = module.normalize_gamefile(info.get("extra.gamefile"))
                if actual != gamefile:
                    raise ValueError("Replay loaded a different game file")
                new_states = [state_snapshot(env, actor_prompt, info, 0, converted["sampling"])]
                if new_states[0]["observation"] != trajectory["states"][0]["observation"]:
                    raise ValueError("Replay initial observation differs from the saved trajectory")
                for index, turn in enumerate(trajectory["turns"]):
                    if turn.get("external_error"):
                        failed = copy.deepcopy(trajectory["states"][index + 1])
                        failed.update({"facts": None, "state_view": "full_state", "scoring_valid": False})
                        new_states.append(failed)
                        if index != len(trajectory["turns"]) - 1:
                            raise ValueError("Cannot enrich states beyond an unresolved external failure")
                        break
                    next_prompt, reward, done, info = env.step(turn["action_response"])
                    after = state_snapshot(env, next_prompt, info, index + 1, converted["sampling"],
                                           prior_turns=trajectory["turns"][:index + 1], done=done)
                    if after["observation"] != trajectory["states"][index + 1]["observation"]:
                        raise ValueError(f"Replay observation diverged after saved turn {index}")
                    if bool(done) != bool(turn.get("done")):
                        raise ValueError(f"Replay termination diverged after saved turn {index}")
                    new_states.append(after)
                if len(new_states) != len(trajectory["states"]):
                    raise ValueError("Replay did not recover every saved state")
                trajectory["states"] = new_states
            finally:
                env.close()
        for index, state in enumerate(trajectory["states"]):
            facts = state.get("facts")
            if facts is None:
                facts = (state.get("env_info") or {}).get("trace_full_facts")
            if facts is None and state.get("scoring_valid") is False:
                state["state_view"] = "full_state"
                continue
            if facts is None:
                raise ValueError(
                    "Saved observations do not contain full environment facts. "
                    "Run enrich --replay to replay the saved actions without sampling new ones.")
            facts = module.normalize_facts(facts)
            history = [{"action": turn.get("action"), "response": turn.get("action_response")}
                       for turn in trajectory["turns"][:index]]
            check = module.evaluate_goal(spec, facts)
            won = flag((state.get("env_info") or {}).get("won"))
            check_valid = won is not None and check is won
            state.update({"facts": facts, "state_view": "full_state",
                          "state_context": module.render_full_state(spec, facts,
                              observation=state.get("observation"), action_history=history),
                          "goal_check": check, "goal_check_valid": check_valid,
                          "scoring_valid": check_valid and flag((state.get("env_info") or {}).get("env_error")) is not True})
        if trajectory.get("outcome", {}).get("valid") and trajectory["states"]:
            check = trajectory["states"][-1].get("goal_check")
            if isinstance(check, bool) and check != trajectory["outcome"].get("success"):
                raise ValueError("Replayed goal predicates disagree with the saved environment outcome")
    hashes = {trajectory.get("goal_spec_hash") for trajectory in converted.get("trajectories", [])}
    if len(hashes) != 1:
        raise ValueError("The four saved rollouts do not share the same actual goal specification")
    return converted


def _enrichment_payload(original, converted, source_hash):
    trajectories = []
    fields = ("goal_response", "goal_spec", "goal_metadata", "goal_source", "goal_task_type",
              "goal_spec_hash", "state_view")
    state_fields = ("state_index", "state_context", "facts", "state_view", "scoring_valid",
                    "goal_check", "goal_check_valid")
    for trajectory in converted["trajectories"]:
        trajectories.append({"trajectory_id": trajectory["trajectory_id"],
                             **{key: trajectory.get(key) for key in fields},
                             "states": [{key: state[key] for key in state_fields if key in state}
                                        for state in trajectory["states"]]})
    return {"schema_version": 1, "source_snapshot_sha256": source_hash,
            "snapshot_id": original["snapshot_id"], "created_at": utc_now(),
            "trajectories": trajectories}


def _merge_enrichment(snapshot, metadata, source_hash):
    if metadata.get("source_snapshot_sha256") != source_hash:
        raise ValueError("goal_metadata.json belongs to a different trajectory cache")
    converted = copy.deepcopy(snapshot)
    records = {item["trajectory_id"]: item for item in metadata["trajectories"]}
    for trajectory in converted["trajectories"]:
        item = records.get(trajectory["trajectory_id"])
        if item is None or len(item["states"]) != len(trajectory["states"]):
            raise ValueError("Goal sidecar does not match saved trajectory/state IDs")
        trajectory.update({key: value for key, value in item.items() if key not in ("trajectory_id", "states")})
        for state, update in zip(trajectory["states"], item["states"]):
            if state["state_index"] != update["state_index"]:
                raise ValueError("Goal sidecar state indices differ from the raw cache")
            state.update(update)
    converted["enrichment_sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return converted


def export_csv(result):
    buffer = io.StringIO()
    fields = ["trajectory_id", "turn_index", "mean_logp", "value", "next_value", "delta",
              "credit", "local_credit", "terminal_fill", "label_source", "outcome_advantage",
              "credit_valid", "credit_error", "action", "action_prompt",
              "action_response", "scoring_prompt", "response"]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for trajectory in result.get("trajectories", []):
        for turn in trajectory["turns"]:
            writer.writerow(dict(turn, trajectory_id=trajectory["trajectory_id"]))
    return buffer.getvalue()


def export_report(result):
    esc = lambda value: html.escape(str(value if value is not None else "invalid"))
    blocks = ["<!doctype html><meta charset='utf-8'><title>ALFWorld TRACE probe</title>",
              "<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;padding:16px}"
              "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:12px}"
              "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px}"
              "details{margin:14px 0}summary{cursor:pointer}</style>",
              "<h1>ALFWorld TRACE probe</h1>", f"<p>Run: {esc(result.get('id'))}</p>",
              f"<pre>{esc(json.dumps(result.get('probe'), ensure_ascii=False, indent=2))}</pre>"]
    for trajectory in result.get("trajectories", []):
        blocks.append(f"<h2>{esc(trajectory['trajectory_id'])}</h2><pre>{esc(trajectory.get('outcome'))}</pre>")
        blocks.append(f"<pre>{esc(trajectory.get('goal_response'))}</pre><pre>{esc(trajectory.get('goal_metadata'))}</pre>")
        blocks.append("<table><tr><th>Turn</th><th>Action</th><th>Route</th><th>Mean LogP</th><th>V</th><th>Delta</th><th>Local credit</th><th>Terminal fill</th><th>Effective credit</th></tr>")
        for turn in trajectory["turns"]:
            blocks.append("<tr>" + "".join(f"<td>{esc(turn.get(key))}</td>" for key in
                           ("turn_index", "action", "label_source", "mean_logp", "value", "delta", "local_credit", "terminal_fill", "credit")) + "</tr>")
        blocks.append("</table>")
        for turn in trajectory["turns"]:
            blocks.append(f"<details><summary>Turn {esc(turn['turn_index'])}: full prompts and responses</summary>")
            for key in ("action_prompt", "action_response", "scoring_prompt", "response", "credit_error"):
                blocks.append(f"<h3>{esc(key)}</h3><pre>{esc(turn.get(key))}</pre>")
            blocks.append("</details>")
        for state in trajectory.get("states", [])[-1:]:
            blocks.append("<details><summary>Final post-action scoring state</summary>")
            for key in ("scoring_prompt", "response", "mean_logp", "value", "error"):
                blocks.append(f"<h3>{esc(key)}</h3><pre>{esc(state.get(key))}</pre>")
            blocks.append("</details>")
    return "\n".join(blocks)


class Experiment:
    def __init__(self, directory, overrides=None, config_path=None):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path).resolve() if config_path else self.directory / "config.json"
        self.config = deep_update(copy.deepcopy(DEFAULT_CONFIG), read_json(self.config_path, {}))
        deep_update(self.config, overrides or {})
        self.validate_config(self.config)
        atomic_json(self.config_path, self.config)
        self.lock = threading.RLock()
        self.job = {"status": "idle", "phase": "", "message": "", "error": None}

    @staticmethod
    def validate_config(config):
        probe, sampling = config["probe"], config["sampling"]
        for key in ("prompt", "response"):
            if not isinstance(probe[key], str) or not probe[key].strip():
                raise ValueError(f"probe.{key} must be nonempty text")
        if re.search(r"\{(?:remaining_steps|turn_index)\}", probe["response"]):
            raise ValueError("Response must remain fixed across turns; only {goal} and {goal_response} are allowed.")
        if probe.get("state_view", "saved") not in ("full_state", "saved"):
            raise ValueError("state_view must be full_state or saved")
        if not isinstance(probe.get("terminal_fill_enabled", True), bool):
            raise ValueError("terminal_fill_enabled must be boolean")
        if not math.isfinite(float(probe.get("terminal_fill_scale", 2.0))) or float(probe.get("terminal_fill_scale", 2.0)) < 0:
            raise ValueError("terminal_fill_scale must be nonnegative and finite")
        if type(probe["k"]) is not int or probe["k"] < 1:
            raise ValueError("K must be a positive integer")
        if not math.isfinite(float(probe["epsilon"])) or float(probe["epsilon"]) <= 0:
            raise ValueError("epsilon must be positive and finite")
        if not math.isfinite(float(probe["gamma"])) or not 0 <= float(probe["gamma"]) <= 1:
            raise ValueError("gamma must be between 0 and 1")
        for key in ("trajectories", "workers", "max_steps", "max_tokens"):
            if type(sampling[key]) is not int or sampling[key] < 1:
                raise ValueError(f"sampling.{key} must be a positive integer")
        if type(sampling["history_length"]) is not int or sampling["history_length"] < 0:
            raise ValueError("history_length must be a nonnegative integer")

    @contextlib.contextmanager
    def session_lock(self):
        import fcntl
        with (self.directory / ".session.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process is using this work-dir.") from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def progress(self, message):
        with self.lock:
            self.job["message"] = message
        print(message, flush=True)

    def save_probe(self, probe):
        with self.lock:
            new_config = copy.deepcopy(self.config)
            if set(probe) - set(new_config["probe"]):
                raise ValueError("Unknown probe configuration key")
            new_config["probe"].update(probe)
            self.validate_config(new_config)
            atomic_json(self.config_path, new_config)
            self.config = new_config

    def sample(self, *, env_factory=make_environment, policy=request_action):
        with self.session_lock():
            snapshot = read_json(self.directory / "trajectories.json")
            config = copy.deepcopy(self.config)
            if snapshot is not None:
                saved_dataset = snapshot["dataset"]
                requested_path = str(resolve_dataset(config["dataset"]["path"]))
                if (saved_dataset["path"] != requested_path or
                        saved_dataset["row_index"] != config["dataset"]["row_index"]):
                    raise RuntimeError(
                        "This work directory already holds a different ALFWorld task. "
                        "Use a new --work-dir to sample another dataset row."
                    )
                if snapshot["sampling"] != config["sampling"]:
                    raise RuntimeError(
                        "Sampling settings differ from saved trajectories. "
                        "Use a new --work-dir or restore the original settings."
                    )
                if snapshot.get("complete"):
                    self.progress("Reusing trajectories.json; no new sampling.")
                    return snapshot
            if snapshot is None:
                dataset = select_row(config["dataset"]["path"], config["dataset"]["row_index"])
                snapshot = {"schema_version": 2, "snapshot_id": uuid.uuid4().hex,
                            "created_at": utc_now(), "dataset": dataset,
                            "sampling": config["sampling"], "complete": False, "trajectories": []}
            save = lambda data: atomic_json(self.directory / "trajectories.json", data)
            if (config["sampling"]["workers"] > 1 and
                    env_factory is make_environment and policy is request_action):
                return capture_parallel_trajectories(
                    snapshot, config["sampling"], save, self.directory, progress=self.progress)
            # Custom test callbacks run serially; they may not be picklable.
            return capture_trajectories(
                snapshot, config["sampling"], save,
                env_factory=env_factory, policy=policy, progress=self.progress)

    def enrich(self, *, replay=False, env_factory=make_environment):
        with self.session_lock():
            snapshot_path = self.directory / "trajectories.json"
            snapshot = read_json(snapshot_path)
            if not snapshot:
                raise ValueError("No saved trajectories to enrich")
            source_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
            converted = enrich_snapshot(snapshot, replay=replay, env_factory=env_factory, progress=self.progress)
            payload = _enrichment_payload(snapshot, converted, source_hash)
            atomic_json(self.directory / "goal_metadata.json", payload)
            self.progress("Saved goal_metadata.json; original trajectories and actions are unchanged.")
            return _merge_enrichment(snapshot, payload, source_hash)

    def _prepare_scoring_snapshot(self, snapshot, probe):
        needs_metadata = "{goal_response}" in probe["response"] or probe.get("state_view") == "full_state"
        if not needs_metadata:
            return snapshot
        source_hash = hashlib.sha256((self.directory / "trajectories.json").read_bytes()).hexdigest()
        sidecar = read_json(self.directory / "goal_metadata.json")
        if sidecar:
            return _merge_enrichment(snapshot, sidecar, source_hash)
        complete = all(trajectory.get("goal_response") and all(
            state.get("state_view") == "full_state" for state in trajectory["states"])
            for trajectory in snapshot.get("trajectories", []))
        if complete:
            hashes = {trajectory.get("goal_spec_hash") for trajectory in snapshot["trajectories"]}
            if len(hashes) != 1:
                raise ValueError("Saved goal specifications differ across rollouts")
            return snapshot
        converted = enrich_snapshot(snapshot, replay=False, progress=self.progress)
        payload = _enrichment_payload(snapshot, converted, source_hash)
        atomic_json(self.directory / "goal_metadata.json", payload)
        return _merge_enrichment(snapshot, payload, source_hash)

    def score(self, *, scorer=None):
        with self.session_lock():
            snapshot = read_json(self.directory / "trajectories.json")
            if not snapshot or not snapshot.get("complete"):
                raise RuntimeError("Complete the initial sampling first. Scoring never samples automatically.")
            config = copy.deepcopy(self.config)
            snapshot = self._prepare_scoring_snapshot(snapshot, config["probe"])
            if scorer is None:
                from alfworld_trace_probe_scoring import score_snapshots
                scorer = score_snapshots
            result = scorer(snapshot, config["probe"], config["scorer"], progress=self.progress)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
            result.update({"id": run_id, "created_at": utc_now(),
                           "snapshot_id": snapshot["snapshot_id"], "probe": config["probe"],
                           "scorer": config["scorer"],
                           "snapshot_sha256": hashlib.sha256(
                               (self.directory / "trajectories.json").read_bytes()).hexdigest()})
            result = jsonable(result)
            atomic_json(self.directory / "runs" / f"{run_id}.json", result)
            atomic_json(self.directory / "latest.json", result)
            (self.directory / "latest.csv").write_text(export_csv(result), encoding="utf-8-sig")
            (self.directory / "report.html").write_text(export_report(result), encoding="utf-8")
            self.progress(f"Saved scoring run {run_id}; original trajectories unchanged.")
            return result

    def snapshot(self):
        """Load the immutable raw cache with its verified goal sidecar."""
        with self.lock:
            path = self.directory / "trajectories.json"
            snapshot = read_json(path)
            sidecar = read_json(self.directory / "goal_metadata.json")
            if snapshot and sidecar:
                source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                snapshot = _merge_enrichment(snapshot, sidecar, source_hash)
            return snapshot

    def state(self, *, lightweight=False):
        with self.lock:
            if lightweight:
                # Full snapshot and scoring details are fetched on demand, not
                # attached to the frequently polled status endpoint.
                runs = [
                    {"id": path.stem,
                     "created_at": datetime.fromtimestamp(
                         path.stat().st_mtime, timezone.utc
                     ).isoformat()}
                    for path in sorted((self.directory / "runs").glob("*.json"), reverse=True)
                ]
                snap_path = self.directory / "trajectories.json"
                stat = snap_path.stat() if snap_path.is_file() else None
                sidecar_path = self.directory / "goal_metadata.json"
                sidecar_stat = sidecar_path.stat() if sidecar_path.is_file() else None
                version = f"{stat.st_mtime_ns}:{stat.st_size}" if stat else None
                if version and sidecar_stat:
                    version += f":goal:{sidecar_stat.st_mtime_ns}:{sidecar_stat.st_size}"
                return {
                    "config": copy.deepcopy(self.config),
                    "config_path": str(self.config_path),
                    "work_dir": str(self.directory),
                    "snapshot": None,
                    "snapshot_version": version,
                    "goal_enrichment": sidecar_stat is not None,
                    "latest": runs[0] if runs and (self.directory / "latest.json").is_file() else None,
                    "runs": runs,
                    "job": dict(self.job),
                }
            runs = []
            for path in sorted((self.directory / "runs").glob("*.json"), reverse=True):
                item = read_json(path)
                runs.append({key: item.get(key) for key in ("id", "created_at", "probe", "summary")})
            runs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
            snapshot = self.snapshot()
            return {"config": copy.deepcopy(self.config), "config_path": str(self.config_path),
                    "work_dir": str(self.directory),
                    "snapshot": snapshot,
                    "latest": read_json(self.directory / "latest.json"),
                    "goal_enrichment": (self.directory / "goal_metadata.json").exists(),
                    "runs": runs, "job": dict(self.job)}

    def start(self, phase, probe=None):
        with self.lock:
            if self.job["status"] == "running":
                raise RuntimeError("A job is already running.")
            if phase == "sample":
                snapshot = read_json(self.directory / "trajectories.json")
                if snapshot and snapshot.get("complete"):
                    raise ValueError("Trajectories already saved. Use score; a new work-dir is needed for a new sample.")
            if probe is not None:
                self.save_probe(probe)
            self.job = {"status": "running", "phase": phase, "message": "Starting…", "error": None}

        def run():
            try:
                if phase == "sample":
                    self.sample()
                elif phase == "enrich":
                    self.enrich(replay=True)
                else:
                    self.score()
            except Exception as error:
                traceback.print_exc()
                with self.lock:
                    self.job.update(status="error", error=str(error), message=str(error))
            else:
                with self.lock:
                    self.job.update(status="done", error=None)
        threading.Thread(target=run, daemon=True).start()



class SampleWorkspace:
    """Independent saved ALFWorld samples; legacy files remain in the root directory."""

    _pattern = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{8}$")

    def __init__(self, experiment):
        self.root = experiment.directory
        self.lock = threading.RLock()
        self.samples_dir = self.root / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.active_file = self.root / "active_sample.json"
        self.instances = {"legacy": experiment}
        self._snapshot_summary_cache = {}
        stored = read_json(self.active_file, {})
        selected = stored.get("sample_id", "legacy") if isinstance(stored, dict) else "legacy"
        if selected != "legacy" and not self._sample_path(selected).is_dir():
            selected = "legacy"
        self.active_id = selected

    def _sample_path(self, sample_id):
        if not isinstance(sample_id, str) or not self._pattern.fullmatch(sample_id):
            raise ValueError("Invalid saved sample ID")
        return self.samples_dir / sample_id

    def _load(self, sample_id):
        if sample_id in self.instances:
            return self.instances[sample_id]
        path = self._sample_path(sample_id)
        if not (path / "config.json").is_file():
            raise ValueError("Saved sample does not have a config.json")
        instance = Experiment(path)
        self.instances[sample_id] = instance
        return instance

    @property
    def current(self):
        return self._load(self.active_id)

    @property
    def directory(self):
        return self.current.directory

    def _ensure_idle(self):
        if self.current.job["status"] == "running":
            raise RuntimeError("Finish the current sampling or scoring job before switching samples.")

    def _snapshot_summary(self, file):
        """Read each large saved snapshot only when it changes."""
        if not file.is_file():
            return None
        stat = file.stat()
        version = (stat.st_mtime_ns, stat.st_size)
        cache_key = str(file)
        cached = self._snapshot_summary_cache.get(cache_key)
        if cached and cached[0] == version:
            return cached[1]
        try:
            snapshot = read_json(file)
        except json.JSONDecodeError as exc:
            # A corrupted historical sample must never break the entire editor.
            # Keep it listed so it can be inspected and explicitly recovered.
            value = {
                "dataset": {}, "created_at": None,
                "complete": False, "trajectory_count": 0,
                "corrupt_error": (
                    f"Invalid JSON in {file}: {exc.msg} at "
                    f"line {exc.lineno}, column {exc.colno} (char {exc.pos}). "
                    "The saved file is preserved. Recover from checkpoints or select another sample."
                ),
            }
        else:
            value = {
                "dataset": (snapshot or {}).get("dataset") or {},
                "created_at": (snapshot or {}).get("created_at"),
                "complete": bool((snapshot or {}).get("complete")),
                "trajectory_count": len((snapshot or {}).get("trajectories") or []),
            }
        self._snapshot_summary_cache[cache_key] = (version, value)
        return value

    def list_samples(self):
        ids = ["legacy"] + sorted(
            (p.name for p in self.samples_dir.iterdir()
             if p.is_dir() and self._pattern.fullmatch(p.name) and (p / "config.json").is_file()),
            reverse=True,
        )
        result = []
        for sample_id in ids:
            directory = self.root if sample_id == "legacy" else self._sample_path(sample_id)
            config = read_json(directory / "config.json", {})
            snapshot = self._snapshot_summary(directory / "trajectories.json")
            has_score = (directory / "latest.json").is_file()
            info = read_json(directory / "sample_info.json", {})
            dataset = config.get("dataset") or {}
            sampling = config.get("sampling") or {}
            meta = (snapshot or {}).get("dataset") or {}
            instance = self.instances.get(sample_id)
            job = instance.job if instance else None
            trajectory_count = (snapshot or {}).get("trajectory_count", 0)
            status = (
                "corrupt" if snapshot and snapshot.get("corrupt_error") else
                "running" if job and job.get("status") == "running" else
                "error" if job and job.get("status") == "error" else
                "complete" if snapshot and snapshot.get("complete") else
                "partial" if snapshot else "not_started"
            )
            result.append({
                "id": sample_id, "row_index": dataset.get("row_index"),
                "question": meta.get("question"),
                "created_at": (snapshot or {}).get("created_at") or info.get("created_at"),
                "trajectories": trajectory_count,
                "expected_trajectories": sampling.get("trajectories"),
                "status": status, "has_score": has_score,
                "corrupt_error": (snapshot or {}).get("corrupt_error"),
                "can_recover": bool(list((directory / "partial_trajectories").glob("traj_*.json"))),
                "legacy": sample_id == "legacy",
            })
        return result

    def state(self):
        with self.lock:
            value = self.current.state(lightweight=True)
            samples = self.list_samples()
            active = next((sample for sample in samples if sample["id"] == self.active_id), None)
            value.update(active_sample_id=self.active_id, samples=samples,
                         workspace_dir=str(self.root))
            if active and active.get("corrupt_error"):
                value["snapshot_error"] = active["corrupt_error"]
                value["snapshot_version"] = None  # Avoid repeatedly fetching invalid JSON.
            return value

    def select_sample(self, sample_id):
        with self.lock:
            self._ensure_idle()
            if sample_id != "legacy":
                self._sample_path(sample_id)
            self._load(sample_id)
            self.active_id = sample_id
            atomic_json(self.active_file, {"sample_id": sample_id})
            return {"sample_id": sample_id}

    def create_sample(self, options):
        """Create, select and start a new sample without changing saved samples."""
        with self.lock:
            self._ensure_idle()
            if not isinstance(options, dict):
                raise ValueError("Expected a sample configuration object")
            allowed = {"row_index", "trajectories", "workers", "seed", "temperature"}
            if set(options) - allowed:
                raise ValueError("Unknown sample configuration field")
            row_index = options.get("row_index")
            if type(row_index) is not int or row_index < 0:
                raise ValueError("row_index must be a nonnegative integer")
            config = copy.deepcopy(self.current.config)
            select_row(config["dataset"]["path"], row_index)
            overrides = {"dataset": {"row_index": row_index}, "sampling": {}}
            for key in ("trajectories", "workers", "seed", "temperature"):
                if key not in options:
                    continue
                value = options[key]
                if key in ("trajectories", "workers") and (type(value) is not int or value < 1):
                    raise ValueError(f"{key} must be a positive integer")
                if key == "seed" and (type(value) is not int or value < 0):
                    raise ValueError("seed must be a nonnegative integer")
                if key == "temperature" and (
                    type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 2
                ):
                    raise ValueError("temperature must be a finite value from 0 to 2")
                overrides["sampling"][key] = value
            deep_update(config, overrides)
            Experiment.validate_config(config)
            sample_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
            path = self._sample_path(sample_id)
            instance = Experiment(path, overrides=config)
            atomic_json(path / "sample_info.json", {"created_at": utc_now(), "sample_id": sample_id})
            self.instances[sample_id] = instance
            self.active_id = sample_id
            atomic_json(self.active_file, {"sample_id": sample_id})
            instance.start("sample")
            return {"sample_id": sample_id}


    def recover_snapshot(self):
        """Explicit recovery from per-worker checkpoints, never silent overwrite."""
        with self.lock:
            self._ensure_idle()
            instance = self.current
            file = instance.directory / "trajectories.json"
            if not file.is_file():
                raise ValueError("No trajectories.json exists; use normal sampling instead.")
            try:
                read_json(file)
            except json.JSONDecodeError:
                pass
            else:
                raise ValueError("trajectories.json is already valid; no recovery required.")
            checkpoints = sorted((instance.directory / "partial_trajectories").glob("traj_*.json"))
            if not checkpoints:
                raise ValueError(
                    "No trajectory checkpoints were found for this sample. "
                    "The damaged JSON has not been modified; inspect the latest.json "
                    "and runs/ backups or create a new sample."
                )
            config = instance.config
            sampling = copy.deepcopy(config["sampling"])
            valid = {}
            snapshot_ids = set()
            for checkpoint in checkpoints:
                try:
                    data = read_json(checkpoint)
                except json.JSONDecodeError:
                    continue
                if data.get("sampling") != {**sampling, "trajectories": 1}:
                    continue
                trajectory = (data.get("trajectories") or [None])[0]
                if not trajectory or not re.fullmatch(r"traj_\d+", trajectory.get("trajectory_id", "")):
                    continue
                index = int(trajectory["trajectory_id"].split("_")[1]) - 1
                if not 0 <= index < sampling["trajectories"]:
                    continue
                if not trajectory.get("states") or len(trajectory["states"]) != len(trajectory.get("turns", [])) + 1:
                    continue
                if not data.get("snapshot_id"):
                    continue
                snapshot_ids.add(data["snapshot_id"])
                valid[index] = trajectory
            if not valid:
                raise ValueError(
                    "No valid trajectory checkpoints matching the current sample were found. "
                    "The damaged JSON has not been modified."
                )
            if len(snapshot_ids) != 1:
                raise ValueError(
                    "Checkpoints belong to different sample sessions. Nothing was modified."
                )
            trajectories = [valid[i] for i in sorted(valid)]
            first = trajectories[0]
            for trajectory in trajectories[1:]:
                if (trajectory.get("initial_gamefile") != first.get("initial_gamefile")
                        or trajectory["states"][0].get("state_context") != first["states"][0].get("state_context")):
                    raise ValueError("Checkpoint initial states differ; nothing was modified.")
            dataset = select_row(config["dataset"]["path"], config["dataset"]["row_index"])
            complete = len(valid) == sampling["trajectories"] and all(
                tr.get("status") in ("completed", "external_error") for tr in trajectories
            )
            restored = {
                "schema_version": 1, "snapshot_id": next(iter(snapshot_ids)),
                "created_at": utc_now(), "dataset": dataset,
                "sampling": sampling, "complete": complete,
                "trajectories": trajectories,
            }
            if complete:
                restored["completed_at"] = utc_now()
            # Never delete the corrupted original, even on successful recovery.
            backup = file.with_name(
                file.name + ".corrupt." +
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") +
                "." + uuid.uuid4().hex[:6] + ".bak"
            )
            shutil.copy2(file, backup)
            atomic_json(file, restored)
            self._snapshot_summary_cache.pop(str(file), None)
            return {
                "restored_trajectories": len(trajectories),
                "complete": complete, "backup_file": str(backup),
            }

    def save_probe(self, probe):
        with self.lock:
            return self.current.save_probe(probe)

    def start(self, phase, probe=None):
        with self.lock:
            return self.current.start(phase, probe)

    def snapshot(self):
        with self.lock:
            return self.current.snapshot()

def serve(experiment, host, port):
    workspace = SampleWorkspace(experiment)

    class Handler(BaseHTTPRequestHandler):
        def send(self, data, content_type="application/json; charset=utf-8", status=200, filename=None):
            if not isinstance(data, (bytes, str)):
                data = json.dumps(jsonable(data), ensure_ascii=False, allow_nan=False)
            data = data.encode("utf-8") if isinstance(data, str) else data
            compressed = False
            if len(data) > 65536 and "gzip" in self.headers.get("Accept-Encoding", "").lower():
                data = gzip.compress(data, compresslevel=5)
                compressed = True
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    return self.send((SCRIPT_DIR / "alfworld_trace_probe.html").read_bytes(), "text/html; charset=utf-8")
                if parsed.path == "/api/state":
                    return self.send(workspace.state())
                if parsed.path == "/api/snapshot":
                    file = workspace.directory / "trajectories.json"
                    try:
                        snapshot = workspace.snapshot()
                    except json.JSONDecodeError as exc:
                        return self.send({
                            "error": f"Invalid JSON in {file}: {exc}. "
                                     "The file is preserved; use checkpoint recovery.",
                            "sample_id": workspace.active_id,
                        }, status=422)
                    if snapshot is None:
                        raise FileNotFoundError("No saved trajectories yet.")
                    return self.send(snapshot)
                if parsed.path == "/api/run":
                    run_id = parse_qs(parsed.query).get("id", [""])[0]
                    if not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", run_id):
                        raise ValueError("Invalid run id")
                    result = read_json(workspace.directory / "runs" / f"{run_id}.json")
                    if result is None:
                        raise FileNotFoundError(f"Saved scoring run not found: {run_id}")
                    return self.send(result)
                if parsed.path == "/api/export":
                    name = parse_qs(parsed.query).get("name", [""])[0]
                    files = {"trajectories": ("trajectories.json", "application/json"),
                             "latest": ("latest.json", "application/json"),
                             "csv": ("latest.csv", "text/csv; charset=utf-8"),
                             "report": ("report.html", "text/html; charset=utf-8")}
                    if name not in files:
                        raise ValueError("Unknown export")
                    filename, content_type = files[name]
                    return self.send((workspace.directory / filename).read_bytes(), content_type, filename=filename)
                self.send({"error": "Not found"}, status=404)
            except (ValueError, FileNotFoundError) as error:
                self.send({"error": str(error)}, status=400)

        def do_POST(self):
            try:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected application/json")
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise ValueError("Cross-origin mutations are not accepted")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                with workspace.lock:
                    if self.path == "/api/recover-snapshot":
                        recovered = workspace.recover_snapshot()
                        return self.send({"ok": True, **recovered})
                    if self.path == "/api/new-sample":
                        created = workspace.create_sample(payload)
                        return self.send({"ok": True, **created})
                    if self.path == "/api/select-sample":
                        selected = workspace.select_sample(payload["sample_id"])
                        return self.send({"ok": True, **selected})
                    if workspace.current.job["status"] == "running":
                        raise RuntimeError("A job is running; finish it before changing the probe.")
                    if self.path == "/api/config":
                        workspace.save_probe(payload["probe"])
                    elif self.path == "/api/sample":
                        workspace.start("sample")
                    elif self.path == "/api/score":
                        workspace.start("score", payload.get("probe"))
                    elif self.path == "/api/enrich":
                        workspace.start("enrich")
                    else:
                        return self.send({"error": "Not found"}, status=404)
                self.send({"ok": True})
            except (ValueError, KeyError, TypeError, RuntimeError, IndexError, FileNotFoundError) as error:
                self.send({"error": str(error)}, status=400)

        def log_message(self, fmt, *args):
            if self.path != "/api/state":
                super().log_message(fmt, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Editor: http://{host}:{server.server_port}\nWork directory: {experiment.directory}\n"
          "Sampling starts only when you click the initial-sampling button. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _shutdown_persistent_pool()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("serve", "sample", "enrich", "score", "run"), default="serve")
    parser.add_argument("--work-dir", default=str(REPO_ROOT / "outputs" / "alfworld_trace_probe"))
    parser.add_argument("--config", help="JSON configuration; defaults to WORK_DIR/config.json")
    parser.add_argument("--replay", action="store_true", help="enrich only: replay saved actions without policy generation")
    parser.add_argument("--dataset")
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--trajectories", type=int)
    parser.add_argument("--workers", type=int, help="Concurrent independent ALFWorld rollout processes")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--policy-base-url")
    parser.add_argument("--policy-model")
    parser.add_argument("--score-base-url")
    parser.add_argument("--score-model")
    parser.add_argument("--tokenizer", help="Tokenizer directory/name matching the frozen scoring model")
    parser.add_argument("--backend", choices=("vllm", "hf"))
    parser.add_argument("--alfworld-root", help="Source directory containing the alfworld Python package")
    parser.add_argument("--alfworld-data", help="Directory containing json_2.1.1/ and logic/")
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    overrides = {}
    mappings = {
        "dataset": ("dataset", "path"), "row_index": ("dataset", "row_index"),
        "trajectories": ("sampling", "trajectories"), "workers": ("sampling", "workers"),
        "max_steps": ("sampling", "max_steps"),
        "seed": ("sampling", "seed"), "temperature": ("sampling", "temperature"),
        "policy_base_url": ("sampling", "base_url"), "policy_model": ("sampling", "model"),
        "score_base_url": ("scorer", "base_url"), "score_model": ("scorer", "model"),
        "tokenizer": ("scorer", "tokenizer"), "backend": ("scorer", "backend"),
        "alfworld_root": ("sampling", "alfworld_root"), "alfworld_data": ("sampling", "alfworld_data"),
    }
    for name, (section, key) in mappings.items():
        value = getattr(args, name)
        if value is not None:
            overrides.setdefault(section, {})[key] = value
    experiment = Experiment(args.work_dir, overrides, args.config)
    if args.mode == "serve":
        serve(experiment, args.host, args.port)
    elif args.mode == "sample":
        experiment.sample()
    elif args.mode == "score":
        experiment.score()
    elif args.mode == "enrich":
        experiment.enrich(replay=args.replay)
    else:
        experiment.sample()
        experiment.score()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as error:
        raise SystemExit(str(error))
