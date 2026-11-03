"""Prepared ALFWorld Parquet artifacts; training reads GT from ``result``.

Arrow is imported only by offline writing. Runtime helpers validate the saved
row/specification and actual game bytes without parsing or converting PDDL.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time


class ParquetGoalError(ValueError):
    """A prepared dataset or its source game is missing, stale or inconsistent."""


_QUESTION = re.compile(r"(train|val_seen|val_unseen)_([a-z0-9]+)_alfworld_(\d+)")
_COLUMNS = ("alfworld_goal_spec", "alfworld_goal_spec_sha256", "alfworld_game_key", "alfworld_goal_row")


def _store_tools():
    name = "_alfworld_parquet_saved_goal_tools"
    if name not in sys.modules:
        path = Path(__file__).with_name("alfworld_goal_store.py")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ParquetGoalError(f"Cannot import saved goal helpers: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _path(value, description):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ParquetGoalError(f"Missing {description}")
    return Path(value).expanduser().resolve()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _game_key(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ParquetGoalError("Prepared Parquet has an invalid game key")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or path.name != "game.tw-pddl":
        raise ParquetGoalError("Prepared Parquet game key must stay inside the games root")
    return path


def _task_payload(task):
    tools = _store_tools()
    if not isinstance(task, Mapping):
        raise ParquetGoalError("ALFWorld TRACE requires a prepared Parquet row")
    question = task.get("question")
    match = _QUESTION.fullmatch(question) if isinstance(question, str) else None
    if match is None:
        raise ParquetGoalError("Prepared Parquet has an invalid question")
    if not isinstance(task.get("result"), str) or not task["result"].strip():
        raise ParquetGoalError("Prepared Parquet result is empty; run the offline prepare command first")
    payload, checksum, raw_row = (task.get(key) for key in ("alfworld_goal_spec", "alfworld_goal_spec_sha256", "alfworld_goal_row"))
    if not isinstance(payload, str) or not tools._digest(checksum) or _sha(payload.encode("utf-8")) != checksum:
        raise ParquetGoalError("Prepared Parquet goal specification failed integrity validation")
    try:
        spec, row = json.loads(payload), json.loads(raw_row)
    except (TypeError, ValueError) as error:
        raise ParquetGoalError("Prepared Parquet goal specification/row is not valid JSON") from error
    if not isinstance(spec, dict) or not isinstance(row, dict):
        raise ParquetGoalError("Prepared Parquet goal specification/row must be JSON objects")
    data_type, task_type, number = match.groups()
    if (row.get("question") != question or row.get("data_type") != data_type
            or row.get("task_type") != task_type or type(row.get("question_id")) is not int
            or row["question_id"] != int(number) or type(row.get("row_index")) is not int
            or row["row_index"] < 0 or row.get("game_key") != task.get("alfworld_game_key")
            or not tools._digest(row.get("dataset_sha256"))
            or not tools._digest(row.get("selection_sha256"))):
        raise ParquetGoalError("Prepared Parquet question and saved row mapping disagree")
    _game_key(task["alfworld_game_key"])
    if task["result"] != spec.get("goal_response"):
        raise ParquetGoalError("Parquet result does not match its preconverted goal specification")
    return spec, row


def parquet_task_gamefile(task, games_root):
    """Use the row's frozen actual-game binding rather than directory order."""
    _task_payload(task)
    root = _path(games_root, "runtime games root")
    game = (root / _game_key(task["alfworld_game_key"])).resolve()
    if not game.is_relative_to(root):
        raise ParquetGoalError("Prepared Parquet game escapes the runtime games root")
    read_parquet_task_goal(task, game)
    return game


def read_parquet_task_goal(task, gamefile):
    """Read the scoring GT from result and verify its embedded offline rules."""
    spec, row = _task_payload(task)
    tools = _store_tools()
    try:
        game, trajectory, game_hash, traj_hash = tools._source_hashes(gamefile)
    except (ValueError, OSError) as error:
        raise ParquetGoalError(str(error)) from error
    if (spec.get("template_version") not in tools.SUPPORTED_TEMPLATE_VERSIONS
            or not tools._valid_spec(spec, spec.get("template_version"), game_hash, traj_hash)):
        raise ParquetGoalError("Prepared Parquet goal rules or actual game's source hashes are invalid/stale")
    spec["goal_response"] = task["result"]
    spec["gamefile"], spec["goal_source"], spec["traj_data_path"] = str(game), str(trajectory), str(trajectory)
    spec.update({"goal_source_type": "parquet.result", "parquet_goal_spec_sha256": task["alfworld_goal_spec_sha256"],
                 "parquet_goal_row": task["alfworld_goal_row"], "goal_store_selection_mode": "dataset",
                 "goal_store_dataset_sha256": row["dataset_sha256"],
                 "goal_store_selection_sha256": row["selection_sha256"],
                 "goal_store_dataset_rows": row.get("dataset_rows"),
                 "goal_store_id": row.get("goal_store_id"), "goal_store_entry": row["game_key"]})
    return spec


def write_goal_parquet(dataset, store_path, output, *, overwrite=False):
    """Fill result and append saved rules, preserving other Arrow columns."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ParquetGoalError("Offline Parquet preparation requires pyarrow") from error
    source, store, destination = _path(dataset, "source dataset"), _path(store_path, "saved goal store"), _path(output, "output Parquet")
    if not store.is_file():
        raise ParquetGoalError(f"Saved goal store does not exist: {store}")
    if os.path.lexists(destination) and not overwrite:
        raise ParquetGoalError(f"Output Parquet already exists: {destination}; use --overwrite explicitly")
    tools = _store_tools()
    try:
        original = source.read_bytes()
        table = pq.read_table(io.BytesIO(original))
        if "question" not in table.column_names or "result" not in table.column_names:
            raise ParquetGoalError("Source Parquet must contain question and result columns")
        questions = table.column("question").to_pylist()
        values = {"result": [], **{key: [] for key in _COLUMNS}}
        with closing(sqlite3.connect(store.as_uri() + "?mode=ro", uri=True)) as connection:
            metadata = tools._store_metadata(connection, store)
            if (metadata.get("selection_mode") != "dataset" or metadata.get("dataset_sha256") != _sha(original)
                    or metadata.get("dataset_rows") != table.num_rows):
                raise ParquetGoalError("Saved goals do not match this exact source Parquet and row count")
            names = ("row_index", "question", "data_type", "task_type", "question_id", "game_key")
            rows = connection.execute("""SELECT d.row_index,d.question,d.data_type,d.task_type,d.question_id,d.game_key,
                       d.row_sha256,g.game_sha256,g.traj_sha256,g.spec_sha256,g.spec_json
                       FROM dataset_rows d JOIN goals g ON g.game_key=d.game_key ORDER BY d.row_index""").fetchall()
            if len(rows) != table.num_rows:
                raise ParquetGoalError("Saved row-to-game mapping is incomplete")
            for index, saved in enumerate(rows):
                row = dict(zip(names, saved[:6]))
                if (row["row_index"] != index or row["question"] != questions[index]
                        or _sha(_json(row).encode("utf-8")) != saved[6]):
                    raise ParquetGoalError(f"Saved row-to-game mapping differs at dataset row {index}")
                _game_key(row["game_key"])
                payload, checksum = saved[10], saved[9]
                if not isinstance(payload, str) or _sha(payload.encode("utf-8")) != checksum:
                    raise ParquetGoalError(f"Saved goal payload failed integrity validation at row {index}")
                spec = json.loads(payload)
                if not tools._valid_spec(spec, metadata["template_version"], saved[7], saved[8]):
                    raise ParquetGoalError(f"Saved goal specification is invalid at row {index}")
                row.update({"dataset_sha256": metadata["dataset_sha256"],
                            "selection_sha256": metadata["selection_sha256"],
                            "dataset_rows": table.num_rows, "goal_store_id": metadata["goal_store_id"]})
                values["result"].append(spec["goal_response"])
                values["alfworld_goal_spec"].append(payload)
                values["alfworld_goal_spec_sha256"].append(checksum)
                values["alfworld_game_key"].append(row["game_key"])
                values["alfworld_goal_row"].append(_json(row))
        for name, column in values.items():
            dtype = pa.string()
            if name == "result" and pa.types.is_large_string(table.schema.field(name).type):
                dtype = pa.large_string()
            array = pa.array(column, type=dtype)
            if name in table.column_names:
                index = table.column_names.index(name)
                table = table.set_column(index, table.schema.field(index).with_type(dtype), array)
            else:
                table = table.append_column(name, array)
        summary = {"status": "complete", "artifact_type": "parquet", "output": str(destination),
                   "dataset_rows": table.num_rows, "count": table.num_rows, "converted": table.num_rows,
                   "result_column": "result", "goal_source": "parquet.result",
                   "dataset_sha256": metadata["dataset_sha256"],
                   "selection_sha256": metadata["selection_sha256"], "goal_store_id": metadata["goal_store_id"]}
        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata[b"alfworld_trace_preparation"] = _json(summary).encode("utf-8")
        table = table.replace_schema_metadata(schema_metadata)
    except (OSError, sqlite3.Error, ValueError, TypeError) as error:
        if isinstance(error, ParquetGoalError):
            raise
        raise ParquetGoalError(f"Cannot prepare Parquet rows: {error}") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".building-", suffix=".parquet", dir=destination.parent)
    os.close(descriptor)
    temporary, ready = Path(temporary), False
    try:
        pq.write_table(table, temporary, compression="zstd")
        checked = pq.read_table(temporary)
        if (not checked.schema.equals(table.schema, check_metadata=True) or not checked.equals(table)
                or any(not isinstance(value, str) or not value.strip() for value in checked.column("result").to_pylist())):
            raise ParquetGoalError("Written Parquet failed row/schema/result verification")
        ready = True
        tools._publish_goal_store(temporary, destination, overwrite)
        return summary
    except (OSError, ValueError) as error:
        retained = f" Complete Parquet retained at {temporary}." if ready and temporary.is_file() else ""
        raise ParquetGoalError(f"Cannot publish prepared Parquet {destination}: {error}.{retained}") from error
    finally:
        if not ready:
            temporary.unlink(missing_ok=True)


def prepare_dataset_parquet(dataset, games_root, output, *, expected_rows=3305,
                            alfworld_root=None, overwrite=False, progress=None,
                            reuse_store=None, catalogue_loader=None):
    """Convert the selected rows once and publish a self-contained Parquet."""
    destination = _path(output, "output Parquet")
    if os.path.lexists(destination) and not overwrite:
        raise ParquetGoalError(f"Output Parquet already exists: {destination}; use --overwrite explicitly")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, cache = tempfile.mkstemp(prefix=destination.name + ".goals-", suffix=".sqlite", dir=destination.parent)
    os.close(descriptor)
    cache = Path(cache)
    cache.unlink()
    tools, started, last = _store_tools(), time.monotonic(), None

    def capture(event):
        nonlocal last
        last = dict(event)
        if progress is not None and event["phase"] != "complete":
            progress(dict(event))

    try:
        tools.build_goal_store(games_root, cache, overwrite=False, progress=capture,
                               dataset=dataset, alfworld_root=alfworld_root, expected_rows=expected_rows,
                               reuse_store=reuse_store, catalogue_loader=catalogue_loader)
        saving = {**(last or {}), "phase": "saving", "artifact_type": "parquet",
                  "elapsed_seconds": time.monotonic() - started}
        if progress is not None:
            progress(saving)
        summary = write_goal_parquet(dataset, cache, destination, overwrite=overwrite)
        # Publication has already succeeded. A temporary-cache cleanup failure
        # must not turn a usable training dataset into an apparent failed run.
        try:
            cache.unlink(missing_ok=True)
        except OSError as error:
            summary["retained_cache"] = str(cache)
            summary["cache_cleanup_warning"] = str(error)
        if progress is not None:
            progress({**saving, "phase": "complete", "elapsed_seconds": time.monotonic() - started})
        return summary
    except (ValueError, OSError) as error:
        retained = (f" Completed goals retained at {cache}; rerun prepare with --reuse-store {cache} "
                    "to avoid converting PDDL again." if cache.is_file() else "")
        raise ParquetGoalError(f"Parquet preparation failed: {error}.{retained}") from error
