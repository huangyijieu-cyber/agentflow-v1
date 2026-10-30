"""Offline ALFWorld goal conversion and read-only, portable training artifacts.

Only ``build_goal_store`` imports the PDDL converter. Training loads indexed
saved specifications by the actual game's source hashes and never translates
or parses a domain/goal. The SQLite file can be shared by independent workers.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from urllib.parse import quote


STORE_FORMAT_VERSION = 1
SUPPORTED_TEMPLATE_VERSIONS = {"alfworld-pddl-controlled-english-v1"}
_REBUILD = "Build the file offline with alfworld_goal_store.py build; training will not convert missing goals."


class GoalStoreError(ValueError):
    """A saved goal artifact is missing, incompatible, stale, or invalid."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def _string_mapping(value):
    return isinstance(value, dict) and all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in value.items())


def _expression(value, depth=0):
    # Validate the serialized AST's shape without parsing any PDDL or
    # regenerating rules. Semantic validation already happened offline.
    return (depth < 128 and isinstance(value, list)
            and (not value or isinstance(value[0], str))
            and all(isinstance(item, str) or _expression(item, depth + 1) for item in value))


def _valid_spec(spec, template, game_hash, traj_hash):
    if not isinstance(spec, dict) or type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
        return False
    required_text = ("goal_response", "goal_task_type", "task_type", "rules_context", "initial_context",
                     "answer_instruction", "raw_goal", "gamefile", "goal_source", "traj_data_path")
    if any(not isinstance(spec.get(key), str) or not spec[key].strip() for key in required_text):
        return False
    if (spec.get("template_version") != template or spec.get("scoring_mode") != "full_state"
            or spec.get("goal_validation_status") != "actual_pddl_goal_and_metadata_checked"
            or spec["task_type"] != spec["goal_task_type"]
            or not _expression(spec.get("goal_ast")) or not spec["goal_ast"]
            or not _string_mapping(spec.get("objects")) or not spec["objects"]
            or not _string_mapping(spec.get("types"))
            or not _string_mapping(spec.get("entity_aliases"))
            or set(spec["entity_aliases"]) != set(spec["objects"])
            or not isinstance(spec.get("pddl_params"), dict)):
        return False
    for key in ("initial_facts", "initial_negative_facts", "initial_numeric_assignments"):
        if not isinstance(spec.get(key), list) or any(not isinstance(item, str) for item in spec[key]):
            return False
    sources = spec.get("source_sha256")
    if (not isinstance(sources, dict)
            or any(not _digest(sources.get(key)) for key in ("gamefile", "traj_data", "pddl_domain", "pddl_problem"))
            or sources["gamefile"] != game_hash or sources["traj_data"] != traj_hash):
        return False
    actions = spec.get("domain_actions")
    if not isinstance(actions, dict):
        return False
    for name, action in actions.items():
        if (not isinstance(name, str) or not isinstance(action, dict)
                or not isinstance(action.get("parameters"), list)
                or any(not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(x, str) for x in pair)
                       for pair in action["parameters"])
                or not _expression(action.get("precondition")) or not _expression(action.get("effect"))):
            return False
    stable = {"template_version": template, "sources": sources,
              "goal": spec["goal_ast"], "response": spec["goal_response"]}
    return spec.get("goal_spec_hash") == _sha(json.dumps(stable, sort_keys=True).encode())


def _converter():
    # A file-based import also works for the standalone CLI and lightweight
    # Probe, without importing agentflow's training package or ALFWorld.
    path = Path(__file__).with_name("alfworld.py")
    spec = importlib.util.spec_from_file_location("_alfworld_offline_goal_converter", path)
    if spec is None or spec.loader is None:
        raise GoalStoreError(f"Cannot import offline converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _path(value, description):
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise GoalStoreError(f"{description} must identify exactly one file")
        value = value[0]
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise GoalStoreError(f"Missing {description}. {_REBUILD}")
    return Path(value).expanduser().resolve()


def _source_hashes(gamefile):
    game = _path(gamefile, "actual extra.gamefile")
    traj = game.with_name("traj_data.json")
    try:
        return game, traj, _sha(game.read_bytes()), _sha(traj.read_bytes())
    except OSError as error:
        raise GoalStoreError(f"Cannot verify actual game's source files: {error}") from error


def _dataset_selector():
    path = Path(__file__).with_name("alfworld_dataset.py")
    spec = importlib.util.spec_from_file_location("_alfworld_offline_dataset_selector", path)
    if spec is None or spec.loader is None:
        raise GoalStoreError(f"Cannot import dataset selector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _publish_goal_store(temporary, destination, overwrite):
    if overwrite:
        os.replace(temporary, destination)
        return
    try:
        os.link(temporary, destination)  # atomic no-clobber when supported
    except OSError as error:
        unsupported = {errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOSYS, errno.EXDEV}
        if error.errno not in unsupported:
            raise
        # FUSE/cloud mounts commonly support rename but not hard links.
        # Serialize fallback publishers and never replace an existing target.
        lock = destination.with_name(destination.name + ".publish.lock")
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if os.path.lexists(destination):
                raise FileExistsError(errno.EEXIST, "Goal store already exists", str(destination))
            os.replace(temporary, destination)
        finally:
            os.close(descriptor)
            lock.unlink(missing_ok=True)
    else:
        temporary.unlink()


def build_goal_store(games_root, output, *, overwrite=False, progress=None, dataset=None,
                     alfworld_root=None, expected_rows=None, reuse_store=None,
                     catalogue_loader=None):
    """Convert all supported games once and atomically publish one SQLite file.

    The root can be the official json_2.1.1 folder or a particular split. The
    official TextWorld exclusions (movable/Sliced and unsolvable games) apply;
    every other conversion error aborts the build and preserves any old file.
    """
    started = time.monotonic()
    root = _path(games_root, "games root")
    destination = _path(output, "goal store output")
    if not root.is_dir():
        raise GoalStoreError(f"Games root is not a directory: {root}")
    if os.path.lexists(destination) and not overwrite:
        raise GoalStoreError(f"Goal store already exists: {destination}; use --overwrite to rebuild explicitly")
    destination.parent.mkdir(parents=True, exist_ok=True)
    converter = None if reuse_store is not None else _converter()
    template_version = converter.TEMPLATE_VERSION if converter is not None else None
    if converter is not None and template_version not in SUPPORTED_TEMPLATE_VERSIONS:
        raise GoalStoreError("Offline converter template is incompatible with this goal store reader")
    if dataset is None and expected_rows is not None:
        raise GoalStoreError("expected_rows requires a dataset")
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".building-", suffix=".sqlite", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    connection = None
    ready_to_publish = False
    records = []
    skipped_unsupported = skipped_unsolvable = 0
    selection = None

    def report(phase, processed, total, game=None):
        if progress is not None:
            progress({"phase": phase, "processed": processed, "total": total,
                      "converted": len(records), "skipped_unsupported": skipped_unsupported,
                      "skipped_unsolvable": skipped_unsolvable,
                      "elapsed_seconds": time.monotonic() - started, "current_game": game,
                      "selection_mode": "dataset" if dataset is not None else "all_games"})

    try:
        report("scanning", 0, None)
        if dataset is not None:
            try:
                selection = _dataset_selector().select_dataset_games(
                    dataset, root, alfworld_root=alfworld_root, expected_rows=expected_rows,
                    catalogue_loader=catalogue_loader)
            except (ValueError, OSError, ImportError) as error:
                raise GoalStoreError(f"Cannot resolve dataset games: {error}") from error
            games = selection["games"]
            keys = [Path(game).relative_to(root).as_posix() for game in games]
            row_keys = [row["game_key"] for row in selection["rows"]]
            if not keys or len(keys) != len(row_keys) or len(set(keys)) != len(keys) or set(keys) != set(row_keys):
                raise GoalStoreError("Dataset rows must map one-to-one to the selected actual games")
        else:
            games = sorted(root.rglob("game.tw-pddl"))
        total = len(games)
        report("converting", 0, total)
        connection = sqlite3.connect(str(temporary))
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE goals (
                game_key TEXT PRIMARY KEY,
                game_sha256 TEXT NOT NULL,
                traj_sha256 TEXT NOT NULL,
                spec_sha256 TEXT NOT NULL,
                spec_json TEXT NOT NULL
            );
            CREATE INDEX goals_sources ON goals (game_sha256, traj_sha256);
        """)
        if selection is not None:
            connection.execute("""CREATE TABLE dataset_rows (
                row_index INTEGER PRIMARY KEY, question TEXT NOT NULL UNIQUE,
                data_type TEXT NOT NULL, task_type TEXT NOT NULL, question_id INTEGER NOT NULL,
                game_key TEXT NOT NULL UNIQUE REFERENCES goals(game_key), row_sha256 TEXT NOT NULL,
                UNIQUE(data_type, task_type, question_id))""")
        for processed, game in enumerate(games, 1):
            key = game.relative_to(root).as_posix()
            if "movable" in key or "Sliced" in key:
                if selection is not None:
                    raise GoalStoreError(f"Selected dataset game is an unsupported official variant: {game}")
                skipped_unsupported += 1
                report("progress", processed, total, key)
                continue
            try:
                raw_game = json.loads(game.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise GoalStoreError(f"Cannot read game {game}: {error}") from error
            if not isinstance(raw_game, dict):
                raise GoalStoreError(f"Game must be a JSON object: {game}")
            if raw_game.get("solvable") is False:
                if selection is not None:
                    raise GoalStoreError(f"Selected dataset game is unsolvable: {game}")
                skipped_unsolvable += 1
                report("progress", processed, total, key)
                continue
            try:
                if reuse_store is not None:
                    saved = read_goal_spec(reuse_store, game)
                    spec = {name: value for name, value in saved.items() if not name.startswith("goal_store_")}
                else:
                    spec = converter.load_goal_spec(game)
            except (ValueError, OSError) as error:
                operation = "Saved goal reuse" if reuse_store is not None else "Offline conversion"
                raise GoalStoreError(f"{operation} failed for {game}: {error}") from error
            if template_version is None:
                template_version = spec["template_version"]
            if spec["template_version"] != template_version:
                raise GoalStoreError("Cannot mix different goal template versions in one store")
            payload = _json(spec)
            payload_hash = _sha(payload.encode("utf-8"))
            sources = spec["source_sha256"]
            connection.execute("INSERT INTO goals VALUES (?, ?, ?, ?, ?)",
                               (key, sources["gamefile"], sources["traj_data"], payload_hash, payload))
            records.append([key, spec["goal_spec_hash"], payload_hash])
            report("progress", processed, total, key)
        if not records:
            raise GoalStoreError("No supported solvable games were found; refusing to publish an empty goal store")
        report("saving", total, total)
        if selection is not None:
            for row in selection["rows"]:
                connection.execute("INSERT INTO dataset_rows VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   tuple(row[key] for key in ("row_index", "question", "data_type", "task_type", "question_id", "game_key"))
                                   + (_sha(_json(row).encode("utf-8")),))
        identity = {"format": STORE_FORMAT_VERSION, "template": template_version, "records": records}
        if selection is not None:
            identity["selection_sha256"] = selection["metadata"]["selection_sha256"]
        store_id = _sha(_json(identity).encode("utf-8"))
        summary = {"goal_store_path": str(destination), "goal_store_id": store_id,
                   "template_version": template_version, "count": len(records),
                   "converted": len(records), "skipped_unsupported": skipped_unsupported,
                   "skipped_unsolvable": skipped_unsolvable, "games_root": str(root),
                   "selection_mode": "dataset" if selection is not None else "all_games"}
        if selection is not None:
            summary.update(selection["metadata"])
        if reuse_store is not None:
            summary["reused"] = len(records)
            summary["reuse_store"] = str(_path(reuse_store, "source goal store"))
        metadata = {"store_format_version": STORE_FORMAT_VERSION, "status": "complete", **summary}
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", ((key, _json(value)) for key, value in metadata.items()))
        connection.commit()
        connection.close()
        connection = None
        # Never truncate an existing artifact or expose a partially built one.
        ready_to_publish = True
        _publish_goal_store(temporary, destination, overwrite)
        report("complete", total, total)
        return summary
    except (sqlite3.Error, OSError) as error:
        retained = (f" Completed SQLite retained at {temporary}; inspect it with show or use it as --reuse-store."
                    if ready_to_publish and temporary.is_file() else "")
        raise GoalStoreError(f"Cannot build goal store {destination}: {error}.{retained}") from error
    finally:
        if connection is not None:
            connection.close()
        if not ready_to_publish:
            temporary.unlink(missing_ok=True)


def _store_metadata(connection, path):
    metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
    if (metadata.get("store_format_version") != STORE_FORMAT_VERSION
            or metadata.get("status") != "complete"
            or metadata.get("template_version") not in SUPPORTED_TEMPLATE_VERSIONS
            or type(metadata.get("count")) is not int or metadata["count"] < 1
            or not _digest(metadata.get("goal_store_id"))
            or metadata.get("selection_mode", "all_games") not in ("all_games", "dataset")):
        raise GoalStoreError(f"Saved goal store is incomplete or incompatible: {path}. {_REBUILD}")
    if connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0] != metadata["count"]:
        raise GoalStoreError(f"Saved goal store is incomplete: goal count differs from the completed build: {path}. {_REBUILD}")
    if metadata.get("selection_mode") == "dataset":
        count = metadata.get("dataset_rows")
        if (type(count) is not int or count != metadata["count"]
                or not _digest(metadata.get("dataset_sha256"))
                or not _digest(metadata.get("selection_sha256"))
                or connection.execute("SELECT COUNT(*) FROM dataset_rows").fetchone()[0] != count):
            raise GoalStoreError(f"Saved dataset selection is incomplete or incompatible: {path}. {_REBUILD}")
    return metadata


def read_dataset_gamefile(store_path, data_type, task_type, question_id, games_root):
    """Resolve a dataset question from its frozen mapping, without rescanning.

    Legacy directory-wide stores return None so their original environment
    selection remains compatible. Dataset-scoped stores reject uncovered rows.
    """
    path = _path(store_path, "ALFWORLD_TRACE_GOALS / saved goal store")
    if not path.is_file():
        raise GoalStoreError(f"Saved goal store does not exist: {path}. {_REBUILD}")
    if type(question_id) is not int or question_id < 0:
        raise GoalStoreError("Dataset question_id must be a nonnegative integer")
    uri = "file:" + quote(str(path), safe="/") + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            metadata = _store_metadata(connection, path)
            if metadata.get("selection_mode") != "dataset":
                return None
            names = ("row_index", "question", "data_type", "task_type", "question_id", "game_key")
            row = connection.execute(
                "SELECT row_index,question,data_type,task_type,question_id,game_key,row_sha256 "
                "FROM dataset_rows WHERE data_type=? AND task_type=? AND question_id=?",
                (data_type, task_type, question_id)).fetchone()
            if row is None:
                raise GoalStoreError(f"Dataset-scoped goal store has no selected question {data_type}_{task_type}_alfworld_{question_id}; use a matching dataset artifact")
            selected = dict(zip(names, row[:-1]))
            if _sha(_json(selected).encode("utf-8")) != row[-1]:
                raise GoalStoreError("Saved dataset row failed integrity validation")
            root = _path(games_root, "runtime games root")
            key = Path(selected["game_key"])
            game = (root / key).resolve()
            if key.is_absolute() or ".." in key.parts or not game.is_relative_to(root):
                raise GoalStoreError("Saved dataset game escapes the runtime games root")
            game, trajectory, game_hash, traj_hash = _source_hashes(game)
            sources = connection.execute("SELECT game_sha256,traj_sha256 FROM goals WHERE game_key=?",
                                         (selected["game_key"],)).fetchone()
            if sources != (game_hash, traj_hash):
                raise GoalStoreError(f"Selected dataset game's sources changed or are missing: {game}. {_REBUILD}")
            return game
    except (sqlite3.Error, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise GoalStoreError(f"Cannot read saved dataset selection {path}: {error}. {_REBUILD}") from error


def read_goal_spec(store_path, gamefile):
    """Read a saved specification; never parse or convert PDDL at training time.

    Source bytes are hashed to reject stale data. Lookup is independent of the
    machine's absolute data path and a dataset's question_id/traversal order.
    Only the requested row is loaded, so large stores are not copied into every
    rollout worker. SQLite is opened read-only and no runtime files are written.
    """
    path = _path(store_path, "ALFWORLD_TRACE_GOALS / saved goal store")
    if not path.is_file():
        raise GoalStoreError(f"Saved goal store does not exist: {path}. {_REBUILD}")
    game, trajectory, game_hash, traj_hash = _source_hashes(gamefile)
    uri = "file:" + quote(str(path), safe="/") + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            metadata = _store_metadata(connection, path)
            row = connection.execute(
                "SELECT game_key, spec_sha256, spec_json FROM goals WHERE game_sha256=? AND traj_sha256=? ORDER BY game_key LIMIT 1",
                (game_hash, traj_hash)).fetchone()
            if row is None:
                raise GoalStoreError(f"No saved goal matches actual game and metadata: {game}. Sources changed or this game was not converted. {_REBUILD}")
            entry, payload_hash, payload = row
            if not isinstance(payload, str) or _sha(payload.encode("utf-8")) != payload_hash:
                raise GoalStoreError(f"Saved goal payload failed integrity validation: {entry}. {_REBUILD}")
            spec = json.loads(payload)
            if not _valid_spec(spec, metadata["template_version"], game_hash, traj_hash):
                raise GoalStoreError(f"Saved goal specification is invalid or incompatible: {entry}. {_REBUILD}")
    except (sqlite3.Error, UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise GoalStoreError(f"Cannot read saved goal store {path}: {error}. {_REBUILD}") from error
    # Rebind paths only; the English response, rules, facts and source hashes
    # are read directly from the saved artifact, never regenerated.
    spec["goal_store_source_gamefile"] = spec.get("gamefile")
    spec["gamefile"] = str(game)
    spec["goal_source"] = str(trajectory)
    spec["traj_data_path"] = str(trajectory)
    spec.update({"goal_store_path": str(path), "goal_store_id": metadata["goal_store_id"],
                 "goal_store_entry": entry, "goal_store_spec_sha256": payload_hash,
                 "goal_store_selection_mode": metadata.get("selection_mode", "all_games"),
                 "goal_store_dataset_sha256": metadata.get("dataset_sha256"),
                 "goal_store_selection_sha256": metadata.get("selection_sha256"),
                 "goal_store_dataset_rows": metadata.get("dataset_rows")})
    return spec


def _duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_printer(every=25, stream=None):
    """Flush human-readable progress to stderr; stdout remains JSON only."""
    stream = sys.stderr if stream is None else stream
    last_elapsed = -5.0

    def print_progress(event):
        nonlocal last_elapsed
        phase, processed, total = event["phase"], event["processed"], event["total"]
        elapsed = event["elapsed_seconds"]
        if phase == "scanning":
            text = ("Resolving Parquet rows using the official game catalogue..."
                    if event.get("selection_mode") == "dataset" else "Finding game.tw-pddl files...")
            print("[scan] " + text, file=stream, flush=True)
            return
        if (phase == "progress" and processed not in (1, total)
                and processed % every and elapsed - last_elapsed < 5.0):
            return
        last_elapsed = elapsed
        skipped = event["skipped_unsupported"] + event["skipped_unsolvable"]
        percent = 100.0 * processed / total if total else 0.0
        eta = _duration(elapsed * (total - processed) / processed) if processed and total else "unknown"
        prefix = {"converting": "convert", "progress": "convert", "saving": "saving", "complete": "complete"}[phase]
        line = (f"[{prefix}] {processed}/{total} ({percent:.1f}%) "
                f"converted={event['converted']} skipped={skipped} "
                f"elapsed={_duration(elapsed)} ETA={eta}")
        if event["current_game"]:
            line += " game=" + event["current_game"]
        artifact = "training Parquet" if event.get("artifact_type") == "parquet" else "goal store"
        if phase == "saving":
            line += f" | Writing and publishing the {artifact}..."
        elif phase == "complete":
            line += f" | {artifact[0].upper() + artifact[1:]} saved successfully."
        print(line, file=stream, flush=True)

    return print_progress


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parquet_tools():
    path = Path(__file__).with_name("alfworld_parquet.py")
    spec = importlib.util.spec_from_file_location("_alfworld_prepared_parquet", path)
    if spec is None or spec.loader is None:
        raise GoalStoreError(f"Cannot import prepared Parquet tools: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Convert supported games offline into one persisted SQLite artifact")
    build.add_argument("--games-root", required=True)
    scope = build.add_mutually_exclusive_group(required=True)
    scope.add_argument("--dataset", help="Convert only games selected by this ALFWorld Parquet")
    scope.add_argument("--all-games", action="store_true", help="Explicitly convert the whole games directory")
    build.add_argument("--alfworld-root", help="Official ALFWorld source directory containing its Python package")
    build.add_argument("--expected-rows", type=_positive_int, help="Require this exact number of dataset rows")
    build.add_argument("--reuse-store", help="Copy saved GT from an existing complete SQLite file without reconversion")
    build.add_argument("--output", required=True)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--progress-every", type=_positive_int, default=25,
                       help="Report every N files (default: 25); first, last and timed updates always print")
    build.add_argument("--quiet", action="store_true", help="Hide progress; keep the final JSON summary")
    prepare = commands.add_parser("prepare", help="Fill dataset.result with selected-game GT and embed scoring rules")
    prepare.add_argument("--dataset", required=True, help="Original ALFWorld training Parquet")
    prepare.add_argument("--games-root", required=True)
    prepare.add_argument("--output", required=True, help="New training Parquet with filled result and goal metadata")
    prepare.add_argument("--alfworld-root", help="Official ALFWorld source directory")
    prepare.add_argument("--expected-rows", type=_positive_int, default=3305)
    prepare.add_argument("--reuse-store", help="Reuse an existing complete SQLite without converting PDDL again")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--progress-every", type=_positive_int, default=25)
    prepare.add_argument("--quiet", action="store_true")
    show = commands.add_parser("show", help="Inspect/export a saved goal without reconversion")
    show.add_argument("--store", required=True)
    show.add_argument("--gamefile", required=True)
    show.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "build":
        progress = None if args.quiet else _progress_printer(args.progress_every)
        print(json.dumps(build_goal_store(args.games_root, args.output, overwrite=args.overwrite,
                                         progress=progress, dataset=args.dataset,
                                         alfworld_root=args.alfworld_root, expected_rows=args.expected_rows,
                                         reuse_store=args.reuse_store), ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        progress = None if args.quiet else _progress_printer(args.progress_every)
        try:
            summary = _parquet_tools().prepare_dataset_parquet(
                args.dataset, args.games_root, args.output, expected_rows=args.expected_rows,
                alfworld_root=args.alfworld_root, overwrite=args.overwrite, progress=progress,
                reuse_store=args.reuse_store)
        except ValueError as error:
            raise GoalStoreError(str(error)) from error
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        content = json.dumps(read_goal_spec(args.store, args.gamefile), ensure_ascii=False, indent=2) + "\n"
        if args.output:
            path = Path(args.output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        else:
            print(content, end="")


if __name__ == "__main__":
    try:
        main()
    except GoalStoreError as error:
        raise SystemExit(str(error))
