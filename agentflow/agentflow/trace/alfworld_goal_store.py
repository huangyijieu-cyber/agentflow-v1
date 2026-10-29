"""Offline ALFWorld goal conversion and read-only, portable training artifacts.

Only ``build_goal_store`` imports the PDDL converter. Training loads indexed
saved specifications by the actual game's source hashes and never translates
or parses a domain/goal. The SQLite file can be shared by independent workers.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import tempfile
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


def build_goal_store(games_root, output, *, overwrite=False):
    """Convert all supported games once and atomically publish one SQLite file.

    The root can be the official json_2.1.1 folder or a particular split. The
    official TextWorld exclusions (movable/Sliced and unsolvable games) apply;
    every other conversion error aborts the build and preserves any old file.
    """
    root = _path(games_root, "games root")
    destination = _path(output, "goal store output")
    if not root.is_dir():
        raise GoalStoreError(f"Games root is not a directory: {root}")
    if destination.exists() and not overwrite:
        raise GoalStoreError(f"Goal store already exists: {destination}; use --overwrite to rebuild explicitly")
    destination.parent.mkdir(parents=True, exist_ok=True)
    converter = _converter()
    if converter.TEMPLATE_VERSION not in SUPPORTED_TEMPLATE_VERSIONS:
        raise GoalStoreError("Offline converter template is incompatible with this goal store reader")
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + ".building-", suffix=".sqlite", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    connection = None
    records = []
    skipped_unsupported = skipped_unsolvable = 0
    try:
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
        for game in sorted(root.rglob("game.tw-pddl")):
            key = game.relative_to(root).as_posix()
            if "movable" in key or "Sliced" in key:
                skipped_unsupported += 1
                continue
            try:
                raw_game = json.loads(game.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise GoalStoreError(f"Cannot read game {game}: {error}") from error
            if not isinstance(raw_game, dict):
                raise GoalStoreError(f"Game must be a JSON object: {game}")
            if raw_game.get("solvable") is False:
                skipped_unsolvable += 1
                continue
            try:
                spec = converter.load_goal_spec(game)
            except (ValueError, OSError) as error:
                raise GoalStoreError(f"Offline conversion failed for {game}: {error}") from error
            payload = _json(spec)
            payload_hash = _sha(payload.encode("utf-8"))
            sources = spec["source_sha256"]
            connection.execute("INSERT INTO goals VALUES (?, ?, ?, ?, ?)",
                               (key, sources["gamefile"], sources["traj_data"], payload_hash, payload))
            records.append([key, spec["goal_spec_hash"], payload_hash])
        if not records:
            raise GoalStoreError("No supported solvable games were found; refusing to publish an empty goal store")
        store_id = _sha(_json({"format": STORE_FORMAT_VERSION,
                              "template": converter.TEMPLATE_VERSION, "records": records}).encode("utf-8"))
        summary = {"goal_store_path": str(destination), "goal_store_id": store_id,
                   "template_version": converter.TEMPLATE_VERSION, "count": len(records),
                   "converted": len(records), "skipped_unsupported": skipped_unsupported,
                   "skipped_unsolvable": skipped_unsolvable, "games_root": str(root)}
        metadata = {"store_format_version": STORE_FORMAT_VERSION, "status": "complete", **summary}
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", ((key, _json(value)) for key, value in metadata.items()))
        connection.commit()
        connection.close()
        connection = None
        # Never truncate an existing artifact or expose a partially built one.
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)  # atomic publish; refuses a racing writer
            temporary.unlink()
        return summary
    except (sqlite3.Error, OSError) as error:
        raise GoalStoreError(f"Cannot build goal store {destination}: {error}") from error
    finally:
        if connection is not None:
            connection.close()
        temporary.unlink(missing_ok=True)


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
            raw_metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            metadata = {key: json.loads(value) for key, value in raw_metadata.items()}
            if (metadata.get("store_format_version") != STORE_FORMAT_VERSION
                    or metadata.get("status") != "complete"
                    or metadata.get("template_version") not in SUPPORTED_TEMPLATE_VERSIONS
                    or type(metadata.get("count")) is not int or metadata["count"] < 1
                    or not _digest(metadata.get("goal_store_id"))):
                raise GoalStoreError(f"Saved goal store is incomplete or incompatible: {path}. {_REBUILD}")
            if connection.execute("SELECT COUNT(*) FROM goals").fetchone()[0] != metadata["count"]:
                raise GoalStoreError(f"Saved goal store is incomplete: goal count differs from the completed build: {path}. {_REBUILD}")
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
                 "goal_store_entry": entry, "goal_store_spec_sha256": payload_hash})
    return spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Convert supported games offline into one persisted SQLite artifact")
    build.add_argument("--games-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--overwrite", action="store_true")
    show = commands.add_parser("show", help="Inspect/export a saved goal without reconversion")
    show.add_argument("--store", required=True)
    show.add_argument("--gamefile", required=True)
    show.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build_goal_store(args.games_root, args.output, overwrite=args.overwrite), ensure_ascii=False, indent=2))
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
