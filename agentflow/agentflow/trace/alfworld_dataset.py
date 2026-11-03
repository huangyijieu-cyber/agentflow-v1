"""Resolve ALFWorld Parquet questions to the official ordered game catalogue.

Selection is separate from conversion. A question suffix is an index into the
same official catalogue used by the training environment, not a sorted path
list or the Parquet's usually one-based ``id`` column.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path
import re
import sys


class DatasetSelectionError(ValueError):
    """A dataset row cannot be matched unambiguously to an actual game."""


_SPLITS = {"train": "train", "val_seen": "eval_in_distribution",
           "val_unseen": "eval_out_of_distribution"}
_QUESTION = re.compile(r"(train|val_seen|val_unseen)_([a-z0-9]+)_alfworld_(\d+)")
_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "envs/alfworld/config"


def _path(value, description):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise DatasetSelectionError(f"Missing {description}")
    return Path(value).expanduser().resolve()


def _official_catalogue(root, data_type, task_type, alfworld_root):
    """Collect exactly as AlfWorldEnv._init_env, without creating a simulator."""
    config_file = _CONFIG_ROOT / f"config_tw_{task_type}.yaml"
    if not config_file.is_file():
        raise DatasetSelectionError(f"Unsupported ALFWorld task configuration: {task_type}")
    if alfworld_root is not None:
        source = _path(alfworld_root, "ALFWorld source directory")
        if not source.is_dir():
            raise DatasetSelectionError(f"ALFWorld source directory does not exist: {source}")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    try:
        import yaml
        official = importlib.import_module("alfworld.agents.environment.alfred_tw_env")
    except ImportError as error:
        raise DatasetSelectionError(
            "Dataset selection requires PyYAML and the existing ALFWorld/TextWorld "
            "Python environment. Install/activate that environment, or set --alfworld-root "
            "to the directory containing the alfworld package. No simulator is started."
        ) from error
    try:
        with config_file.open(encoding="utf-8") as reader:
            config = yaml.safe_load(reader)
        if config["env"]["type"] != "AlfredTWEnv":
            raise DatasetSelectionError("Dataset goal selection only supports AlfredTWEnv")
        # The root is the official json_2.1.1 directory. Only catalogue paths
        # are substituted; the constructor never starts an environment or reads
        # the logic/grammar needed for stepping it.
        prefix = "$ALFWORLD_DATA/json_2.1.1"
        for key, value in config["dataset"].items():
            if isinstance(value, str):
                config["dataset"][key] = value.replace(prefix, str(root))
        # The official constructor calls collect_game_files, including the
        # actual installed package's filtering, limits and unsorted traversal.
        # Keep its diagnostics out of the CLI's machine-readable JSON stdout.
        with redirect_stdout(sys.stderr):
            collector = official.AlfredTWEnv(config, train_eval=_SPLITS[data_type])
        return list(collector.game_files)
    except DatasetSelectionError:
        raise
    except Exception as error:
        raise DatasetSelectionError(
            f"Cannot collect official games for {data_type}/{task_type}: {error}"
        ) from error


def select_dataset_games(dataset, games_root, *, alfworld_root=None,
                         expected_rows=None, catalogue_loader=None):
    """Select only Parquet rows and retain their exact game bindings.

    ``catalogue_loader(root, data_type, task_type, alfworld_root)`` is an
    injectable catalogue source for tests. The default calls the installed
    official collector once for each split/task group. It never sorts that
    collector's result, loads a model, resets or steps an environment.
    """
    dataset_path = _path(dataset, "ALFWorld Parquet dataset")
    root = _path(games_root, "games root")
    if not dataset_path.is_file():
        raise DatasetSelectionError(f"Dataset does not exist: {dataset_path}")
    if not root.is_dir():
        raise DatasetSelectionError(f"Games root is not a directory: {root}")
    if expected_rows is not None and (type(expected_rows) is not int or expected_rows < 1):
        raise DatasetSelectionError("expected_rows must be a positive integer")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise DatasetSelectionError("Reading --dataset requires pyarrow in the active Python environment") from error
    try:
        source = dataset_path.read_bytes()
        table = parquet.ParquetFile(io.BytesIO(source))
        count = table.metadata.num_rows
        if not count:
            raise DatasetSelectionError("Dataset is empty; no goals will be selected")
        if expected_rows is not None and count != expected_rows:
            raise DatasetSelectionError(f"Dataset has {count} rows; expected exactly {expected_rows}")
        if "question" not in table.schema_arrow.names:
            raise DatasetSelectionError("Dataset must contain a question column")
        questions = table.read(columns=["question"]).column("question").to_pylist()
    except DatasetSelectionError:
        raise
    except Exception as error:
        raise DatasetSelectionError(f"Cannot read Parquet dataset {dataset_path}: {error}") from error

    parsed, seen_questions = [], set()
    for index, question in enumerate(questions):
        match = _QUESTION.fullmatch(question) if isinstance(question, str) else None
        if match is None:
            raise DatasetSelectionError(f"Unsupported ALFWorld question at row {index}: {question!r}")
        data_type, task_type, number = match.groups()
        if not (_CONFIG_ROOT / f"config_tw_{task_type}.yaml").is_file():
            raise DatasetSelectionError(f"Unsupported ALFWorld task configuration at row {index}: {task_type}")
        if question in seen_questions:
            raise DatasetSelectionError(f"Duplicate question at row {index}: {question}")
        seen_questions.add(question)
        parsed.append((index, question, data_type, task_type, int(number)))

    loader = _official_catalogue if catalogue_loader is None else catalogue_loader
    catalogues, rows, games, selected = {}, [], [], set()
    for index, question, data_type, task_type, question_id in parsed:
        group = data_type, task_type
        if group not in catalogues:
            try:
                catalogues[group] = list(loader(root, data_type, task_type, alfworld_root))
            except DatasetSelectionError:
                raise
            except Exception as error:
                raise DatasetSelectionError(f"Cannot collect games for {data_type}/{task_type}: {error}") from error
        catalogue = catalogues[group]
        if question_id >= len(catalogue):
            raise DatasetSelectionError(
                f"Dataset row {index} ({question}) selects question_id {question_id}, "
                f"outside official catalogue size {len(catalogue)} for {data_type}/{task_type}"
            )
        game = _path(catalogue[question_id], f"game for dataset row {index}")
        try:
            key = game.relative_to(root).as_posix()
        except ValueError as error:
            raise DatasetSelectionError(f"Selected game is outside --games-root at row {index}: {game}") from error
        if game.name != "game.tw-pddl" or not game.is_file():
            raise DatasetSelectionError(f"Selected game is missing or invalid at row {index}: {game}")
        if not game.with_name("traj_data.json").is_file():
            raise DatasetSelectionError(f"Selected game metadata is missing at row {index}: {game}")
        if game in selected:
            raise DatasetSelectionError(f"Multiple dataset rows resolve to the same game: row {index}, {game}")
        selected.add(game)
        games.append(game)
        rows.append({"row_index": index, "question": question, "data_type": data_type,
                     "task_type": task_type, "question_id": question_id, "game_key": key})

    dataset_hash = hashlib.sha256(source).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({"dataset_sha256": dataset_hash, "rows": rows},
                                           ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"games": games, "rows": rows,
            "metadata": {"dataset_path": str(dataset_path), "dataset_sha256": dataset_hash,
                         "dataset_rows": count, "selected_games": len(games),
                         "selection_sha256": fingerprint, "selection_mode": "dataset"}}
