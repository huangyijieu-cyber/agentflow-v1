"""Persisted ALFWorld goals survive restarts; training never reconverts PDDL."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import hashlib
import errno
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import test_alfworld_goal_conversion as fixtures
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace as NS


MODULE = Path(__file__).resolve().parents[1] / "agentflow/trace/alfworld_goal_store.py"
SPEC = importlib.util.spec_from_file_location("standalone_alfworld_goal_store_test", MODULE)
store = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(store)


@contextmanager
def database(path):
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            yield connection


class GoalStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.games = self.root / "json_2.1.1"
        self.games.mkdir()
        self.output = self.root / "saved_goals.sqlite"
        self.fixture = fixtures.GoalTests()
        self.fixture.root = self.games

    def tearDown(self):
        self.temp.cleanup()

    def game(self, *args, **kwargs):
        return self.fixture.create(*args, **kwargs)

    def build(self):
        return store.build_goal_store(self.games, self.output)

    def test_offline_build_then_restart_reads_exact_saved_spec_without_converter(self):
        game = self.game()
        expected = fixtures.converter.load_goal_spec(game)
        summary = self.build()
        before = self.output.read_bytes()
        with mock.patch.object(store, "_converter", side_effect=AssertionError("runtime conversion forbidden")):
            actual = store.read_goal_spec(self.output, [str(game)])
        for key in ("goal_response", "goal_ast", "rules_context", "initial_facts", "goal_spec_hash", "source_sha256"):
            self.assertEqual(actual[key], expected[key])
        self.assertEqual(actual["goal_store_id"], summary["goal_store_id"])
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["converted"], 1)
        self.assertEqual(actual["goal_store_entry"], "0/game.tw-pddl")
        # A separate process has none of the builder's memory or module state.
        exported = self.root / "read_after_restart.json"
        result = subprocess.run([sys.executable, str(MODULE), "show", "--store", str(self.output),
                                 "--gamefile", str(game), "--output", str(exported)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(exported.read_text())["goal_response"], expected["goal_response"])
        self.assertEqual(self.output.read_bytes(), before)

    def test_all_six_families_persist_and_keep_distinct_entries(self):
        games = [self.game(task) for task in fixtures.GOALS]
        summary = self.build()
        self.assertEqual(summary["count"], 6)
        self.assertEqual({store.read_goal_spec(self.output, path)["task_type"] for path in games}, set(fixtures.GOALS))

    def test_lookup_is_portable_when_data_and_store_move(self):
        original = self.game()
        self.build()
        copied_root = self.root / "server_data"
        shutil.copytree(self.games, copied_root)
        moved_store = self.root / "server_goals.sqlite"
        shutil.copyfile(self.output, moved_store)
        game = copied_root / original.relative_to(self.games)
        spec = store.read_goal_spec(moved_store, game)
        self.assertEqual(spec["gamefile"], str(game.resolve()))
        self.assertEqual(spec["goal_source"], str(game.with_name("traj_data.json").resolve()))
        self.assertEqual(spec["goal_store_source_gamefile"], str(original.resolve()))
        self.assertEqual(spec["goal_store_path"], str(moved_store.resolve()))

    def test_missing_store_does_not_generate_or_create_any_file(self):
        game = self.game()
        with mock.patch.object(store, "_converter", side_effect=AssertionError("forbidden")):
            with self.assertRaisesRegex(store.GoalStoreError, "does not exist"):
                store.read_goal_spec(self.output, game)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(store.GoalStoreError, "Missing"):
            store.read_goal_spec("", game)

    def test_changed_game_or_metadata_is_rejected_instead_of_reconverted(self):
        game = self.game()
        self.build()
        for path in (game, game.with_name("traj_data.json")):
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(path=path.name), self.assertRaisesRegex(store.GoalStoreError, "Sources changed"):
                store.read_goal_spec(self.output, game)
            path.write_bytes(original)
        store.read_goal_spec(self.output, game)

    def test_unconverted_game_and_missing_sources_are_explicit_errors(self):
        self.game()
        self.build()
        missing = self.game("pick_cool_then_place_in_recep")
        with self.assertRaisesRegex(store.GoalStoreError, "not converted"):
            store.read_goal_spec(self.output, missing)
        missing.with_name("traj_data.json").unlink()
        with self.assertRaisesRegex(store.GoalStoreError, "source files"):
            store.read_goal_spec(self.output, missing)

    def test_official_exclusions_skip_unsupported_paths_and_unsolvable(self):
        self.game()
        sliced = self.games / "pick-AppleSliced" / "trial"
        sliced.mkdir(parents=True)
        (sliced / "game.tw-pddl").write_text("not parsed; official excluded variant")
        movable = self.games / "pick_and_place_with_movable_recep" / "trial"
        movable.mkdir(parents=True)
        (movable / "game.tw-pddl").write_text("not parsed; official excluded variant")
        unsolvable = self.games / "unsolvable" / "trial"
        unsolvable.mkdir(parents=True)
        (unsolvable / "game.tw-pddl").write_text(json.dumps({"solvable": False}))
        summary = self.build()
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["skipped_unsupported"], 2)
        self.assertEqual(summary["skipped_unsolvable"], 1)

    def test_empty_build_and_conversion_failure_never_publish_partial_file(self):
        with self.assertRaisesRegex(store.GoalStoreError, "empty"):
            self.build()
        self.assertFalse(self.output.exists())
        self.game()
        self.game(goal="(unsupported mug0)")
        with self.assertRaisesRegex(store.GoalStoreError, "Offline conversion failed"):
            self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("*.building-*")), [])

    def test_failed_rebuild_preserves_existing_store_byte_for_byte(self):
        game = self.game()
        original_summary = self.build()
        original = self.output.read_bytes()
        with self.assertRaisesRegex(store.GoalStoreError, "already exists"):
            self.build()
        self.game(goal="(unsupported mug0)")
        with self.assertRaisesRegex(store.GoalStoreError, "Offline conversion failed"):
            store.build_goal_store(self.games, self.output, overwrite=True)
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(store.read_goal_spec(self.output, game)["goal_store_id"], original_summary["goal_store_id"])

    def test_explicit_successful_rebuild_changes_id_when_content_changes(self):
        self.game()
        first = self.build()
        self.game("pick_heat_then_place_in_recep")
        second = store.build_goal_store(self.games, self.output, overwrite=True)
        self.assertEqual(second["count"], 2)
        self.assertNotEqual(second["goal_store_id"], first["goal_store_id"])

    def test_malformed_and_incompatible_store_fail_closed(self):
        game = self.game()
        self.output.write_text("not SQLite")
        with self.assertRaisesRegex(store.GoalStoreError, "Cannot read saved goal store"):
            store.read_goal_spec(self.output, game)
        self.output.unlink()
        self.build()
        for key, value in (("store_format_version", 999), ("status", "building"),
                           ("template_version", "unknown-template"), ("count", 0)):
            with database(self.output) as connection:
                original = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0]
                connection.execute("UPDATE metadata SET value=? WHERE key=?", (json.dumps(value), key))
            with self.subTest(key=key), self.assertRaisesRegex(store.GoalStoreError, "incompatible"):
                store.read_goal_spec(self.output, game)
            with database(self.output) as connection:
                connection.execute("UPDATE metadata SET value=? WHERE key=?", (original, key))

    def test_modified_saved_response_is_not_silently_accepted(self):
        game = self.game()
        self.build()
        with database(self.output) as connection:
            payload = json.loads(connection.execute("SELECT spec_json FROM goals").fetchone()[0])
            payload["goal_response"] = "An unrelated task is done."
            connection.execute("UPDATE goals SET spec_json=?", (json.dumps(payload),))
        with self.assertRaisesRegex(store.GoalStoreError, "integrity"):
            store.read_goal_spec(self.output, game)

    def test_malformed_saved_payload_with_valid_checksum_is_rejected(self):
        game = self.game()
        self.build()
        payload = json.dumps({"schema_version": 1, "goal_response": "not a complete saved specification"})
        with database(self.output) as connection:
            connection.execute("UPDATE goals SET spec_json=?, spec_sha256=?",
                               (payload, hashlib.sha256(payload.encode()).hexdigest()))
        with self.assertRaisesRegex(store.GoalStoreError, "invalid or incompatible"):
            store.read_goal_spec(self.output, game)

    def test_corrupt_runtime_fields_raise_uniform_errors_without_converter(self):
        game = self.game()
        self.build()
        with database(self.output) as connection:
            original = json.loads(connection.execute("SELECT spec_json FROM goals").fetchone()[0])
        for key, value in (("source_sha256", []), ("objects", []), ("types", []),
                           ("entity_aliases", None), ("initial_facts", {}),
                           ("goal_ast", ["or", None]), ("domain_actions", []),
                           ("answer_instruction", None), ("goal_spec_hash", "wrong")):
            payload = json.dumps({**original, key: value})
            with database(self.output) as connection:
                connection.execute("UPDATE goals SET spec_json=?, spec_sha256=?",
                                   (payload, hashlib.sha256(payload.encode()).hexdigest()))
            with self.subTest(key=key), self.assertRaisesRegex(store.GoalStoreError, "invalid or incompatible"):
                store.read_goal_spec(self.output, game)

    def test_removed_row_invalidates_completed_artifact(self):
        game = self.game()
        self.game("pick_heat_then_place_in_recep")
        self.build()
        with database(self.output) as connection:
            connection.execute("DELETE FROM goals WHERE game_key='1/game.tw-pddl'")
        with self.assertRaisesRegex(store.GoalStoreError, "incomplete"):
            store.read_goal_spec(self.output, game)

    def test_independent_read_only_workers_do_not_mutate_store_or_each_other(self):
        game = self.game()
        self.build()
        original = self.output.read_bytes()
        with ThreadPoolExecutor(max_workers=4) as executor:
            rows = list(executor.map(lambda _: store.read_goal_spec(self.output, game), range(20)))
        self.assertEqual(len({row["goal_store_spec_sha256"] for row in rows}), 1)
        rows[0]["initial_facts"].clear()
        self.assertTrue(rows[1]["initial_facts"])
        self.assertEqual(self.output.read_bytes(), original)

    def test_progress_counts_include_skipped_files_and_complete_only_after_publish(self):
        game = self.game()
        sliced = self.games / "AppleSliced"
        sliced.mkdir()
        (sliced / "game.tw-pddl").write_text("officially excluded")
        unsolvable = self.games / "unsolvable"
        unsolvable.mkdir()
        (unsolvable / "game.tw-pddl").write_text(json.dumps({"solvable": False}))
        events = []

        def observe(event):
            events.append(event)
            if event["phase"] == "complete":
                self.assertTrue(self.output.is_file())
                self.assertEqual(store.read_goal_spec(self.output, game)["task_type"],
                                 "pick_clean_then_place_in_recep")
            else:
                self.assertFalse(self.output.exists())

        summary = store.build_goal_store(self.games, self.output, progress=observe)
        self.assertEqual(events[0]["phase"], "scanning")
        self.assertIsNone(events[0]["total"])
        rows = [row for row in events if row["phase"] == "progress"]
        self.assertEqual([row["processed"] for row in rows], [1, 2, 3])
        for row in rows:
            self.assertEqual(row["total"], 3)
            self.assertEqual(row["processed"], row["converted"] + row["skipped_unsupported"] + row["skipped_unsolvable"])
            self.assertGreaterEqual(row["elapsed_seconds"], 0)
        self.assertEqual([row["phase"] for row in events[-2:]], ["saving", "complete"])
        self.assertEqual(summary["converted"], 1)
        self.assertEqual(events[-1]["skipped_unsupported"], 1)
        self.assertEqual(events[-1]["skipped_unsolvable"], 1)
        with database(self.output) as connection:
            keys = {row[0] for row in connection.execute("SELECT key FROM metadata")}
        self.assertNotIn("elapsed_seconds", keys)

    def test_failed_conversion_does_not_report_complete_or_replace_old_artifact(self):
        self.game()
        self.build()
        before = self.output.read_bytes()
        self.game(goal="(unsupported mug0)")
        events = []
        with self.assertRaisesRegex(store.GoalStoreError, "Offline conversion failed"):
            store.build_goal_store(self.games, self.output, overwrite=True, progress=events.append)
        self.assertNotIn("complete", [row["phase"] for row in events])
        self.assertEqual(self.output.read_bytes(), before)

    def test_cli_progress_uses_stderr_and_quiet_preserves_json_summary(self):
        self.game()
        self.game("pick_heat_then_place_in_recep")
        command = [sys.executable, str(MODULE), "build", "--all-games", "--games-root", str(self.games),
                   "--output", str(self.output), "--progress-every", "1"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["converted"], 2)
        for text in ("[scan]", "1/2 (50.0%)", "2/2 (100.0%)", "ETA=", "[saving]", "[complete]"):
            self.assertIn(text, result.stderr)
        command[command.index(str(self.output))] = str(self.root / "quiet.sqlite")
        quiet = subprocess.run(command + ["--quiet"], capture_output=True, text=True)
        self.assertEqual(quiet.returncode, 0, quiet.stderr)
        self.assertEqual(quiet.stderr, "")
        self.assertEqual(json.loads(quiet.stdout)["converted"], 2)

    def test_progress_printer_throttles_but_always_reports_last_file(self):
        stream = io.StringIO()
        reporter = store._progress_printer(every=2, stream=stream)
        for index in range(1, 6):
            reporter({"phase": "progress", "processed": index, "total": 5,
                      "converted": index, "skipped_unsupported": 0, "skipped_unsolvable": 0,
                      "elapsed_seconds": index * .1, "current_game": f"{index}/game.tw-pddl"})
        text = stream.getvalue()
        for index in (1, 2, 4, 5):
            self.assertIn(f"{index}/5", text)
        self.assertNotIn("3/5", text)
        invalid = subprocess.run([sys.executable, str(MODULE), "build", "--all-games", "--games-root", str(self.games),
                                  "--output", str(self.output), "--progress-every", "0"],
                                 capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("positive integer", invalid.stderr)
        self.assertFalse(self.output.exists())

    def dataset(self, questions):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is required for dataset selection tests")
        path = self.root / "train.parquet"
        pq.write_table(pa.table({"question": questions, "id": list(range(1, len(questions) + 1))}), path)
        return path

    def test_dataset_build_only_converts_selected_rows_and_freezes_actual_game(self):
        first = self.game()
        second = self.game("pick_heat_then_place_in_recep")
        third = self.game("pick_cool_then_place_in_recep")
        dataset = self.dataset(["train_all_alfworld_2", "train_all_alfworld_0"])
        catalogue = [third, first, second]  # actual collector order, deliberately nonlexical
        summary = store.build_goal_store(self.games, self.output, dataset=dataset, expected_rows=2,
                                         catalogue_loader=lambda *args: catalogue)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["dataset_rows"], 2)
        self.assertEqual(summary["selected_games"], 2)
        self.assertEqual(summary["selection_mode"], "dataset")
        self.assertEqual(store.read_dataset_gamefile(self.output, "train", "all", 2, self.games), second.resolve())
        self.assertEqual(store.read_dataset_gamefile(self.output, "train", "all", 0, self.games), third.resolve())
        with self.assertRaisesRegex(store.GoalStoreError, "no selected question"):
            store.read_dataset_gamefile(self.output, "train", "all", 1, self.games)
        with self.assertRaisesRegex(store.GoalStoreError, "not converted"):
            store.read_goal_spec(self.output, first)
        with database(self.output) as connection:
            rows = connection.execute("SELECT question,game_key FROM dataset_rows ORDER BY row_index").fetchall()
        self.assertEqual(rows, [("train_all_alfworld_2", "1/game.tw-pddl"),
                                ("train_all_alfworld_0", "2/game.tw-pddl")])

    def test_dataset_subset_can_reuse_completed_store_without_converter(self):
        first = self.game()
        second = self.game("pick_heat_then_place_in_recep")
        self.build()
        before = self.output.read_bytes()
        subset = self.root / "selected.sqlite"
        dataset = self.dataset(["train_all_alfworld_1"])
        with mock.patch.object(store, "_converter", side_effect=AssertionError("reconversion forbidden")):
            summary = store.build_goal_store(self.games, subset, dataset=dataset, expected_rows=1,
                                             catalogue_loader=lambda *args: [first, second], reuse_store=self.output)
        self.assertEqual(summary["reused"], 1)
        self.assertEqual(summary["dataset_rows"], 1)
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(store.read_goal_spec(subset, second)["goal_response"],
                         store.read_goal_spec(self.output, second)["goal_response"])
        moved = self.root / "moved_data"
        shutil.copytree(self.games, moved)
        self.assertEqual(store.read_dataset_gamefile(subset, "train", "all", 1, moved), (moved / "1/game.tw-pddl").resolve())

    def test_dataset_scope_rejects_wrong_count_and_tampered_row(self):
        game = self.game()
        dataset = self.dataset(["train_all_alfworld_0"])
        with self.assertRaisesRegex(store.GoalStoreError, "expected exactly 3305"):
            store.build_goal_store(self.games, self.output, dataset=dataset, expected_rows=3305,
                                   catalogue_loader=lambda *args: [game])
        self.assertFalse(self.output.exists())
        store.build_goal_store(self.games, self.output, dataset=dataset,
                               catalogue_loader=lambda *args: [game])
        with database(self.output) as connection:
            connection.execute("UPDATE dataset_rows SET question='train_all_alfworld_999'")
        with self.assertRaisesRegex(store.GoalStoreError, "row failed integrity"):
            store.read_dataset_gamefile(self.output, "train", "all", 0, self.games)

    def test_legacy_store_keeps_environment_catalogue_selection(self):
        self.game()
        self.build()
        self.assertIsNone(store.read_dataset_gamefile(self.output, "train", "all", 0, self.games))

    def test_hardlink_unsupported_mount_publishes_without_reconverting(self):
        game = self.game()
        with mock.patch.object(store.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "Operation not supported")):
            summary = self.build()
        self.assertEqual(summary["count"], 1)
        self.assertEqual(store.read_goal_spec(self.output, game)["goal_store_id"], summary["goal_store_id"])
        self.assertFalse(self.output.with_name(self.output.name + ".publish.lock").exists())
        self.assertEqual(list(self.root.glob("*.building-*")), [])

    def test_publish_failure_keeps_complete_temporary_sqlite_for_recovery(self):
        game = self.game()
        with mock.patch.object(store.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "Operation not supported")), \
             mock.patch.object(store.os, "replace", side_effect=OSError(errno.EIO, "rename unavailable")):
            with self.assertRaisesRegex(store.GoalStoreError, "Completed SQLite retained"):
                self.build()
        self.assertFalse(self.output.exists())
        temporary = list(self.root.glob("*.building-*.sqlite"))
        self.assertEqual(len(temporary), 1)
        self.assertTrue(store.read_goal_spec(temporary[0], game)["goal_response"])
        self.assertFalse(self.output.with_name(self.output.name + ".publish.lock").exists())

    def test_fallback_publish_never_replaces_racing_existing_target(self):
        temporary = self.root / "ready.sqlite"
        temporary.write_bytes(b"new")
        self.output.write_bytes(b"existing")
        with mock.patch.object(store.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "Operation not supported")):
            with self.assertRaises(FileExistsError):
                store._publish_goal_store(temporary, self.output, False)
        self.assertEqual(self.output.read_bytes(), b"existing")
        self.assertEqual(temporary.read_bytes(), b"new")


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


DATASET_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = DATASET_ROOT / "agentflow/agentflow/trace/alfworld_goal_store.py"
DATASET_SPEC = importlib.util.spec_from_file_location("alfworld_dataset_selection_test", DATASET_PATH)
selector = importlib.util.module_from_spec(DATASET_SPEC)
DATASET_SPEC.loader.exec_module(selector)


@unittest.skipIf(pa is None, "pyarrow is needed to create real Parquet fixtures")
class DatasetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "json_2.1.1"
        self.root.mkdir()
        self.dataset = self.base / "train.parquet"

    def tearDown(self):
        self.temp.cleanup()

    def game(self, name):
        path = self.root / name / "trial" / "game.tw-pddl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        path.with_name("traj_data.json").write_text("{}")
        return path

    def table(self, questions, *, ids=None):
        ids = list(range(1, len(questions) + 1)) if ids is None else ids
        pq.write_table(pa.table({"id": ids, "question": questions}), self.dataset)
        return self.dataset

    def select(self, questions, catalogue, **kwargs):
        self.table(questions)
        loader = mock.Mock(return_value=catalogue)
        result = selector.select_dataset_games(self.dataset, self.root,
                                               catalogue_loader=loader, **kwargs)
        return result, loader

    def test_preserves_nonlexical_official_order_and_ignores_one_based_id(self):
        first, second, extra = [self.game(name) for name in ("train/z-first", "train/a-second", "train/unselected")]
        result, loader = self.select(["train_all_alfworld_0", "train_all_alfworld_1"],
                                     [first, second, extra], expected_rows=2)
        self.assertEqual(result["games"], [first, second])
        self.assertEqual([row["question_id"] for row in result["rows"]], [0, 1])
        self.assertEqual([row["row_index"] for row in result["rows"]], [0, 1])
        self.assertEqual(result["rows"][0]["game_key"], "train/z-first/trial/game.tw-pddl")
        loader.assert_called_once_with(self.root, "train", "all", None)
        self.assertNotIn(extra, result["games"])
        self.assertEqual(result["metadata"]["dataset_rows"], 2)
        self.assertEqual(result["metadata"]["selected_games"], 2)
        self.assertEqual(result["metadata"]["selection_mode"], "dataset")

    def test_collects_each_split_and_task_once_and_keeps_dataset_row_order(self):
        train0, train1, seen = [self.game(name) for name in ("train/t0", "train/t1", "valid_seen/seen")]
        self.table(["train_all_alfworld_1", "val_seen_look_alfworld_0", "train_all_alfworld_0"])
        loader = mock.Mock(side_effect=lambda root, split, task, source:
                           [seen] if split == "val_seen" else [train0, train1])
        result = selector.select_dataset_games(self.dataset, self.root, alfworld_root="source",
                                               catalogue_loader=loader)
        self.assertEqual(result["games"], [train1, seen, train0])
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(loader.call_args_list[1].args, (self.root, "val_seen", "look", "source"))

    def test_only_selected_train_games_are_returned_without_scanning_other_splits(self):
        train = self.game("train/selected")
        self.game("valid_unseen/not-selected")
        self.table(["train_all_alfworld_0"])
        with mock.patch.object(Path, "rglob", side_effect=AssertionError("selector must use official catalogue")):
            result = selector.select_dataset_games(self.dataset, self.root,
                                                   catalogue_loader=lambda *args: [train])
        self.assertEqual(result["games"], [train])

    def test_invalid_rows_fail_before_catalogue_collection(self):
        for questions, expected in (([], "empty"), ([None], "Unsupported ALFWorld question"),
                                    (["train_unknown_alfworld_0"], "Unsupported ALFWorld task"),
                                    (["train_all_alfworld_0", "train_all_alfworld_0"], "Duplicate question"),
                                    (["not-an-alfworld-question"], "Unsupported ALFWorld question")):
            self.table(questions)
            loader = mock.Mock(side_effect=AssertionError("must not collect"))
            with self.subTest(questions=questions), self.assertRaisesRegex(selector.DatasetSelectionError, expected):
                selector.select_dataset_games(self.dataset, self.root, catalogue_loader=loader)
            loader.assert_not_called()

    def test_wrong_expected_count_is_an_explicit_error(self):
        self.table(["train_all_alfworld_0"])
        loader = mock.Mock()
        with self.assertRaisesRegex(selector.DatasetSelectionError, "expected exactly 3305"):
            selector.select_dataset_games(self.dataset, self.root, expected_rows=3305,
                                          catalogue_loader=loader)
        loader.assert_not_called()
        for value in (True, 0, -1):
            with self.subTest(value=value), self.assertRaisesRegex(selector.DatasetSelectionError, "positive integer"):
                selector.select_dataset_games(self.dataset, self.root, expected_rows=value)

    def test_out_of_range_and_missing_game_or_metadata_fail(self):
        game = self.game("train/exists")
        self.table(["train_all_alfworld_1"])
        with self.assertRaisesRegex(selector.DatasetSelectionError, "outside official catalogue size 1"):
            selector.select_dataset_games(self.dataset, self.root, catalogue_loader=lambda *args: [game])
        self.table(["train_all_alfworld_0"])
        game.with_name("traj_data.json").unlink()
        with self.assertRaisesRegex(selector.DatasetSelectionError, "metadata is missing"):
            selector.select_dataset_games(self.dataset, self.root, catalogue_loader=lambda *args: [game])
        game.unlink()
        with self.assertRaisesRegex(selector.DatasetSelectionError, "game is missing"):
            selector.select_dataset_games(self.dataset, self.root, catalogue_loader=lambda *args: [game])

    def test_outside_root_and_multiple_rows_resolving_to_same_game_are_rejected(self):
        game = self.game("train/one")
        self.table(["train_all_alfworld_0", "train_all_alfworld_1"])
        with self.assertRaisesRegex(selector.DatasetSelectionError, "same game"):
            selector.select_dataset_games(self.dataset, self.root, catalogue_loader=lambda *args: [game, game])
        self.table(["train_all_alfworld_0"])
        outside = self.base / "elsewhere" / "game.tw-pddl"
        outside.parent.mkdir()
        outside.write_text("{}")
        with self.assertRaisesRegex(selector.DatasetSelectionError, "outside --games-root"):
            selector.select_dataset_games(self.dataset, self.root, catalogue_loader=lambda *args: [outside])

    def test_selection_fingerprint_is_stable_and_changes_with_dataset(self):
        first, second = self.game("train/first"), self.game("train/second")
        self.table(["train_all_alfworld_0", "train_all_alfworld_1"])
        loader = lambda *args: [first, second]
        one = selector.select_dataset_games(self.dataset, self.root, catalogue_loader=loader)
        two = selector.select_dataset_games(self.dataset, self.root, catalogue_loader=loader)
        self.assertEqual(one["metadata"], two["metadata"])
        self.assertEqual(len(one["metadata"]["dataset_sha256"]), 64)
        self.table(["train_all_alfworld_1", "train_all_alfworld_0"])
        changed = selector.select_dataset_games(self.dataset, self.root, catalogue_loader=loader)
        self.assertNotEqual(one["metadata"]["selection_sha256"], changed["metadata"]["selection_sha256"])

    def test_default_loader_uses_official_constructor_config_and_no_simulator(self):
        game = self.game("train/z-first")
        calls = []

        class Official:
            def __init__(self, config, train_eval):
                calls.append((config, train_eval))
                print("official catalogue diagnostic")
                self.game_files = [str(game)]

            def init_env(self, *args, **kwargs):
                raise AssertionError("must not create simulator")

            def reset(self, *args, **kwargs):
                raise AssertionError("must not reset simulator")

        self.table(["train_all_alfworld_0"])
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(selector.importlib, "import_module", return_value=NS(AlfredTWEnv=Official)) as imported, \
             redirect_stdout(out), redirect_stderr(err):
            result = selector.select_dataset_games(self.dataset, self.root)
        imported.assert_called_once_with("alfworld.agents.environment.alfred_tw_env")
        self.assertEqual(result["games"], [game])
        self.assertEqual(calls[0][1], "train")
        self.assertEqual(calls[0][0]["dataset"]["data_path"], str(self.root / "train"))
        self.assertEqual(calls[0][0]["env"]["task_types"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(out.getvalue(), "")
        self.assertIn("official catalogue diagnostic", err.getvalue())

    def test_missing_question_column_and_invalid_parquet_are_clear_errors(self):
        pq.write_table(pa.table({"id": [1]}), self.dataset)
        with self.assertRaisesRegex(selector.DatasetSelectionError, "question column"):
            selector.select_dataset_games(self.dataset, self.root)
        self.dataset.write_bytes(b"not Parquet")
        with self.assertRaisesRegex(selector.DatasetSelectionError, "Cannot read Parquet"):
            selector.select_dataset_games(self.dataset, self.root)


PARQUET_MODULE = Path(__file__).resolve().parents[1] / "agentflow/trace/alfworld_goal_store.py"
PARQUET_SPEC = importlib.util.spec_from_file_location("standalone_alfworld_parquet_tests", PARQUET_MODULE)
parquet = importlib.util.module_from_spec(PARQUET_SPEC)
PARQUET_SPEC.loader.exec_module(parquet)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is needed for offline preparation tests")
class PreparedParquetTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa, self.pq = pa, pq
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.games = self.root / "json_2.1.1"
        self.games.mkdir()
        self.fixture = fixtures.GoalTests()
        self.fixture.root = self.games
        self.clean = self.fixture.create()
        self.heat = self.fixture.create("pick_heat_then_place_in_recep")
        self.cool = self.fixture.create("pick_cool_then_place_in_recep")
        self.catalogue = [self.heat, self.clean, self.cool]
        self.input = self.root / "train.parquet"
        self.output = self.root / "train_with_gt.parquet"
        self.store = self.root / "selected.sqlite"
        schema = pa.schema([pa.field("id", pa.int64(), metadata={b"original": b"id"}),
                            ("question", pa.string()), ("result", pa.string()),
                            ("source", pa.string()), ("extra_info_ground_truth", pa.string()),
                            ("extra_info_idx", pa.int32())], metadata={b"original_schema": b"retained"})
        self.original = pa.Table.from_pydict({"id": [1, 2],
            "question": ["train_all_alfworld_1", "train_all_alfworld_0"],
            "result": ["", ""], "source": ["AlfWorld_train_all", "AlfWorld_train_all"],
            "extra_info_ground_truth": ["", ""], "extra_info_idx": [1, 2]}, schema=schema)
        pq.write_table(self.original, self.input)
        self.input_bytes = self.input.read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return parquet.build_goal_store(self.games, self.store, dataset=self.input,
            expected_rows=2, catalogue_loader=lambda *args: self.catalogue)

    def prepare(self, **kwargs):
        return parquet.prepare_dataset_parquet(self.input, self.games, self.output,
            expected_rows=2, catalogue_loader=lambda *args: self.catalogue, **kwargs)

    def read_rows(self):
        return self.pq.read_table(self.output).to_pylist()

    def test_result_filled_in_original_row_order_and_other_columns_untouched(self):
        summary = self.prepare()
        actual = self.pq.read_table(self.output)
        rows = actual.to_pylist()
        self.assertEqual(summary["dataset_rows"], 2)
        self.assertEqual(rows[0]["result"], fixtures.converter.load_goal_spec(self.clean)["goal_response"])
        self.assertEqual(rows[1]["result"], fixtures.converter.load_goal_spec(self.heat)["goal_response"])
        for column in self.original.column_names:
            if column != "result":
                self.assertEqual(actual.column(column), self.original.column(column))
                self.assertEqual(actual.schema.field(column), self.original.schema.field(column))
        self.assertEqual(actual.schema.metadata[b"original_schema"], b"retained")
        self.assertEqual(self.input.read_bytes(), self.input_bytes)
        self.assertEqual(actual.column_names, self.original.column_names + list(parquet._COLUMNS))
        self.assertEqual(json.loads(rows[0]["alfworld_goal_row"])["question_id"], 1)
        self.assertEqual(rows[0]["alfworld_game_key"], "0/game.tw-pddl")
        self.assertEqual(list(self.root.glob("*.goals-*.sqlite")), [])

    def test_export_reads_only_saved_specs_without_actual_game_files(self):
        self.build()
        shutil.rmtree(self.games)
        with mock.patch.object(parquet, "_converter", side_effect=AssertionError("conversion forbidden")):
            summary = parquet.write_goal_parquet(self.input, self.store, self.output)
        self.assertEqual(summary["count"], 2)
        self.assertTrue(all(row["result"] for row in self.read_rows()))

    def test_runtime_uses_result_without_sqlite_converter_or_pyarrow(self):
        self.build()
        parquet.write_goal_parquet(self.input, self.store, self.output)
        self.store.unlink()
        task = self.read_rows()[0]
        task_path = self.root / "task.json"
        task_path.write_text(json.dumps(task))
        code = """
import builtins,importlib.util,json,sys
original=builtins.__import__
def checked(name,*args,**kwargs):
    if name.startswith('pyarrow'):
        raise AssertionError('runtime must not import pyarrow')
    return original(name,*args,**kwargs)
builtins.__import__=checked
spec=importlib.util.spec_from_file_location('row_runtime',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m._converter=lambda: (_ for _ in ()).throw(AssertionError('runtime conversion forbidden'))
task=json.load(open(sys.argv[2]))
path=m.parquet_task_gamefile(task,sys.argv[3]);result=m.read_parquet_task_goal(task,path)
assert result['goal_response']==task['result']
assert result['goal_source_type']=='parquet.result'
assert not any(n.startswith('pyarrow') for n in sys.modules)
print('result-only-runtime-ok')
"""
        result = subprocess.run([sys.executable, "-c", code, str(PARQUET_MODULE), str(task_path), str(self.games)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("result-only-runtime-ok", result.stdout)

    def test_runtime_frozen_game_mapping_survives_moved_data(self):
        self.prepare()
        task = self.read_rows()[0]
        moved = self.root / "server_games"
        shutil.copytree(self.games, moved)
        expected = (moved / "0/game.tw-pddl").resolve()
        self.assertEqual(parquet.parquet_task_gamefile(task, moved), expected)
        spec = parquet.read_parquet_task_goal(task, expected)
        self.assertEqual(spec["gamefile"], str(expected))
        self.assertEqual(spec["goal_response"], task["result"])

    def test_empty_or_changed_result_and_corrupt_metadata_fail_closed(self):
        self.prepare()
        task = self.read_rows()[0]
        for updates, message in (({"result": ""}, "result is empty"),
                                 ({"result": "Another task is done."}, "result does not match"),
                                 ({"question": "train_all_alfworld_5"}, "mapping disagree"),
                                 ({"alfworld_goal_spec": "{}"}, "integrity validation"),
                                 ({"alfworld_game_key": "../game.tw-pddl"}, "mapping disagree")):
            with self.subTest(updates=updates), self.assertRaisesRegex(parquet.ParquetGoalError, message):
                parquet.parquet_task_gamefile({**task, **updates}, self.games)
        raw = self.clean.read_bytes()
        self.clean.write_bytes(raw + b"\n")
        with self.assertRaisesRegex(parquet.ParquetGoalError, "source hashes"):
            parquet.read_parquet_task_goal(task, self.clean)

    def test_wrong_input_dataset_never_publishes_output(self):
        self.build()
        self.pq.write_table(self.original.set_column(0, "id", self.pa.array([8, 9])), self.input)
        with self.assertRaisesRegex(parquet.ParquetGoalError, "exact source Parquet"):
            parquet.write_goal_parquet(self.input, self.store, self.output)
        self.assertFalse(self.output.exists())

    def test_reuse_store_avoids_second_conversion_and_keeps_input_store(self):
        tools = parquet
        tools.build_goal_store(self.games, self.store)
        before = self.store.read_bytes()
        with mock.patch.object(tools, "_converter", side_effect=AssertionError("reconversion forbidden")):
            self.prepare(reuse_store=self.store)
        self.assertEqual(self.store.read_bytes(), before)
        self.assertTrue(all(row["result"] for row in self.read_rows()))

    def test_unsupported_hardlinks_publish_and_progress_completes_after_parquet(self):
        tools = parquet
        events = []
        def progress(event):
            events.append(event)
            if event["phase"] == "complete":
                self.assertEqual(event["artifact_type"], "parquet")
                self.assertEqual(self.pq.read_table(self.output).num_rows, 2)
        with mock.patch.object(tools.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "unsupported")):
            self.prepare(progress=progress)
        self.assertEqual(len([event for event in events if event["phase"] == "complete"]), 1)
        self.assertEqual(events[-1]["phase"], "complete")

    def test_export_failure_keeps_complete_goal_cache_and_no_complete_event(self):
        events = []
        with mock.patch.object(parquet, "write_goal_parquet", side_effect=parquet.ParquetGoalError("export failed")):
            with self.assertRaisesRegex(parquet.ParquetGoalError, "--reuse-store"):
                self.prepare(progress=events.append)
        self.assertFalse(self.output.exists())
        self.assertNotIn("complete", [event["phase"] for event in events])
        cached = list(self.root.glob("*.goals-*.sqlite"))
        self.assertEqual(len(cached), 1)
        self.assertTrue(parquet.read_goal_spec(cached[0], self.clean)["goal_response"])

    def test_existing_output_is_preserved_without_explicit_overwrite(self):
        self.output.write_bytes(b"keep existing")
        with self.assertRaisesRegex(parquet.ParquetGoalError, "already exists"):
            self.prepare()
        self.assertEqual(self.output.read_bytes(), b"keep existing")
        self.assertEqual(list(self.root.glob("*.goals-*.sqlite")), [])

    def test_successful_parquet_is_not_reported_failed_if_cache_cleanup_is_denied(self):
        original_unlink = Path.unlink
        def cleanup(path, *args, **kwargs):
            if (self.output.exists() and ".goals-" in path.name and ".building-" not in path.name):
                raise OSError(errno.EACCES, "cache cleanup denied")
            return original_unlink(path, *args, **kwargs)
        events = []
        with mock.patch.object(Path, "unlink", new=cleanup):
            summary = self.prepare(progress=events.append)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(events[-1]["phase"], "complete")
        self.assertTrue(Path(summary["retained_cache"]).is_file())
        self.assertEqual(self.pq.read_table(self.output).num_rows, 2)


if __name__ == "__main__":
    unittest.main()
