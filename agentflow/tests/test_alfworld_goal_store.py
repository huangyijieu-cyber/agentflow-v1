"""Persisted ALFWorld goals survive restarts; training never reconverts PDDL."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import hashlib
import importlib.util
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


if __name__ == "__main__":
    unittest.main()
