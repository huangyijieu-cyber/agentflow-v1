"""Prepared Parquet carries result GT and saved rules without a runtime DB."""
import errno
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import test_alfworld_goal_conversion as fixtures


MODULE = Path(__file__).resolve().parents[1] / "agentflow/trace/alfworld_parquet.py"
SPEC = importlib.util.spec_from_file_location("standalone_alfworld_parquet_tests", MODULE)
parquet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parquet)


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
        return parquet._store_tools().build_goal_store(self.games, self.store, dataset=self.input,
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
        with mock.patch.object(parquet._store_tools(), "_converter", side_effect=AssertionError("conversion forbidden")):
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
m._store_tools()._converter=lambda: (_ for _ in ()).throw(AssertionError('runtime conversion forbidden'))
task=json.load(open(sys.argv[2]))
path=m.parquet_task_gamefile(task,sys.argv[3]);result=m.read_parquet_task_goal(task,path)
assert result['goal_response']==task['result']
assert result['goal_source_type']=='parquet.result'
assert not any(n.startswith('pyarrow') for n in sys.modules)
print('result-only-runtime-ok')
"""
        result = subprocess.run([sys.executable, "-c", code, str(MODULE), str(task_path), str(self.games)],
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
        tools = parquet._store_tools()
        tools.build_goal_store(self.games, self.store)
        before = self.store.read_bytes()
        with mock.patch.object(tools, "_converter", side_effect=AssertionError("reconversion forbidden")):
            self.prepare(reuse_store=self.store)
        self.assertEqual(self.store.read_bytes(), before)
        self.assertTrue(all(row["result"] for row in self.read_rows()))

    def test_unsupported_hardlinks_publish_and_progress_completes_after_parquet(self):
        tools = parquet._store_tools()
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
        self.assertTrue(parquet._store_tools().read_goal_spec(cached[0], self.clean)["goal_response"])

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
