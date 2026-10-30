"""Resolve dataset questions without guessing the official catalogue order."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = pq = None


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "agentflow/agentflow/trace/alfworld_dataset.py"
SPEC = importlib.util.spec_from_file_location("alfworld_dataset_selection_test", PATH)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


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


if __name__ == "__main__":
    unittest.main()
