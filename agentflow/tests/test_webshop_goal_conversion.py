"""All goals below are explicit unittest fixtures, never recovered WebShop targets."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[2] / "train-roma" / "prepare_webshop_gt.py"
MODULE = Path(__file__).resolve().parents[1] / "agentflow" / "trace" / "webshop_goal.py"
SPEC = importlib.util.spec_from_file_location("webshop_goal", MODULE)
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def fixture_goal(index=0):
    return {
        "asin": f"FIXTURE_ASIN_{index}", "name": f"FIXTURE REFERENCE TITLE {index}",
        "instruction_text": "UNITTEST FIXTURE: buy the requested fixture item.",
        "category": "fixture category", "query": "fixture query",
        "product_category": "fixture parent › fixture child",
        "attributes": ["fixture attribute one", "fixture attribute two"],
        "goal_options": {"size": "fixture medium", "color": "fixture blue"},
        "price_upper": 40.0, "weight": 1,
    }


def fixture_manifest(goals=None):
    goals = goals if goals is not None else [fixture_goal(i) for i in range(3)]
    return c.make_manifest(goals, {g["asin"]: 15.5 for g in goals},
                           {"fixture_only": True}, {"kind": "unittest-fixture"})


def fixture_table(questions=(2, 0, 1), result="", question_type=pa.int64(), sessions=True):
    extra_fields = [pa.field("ground_truth", pa.string()), pa.field("idx", pa.int32())]
    if sessions:
        extra_fields.append(pa.field("session", pa.int64()))
    extra_fields.append(pa.field("untouched", pa.struct([pa.field("tags", pa.list_(pa.string()))])))
    schema = pa.schema([
        pa.field("id", pa.int32(), metadata={b"original": b"id-metadata"}),
        pa.field("question", question_type, metadata={b"role": b"index"}),
        pa.field("result", pa.large_string(), metadata={b"keep": b"result-field"}),
        pa.field("source", pa.string()), pa.field("extra_info", pa.struct(extra_fields)),
        pa.field("unrelated", pa.binary()),
    ], metadata={b"fixture_only": b"not a real task corpus"})
    rows = []
    for i, question in enumerate(questions):
        extra = {"ground_truth": "", "idx": i, "untouched": {"tags": ["fixture", str(i)]}}
        if sessions:
            extra["session"] = question if type(question) is int else 0
        rows.append({"id": 100 - i * 7, "question": question, "result": result,
                     "source": "webshop", "extra_info": extra, "unrelated": bytes([i])})
    return pa.Table.from_pylist(rows, schema=schema)


class TemplateTests(unittest.TestCase):
    def test_stable_named_options_all_constraints_and_no_unique_identity(self):
        goal = fixture_goal()
        text, spec = c.render_goal(goal)
        reverse = copy.deepcopy(goal)
        reverse["goal_options"] = dict(reversed(list(goal["goal_options"].items())))
        self.assertEqual((text, spec), c.render_goal(reverse))
        self.assertEqual(c.sha256_json(goal), c.sha256_json(reverse))
        for attribute in goal["attributes"]:
            self.assertIn(json.dumps(attribute), text)
        self.assertIn('"color" = "fixture blue"; "size" = "fixture medium"', text)
        self.assertIn("The current candidate product matches", text)
        self.assertNotIn("purchase", text.lower())
        self.assertIn("product type required by the shopping instruction", text)
        self.assertIn("Its price satisfies the task's price limit.", text)
        self.assertNotIn("40", text)
        self.assertNotIn(goal["asin"], text)
        self.assertNotIn(goal["name"], text)
        for key in ("instruction_text", "name", "category", "query", "product_category", "price_upper"):
            self.assertEqual(spec[key], goal[key])
        self.assertFalse(spec["reference_identity_required"])
        self.assertEqual(spec["original_goal"], goal)
        self.assertEqual(c.sha256_json(spec["original_goal"]), c.sha256_json(goal))

    def test_list_options_have_values_only_and_empty_options_are_valid(self):
        goal = fixture_goal()
        goal["goal_options"] = ["fixture blue", "fixture medium"]
        text, spec = c.render_goal(goal)
        self.assertIn('options for this product are: "fixture blue"; "fixture medium"', text)
        self.assertNotIn('"color" =', text)
        self.assertEqual(spec["option_kind"], "values_only")
        for options in ({}, []):
            goal["goal_options"] = options
            self.assertNotIn("selected options", c.render_goal(goal)[0])

    def test_exact_sentinel_and_unreliable_prices(self):
        goal = fixture_goal()
        goal["price_upper"] = 1_000_000
        text, spec = c.render_goal(goal)
        self.assertNotIn("price limit", text)
        self.assertEqual(spec["price_upper"], 1_000_000)
        self.assertFalse(spec["price_constraint"]["enabled"])
        goal["price_upper"] = 1_000_001
        self.assertIn("price limit", c.render_goal(goal)[0])
        self.assertNotIn("1000001", c.render_goal(goal)[0])
        for value in (0, -1, "40", None, True, float("inf"), float("nan")):
            with self.subTest(price=value), self.assertRaises(c.ConversionError):
                goal["price_upper"] = value
                c.render_goal(goal)

    def test_invalid_attributes_options_and_instruction(self):
        for key, values in {
            "attributes": (None, [], [""], ["   "], [5], "attribute"),
            "goal_options": (None, {"color": 1}, {"": "x"}, [1]),
            "instruction_text": (None, "", "  "),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(c.ConversionError):
                    goal = fixture_goal()
                    goal[key] = value
                    c.render_goal(goal)

    def test_integer_indices_without_coercion(self):
        for value in (True, False, "0", 0.0, None):
            with self.subTest(value=value), self.assertRaises(c.ConversionError):
                c.integer_index(value, "question")
        self.assertEqual(c.integer_index(0, "question"), 0)
        import numpy as np
        self.assertEqual(c.integer_index(np.int64(2), "question"), 2)

    def test_native_type_context_requires_strings_and_permits_empty_strings(self):
        for key in ("name", "query", "product_category"):
            for value in (None, 7, []):
                with self.subTest(key=key, value=value):
                    goal = fixture_goal()
                    goal[key] = value
                    with self.assertRaisesRegex(c.ConversionError, "invalid_type_context"):
                        c.render_goal(goal)
                    converted, audit = c.convert_table(fixture_table((0,)), fixture_manifest([goal]))
                    self.assertEqual(audit["invalid_count"], 1)
                    self.assertEqual(converted["result"][0].as_py(), "")
            goal = fixture_goal()
            del goal[key]
            with self.assertRaisesRegex(c.ConversionError, "invalid_type_context"):
                c.render_goal(goal)
            goal[key] = ""
            self.assertTrue(c.render_goal(goal)[0])
        goal = fixture_goal()
        del goal["category"]
        self.assertTrue(c.render_goal(goal)[0])


class ManifestTests(unittest.TestCase):
    def test_catalog_prices_goal_references_and_sampling_weights_are_recoverable(self):
        for price in (-0.1, "15", None, True):
            with self.subTest(price=price), self.assertRaises(c.ConversionError):
                c.make_manifest([fixture_goal()], {"FIXTURE_ASIN_0": price}, {}, {})
        self.assertEqual(c.make_manifest([fixture_goal()], {"FIXTURE_ASIN_0": 0}, {}, {})["product_price_count"], 1)
        for asin in (None, "", "  ", "MISSING_FIXTURE_ASIN", 7):
            with self.subTest(asin=asin), self.assertRaises(c.ConversionError):
                goal = fixture_goal()
                goal["asin"] = asin
                c.make_manifest([goal], {"FIXTURE_ASIN_0": 15}, {}, {})
        for weight in (None, 0, -1, "1", True, float("inf"), float("nan")):
            with self.subTest(weight=weight), self.assertRaises(c.ConversionError):
                goal = fixture_goal()
                goal["weight"] = weight
                c.make_manifest([goal], {"FIXTURE_ASIN_0": 15}, {}, {})

    def test_content_and_table_hashes_reject_tampering(self):
        manifest = fixture_manifest()
        for table in ("goals", "product_prices"):
            altered = copy.deepcopy(manifest)
            if table == "goals":
                altered[table][0]["attributes"] = ["tampered fixture"]
            else:
                altered[table]["FIXTURE_ASIN_0"] = 99
            with self.assertRaisesRegex(c.ConversionError, "content hash"):
                c.validate_manifest(altered)
            altered["manifest_sha256"] = c.manifest_hash(altered)
            with self.assertRaisesRegex(c.ConversionError, "table hash"):
                c.validate_manifest(altered)
        changed_time = copy.deepcopy(manifest)
        changed_time["created_at_utc"] = "nonsemantic timestamp"
        self.assertEqual(c.manifest_hash(changed_time), c.manifest_hash(manifest))
        changed_time["configuration"]["seed"] = 42
        self.assertNotEqual(c.manifest_hash(changed_time), c.manifest_hash(manifest))

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"manifest_version":"one","manifest_version":"two"}')
            with self.assertRaisesRegex(c.ConversionError, "Duplicate JSON key"):
                c.read_manifest(path)


class TableTests(unittest.TestCase):
    def test_original_schema_nested_fields_row_order_and_hashes_survive(self):
        table, manifest = fixture_table(), fixture_manifest()
        converted, audit = c.convert_table(table, manifest)
        self.assertEqual(audit["valid_count"], 3)
        self.assertEqual(audit["invalid_count"], 0)
        self.assertEqual(converted.schema.metadata, table.schema.metadata)
        for name in table.column_names:
            self.assertEqual(converted.schema.field(name), table.schema.field(name))
            if name != "result":
                self.assertTrue(converted[name].equals(table[name]), name)
        self.assertEqual(converted["webshop_goal_sha256"][0].as_py(), c.sha256_json(manifest["goals"][2]))
        self.assertTrue(converted["webshop_task_key"][0].as_py().endswith(":goal:2"))
        repeated, _ = c.convert_table(converted, manifest)
        self.assertTrue(repeated.equals(converted))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.parquet"
            pq.write_table(converted, path)
            restored = pq.read_table(path)
            for name in table.column_names:
                if name != "result":
                    self.assertTrue(restored[name].equals(table[name]), name)
                self.assertEqual(restored.schema.field(name), table.schema.field(name))
            self.assertEqual(restored.schema.metadata, table.schema.metadata)

    def test_missing_session_is_allowed_but_present_invalid_session_is_not(self):
        _, audit = c.convert_table(fixture_table(sessions=False), fixture_manifest())
        self.assertEqual(audit["invalid_count"], 0)
        for session, message in ((9, "session_mismatch"), (None, "invalid_session")):
            table = fixture_table()
            rows = table.to_pylist()
            rows[0]["extra_info"]["session"] = session
            converted, audit = c.convert_table(pa.Table.from_pylist(rows, schema=table.schema), fixture_manifest())
            self.assertEqual(audit["invalid_count"], 1)
            self.assertIn(message, audit["invalid_rows"][0]["error"])
            self.assertEqual(converted["result"][0].as_py(), "")

    def test_missing_invalid_and_out_of_range_questions(self):
        for questions, arrow_type in (((None,), pa.int64()), ((3,), pa.int64()),
                                      ((-1,), pa.int64()), (("0",), pa.string()), ((True,), pa.bool_())):
            with self.subTest(questions=questions):
                converted, audit = c.convert_table(fixture_table(questions, question_type=arrow_type), fixture_manifest())
                self.assertEqual(audit["invalid_count"], 1)
                self.assertFalse(converted["webshop_gt_valid"][0].as_py())
                self.assertEqual(converted["result"][0].as_py(), "")
        with self.assertRaisesRegex(c.ConversionError, "Missing required columns"):
            c.convert_table(fixture_table().drop(["question"]), fixture_manifest())

    def test_nonempty_results_and_spoofed_version_are_rejected(self):
        with self.assertRaisesRegex(c.ConversionError, "nonempty result"):
            c.convert_table(fixture_table(result="existing fixture answer"), fixture_manifest())
        converted, _ = c.convert_table(fixture_table(), fixture_manifest())
        i = converted.schema.get_field_index("result")
        converted = converted.set_column(i, converted.schema.field(i),
                                         pa.array(["modified fixture result"] * 3, type=pa.large_string()))
        with self.assertRaisesRegex(c.ConversionError, "nonempty result"):
            c.convert_table(converted, fixture_manifest())


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.source = self.directory / "fixture_train.parquet"
        self.manifest_path = self.directory / "fixture_manifest.json.gz"
        pq.write_table(fixture_table(), self.source)
        c.publish_files([(self.manifest_path, c.json_writer(fixture_manifest(), True))])

    def run_cli(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return c.main(["--input", str(self.source), "--manifest-in", str(self.manifest_path), *map(str, extra)])

    def test_manifest_mode_cli_never_imports_native_and_preserves_input(self):
        before = self.source.read_bytes()
        bootstrap = '''import importlib.abc, runpy, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'gym','spacy','torch','web_agent_site'}:
            raise RuntimeError('Forbidden native import: ' + fullname)
sys.meta_path.insert(0, BlockNative())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        completed = subprocess.run([sys.executable, "-B", "-c", bootstrap, str(SCRIPT),
                                    "--input", str(self.source), "--manifest-in", str(self.manifest_path),
                                    "--expected-rows", "3"], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["valid_count"], 3)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertTrue((self.directory / "fixture_train_with_gt.manifest.json.gz").is_file())
        self.assertEqual(c.read_manifest(summary["manifest_output"])["manifest_sha256"],
                         c.read_manifest(self.manifest_path)["manifest_sha256"])
        self.assertLess(len(summary["samples"]), 4)

    def test_default_invalid_failure_and_explicit_allow_invalid(self):
        goals = [fixture_goal(i) for i in range(3)]
        goals[0]["attributes"] = []
        altered = self.directory / "invalid_fixture.json"
        c.publish_files([(altered, c.json_writer(fixture_manifest(goals)))])
        args = ["--input", str(self.source), "--manifest-in", str(altered)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(c.main(args), 2)
        failed = self.directory / "fixture_train_with_gt.report.json"
        self.assertEqual(json.loads(failed.read_text())["invalid_count"], 1)
        self.assertFalse((self.directory / "fixture_train_with_gt.parquet").exists())
        self.assertFalse((self.directory / "fixture_train_with_gt.manifest.json.gz").exists())
        output = self.directory / "allowed.parquet"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(c.main(args + ["--allow-invalid", "--output", str(output)]), 0)
        table = pq.read_table(output)
        self.assertEqual(table.num_rows, 3)
        self.assertEqual(table["webshop_gt_valid"].to_pylist(), [True, False, True])
        self.assertEqual(table["result"][1].as_py(), "")
        self.assertIn("original_goal", json.loads(table["webshop_goal_spec"][1].as_py()))

    def test_preflight_refuses_each_existing_output_and_input_alias(self):
        for kind in ("parquet", "manifest", "report", "input"):
            with self.subTest(kind=kind):
                output = self.directory / (kind + ".parquet")
                manifest = self.directory / (kind + ".manifest.json")
                report = self.directory / (kind + ".report.json")
                existing = {"parquet": output, "manifest": manifest, "report": report, "input": self.source}[kind]
                if kind != "input":
                    existing.write_bytes(b"preserve fixture bytes")
                before = existing.read_bytes()
                if kind == "input":
                    output = self.source
                self.assertEqual(self.run_cli("--output", output, "--manifest-out", manifest, "--report", report), 2)
                self.assertEqual(existing.read_bytes(), before)
        self.assertFalse(any(p.name.endswith(".tmp") for p in self.directory.iterdir()))

    def test_expected_row_count_and_tampered_manifest_write_failed_report_only(self):
        self.assertEqual(self.run_cli("--expected-rows", "6810"), 2)
        report = json.loads((self.directory / "fixture_train_with_gt.report.json").read_text())
        self.assertIn("Expected 6810 rows", report["error"])
        self.assertFalse((self.directory / "fixture_train_with_gt.parquet").exists())
        manifest = c.read_manifest(self.manifest_path)
        manifest["goals"][0]["price_upper"] = 3
        altered = self.directory / "tampered.json"
        altered.write_text(json.dumps(manifest))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = c.main(["--input", str(self.source), "--manifest-in", str(altered),
                             "--output", str(self.directory / "tampered_output.parquet")])
        self.assertEqual(status, 2)
        self.assertFalse((self.directory / "tampered_output.parquet").exists())

    def test_atomic_publish_rolls_back_and_cleans_staging_files(self):
        directory = self.directory / "atomic"
        directory.mkdir()
        one, two = directory / "one.json", directory / "two.json"
        real_link = c.os.link
        def fail_second(source, target):
            if Path(target) == two:
                raise OSError("fixture simulated publication failure")
            return real_link(source, target)
        with mock.patch.object(c.os, "link", side_effect=fail_second):
            with self.assertRaises(OSError):
                c.publish_files([(one, c.json_writer({"fixture": 1})), (two, c.json_writer({"fixture": 2}))])
        self.assertEqual(list(directory.iterdir()), [])

    def test_help_does_not_require_pyarrow_or_native_dependencies(self):
        bootstrap = '''import importlib.abc, runpy, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyarrow','numpy','gym','spacy','torch','web_agent_site'}:
            raise RuntimeError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        completed = subprocess.run([sys.executable, "-B", "-c", bootstrap, str(SCRIPT)],
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--manifest-in", completed.stdout)


if __name__ == "__main__":
    unittest.main()
