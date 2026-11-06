"""CPU-only fixtures for frozen-corpus restoration and exact task binding."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.trace import webshop_goal as goal_tools


def fixture_goal(index=0):
    return {"asin": f"FIXTURE{index}", "name": f"REFERENCE TITLE {index}",
            "instruction_text": "Find a fixture travel cup with the requested features.",
            "category": "reference category", "query": "reference query",
            "product_category": "reference product category", "attributes": ["insulated", "reusable"],
            "goal_options": {"color": "blue"}, "price_upper": 40.0, "weight": index + 1}


def fixture_row(manifest, index=0):
    goal = manifest["goals"][index]
    text, spec = goal_tools.render_goal(goal)
    corpus = manifest["manifest_sha256"]
    return {"id": 87, "question": index, "source": "webshop", "result": text,
            "extra_info": {"idx": 18, "session": index},
            "webshop_goal_spec": goal_tools.canonical_json(spec),
            "webshop_goal_sha256": goal_tools.sha256_json(goal),
            "webshop_task_key": f"{goal_tools.MANIFEST_VERSION}:{corpus}:goal:{index}",
            "webshop_gt_template": goal_tools.TEMPLATE, "webshop_gt_valid": True,
            "webshop_gt_error": "", "webshop_manifest_sha256": corpus}


def fixture_manifest():
    goals = [fixture_goal(i) for i in range(2)]
    return goal_tools.make_manifest(goals, {"FIXTURE0": 15.0, "FIXTURE1": 31.25}, {}, {})


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = fixture_manifest()
        self.row = fixture_row(self.manifest, 1)

    def bind(self, row=None, actual=None, question=None):
        return goal_tools.read_parquet_task_goal(
            self.row if row is None else row, self.manifest,
            self.manifest["goals"][1] if actual is None else actual, question)

    def test_bind_original_goal_and_stable_target_without_reference_identity(self):
        spec, metadata = self.bind(question=1)
        self.assertEqual(spec["goal_response"], self.row["result"])
        self.assertEqual(spec["original_goal"], self.manifest["goals"][1])
        self.assertEqual(spec["goal_spec_hash"], self.row["webshop_goal_sha256"])
        self.assertEqual(metadata["scoring_mode"], "webshop_state")
        self.assertEqual(metadata["template_version"], "webshop-candidate-state-v1")
        self.assertIs(type(metadata["question"]), int)
        self.assertTrue(metadata["goal_check_valid"])
        self.assertNotIn("REFERENCE TITLE", spec["goal_response"])
        self.assertNotIn("FIXTURE1", spec["goal_response"])
        self.assertNotIn("purchase", spec["goal_response"].lower())

    def test_reject_each_mismatched_binding_column(self):
        for key, changed in {
            "result": "Other GT", "webshop_gt_template": "other-template",
            "webshop_goal_sha256": "other-goal", "webshop_manifest_sha256": "other-manifest",
            "webshop_task_key": "other-task", "webshop_gt_valid": False,
            "webshop_gt_error": "conversion failed", "source": "alfworld",
        }.items():
            with self.subTest(key=key), self.assertRaises(goal_tools.ConversionError):
                self.bind({**self.row, key: changed})

    def test_original_goal_full_spec_and_actual_native_goal_are_checked(self):
        parsed = json.loads(self.row["webshop_goal_spec"])
        for path in (("original_goal", "asin"), ("original_goal", "name"), ("attributes",), ("reference_identity_required",)):
            altered = copy.deepcopy(parsed)
            target = altered
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = "tampered"
            with self.subTest(path=path), self.assertRaisesRegex(goal_tools.ConversionError, "goal spec"):
                self.bind({**self.row, "webshop_goal_spec": goal_tools.canonical_json(altered)})
        actual = copy.deepcopy(self.manifest["goals"][1])
        actual["instruction_text"] = "Assigned replacement instruction"
        with self.assertRaisesRegex(goal_tools.ConversionError, "Actual native goal"):
            self.bind(actual=actual)

    def test_question_and_session_indices_are_not_string_or_boolean_coercions(self):
        for value in (True, False, "1", 1.0, None):
            for key in ("question", "session"):
                row = copy.deepcopy(self.row)
                if key == "session":
                    row["extra_info"]["session"] = value
                else:
                    row[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(goal_tools.ConversionError):
                    self.bind(row)
        for value in (-1, 2):
            with self.assertRaises(goal_tools.ConversionError):
                self.bind({**self.row, "question": value})
        with self.assertRaises(goal_tools.ConversionError):
            self.bind(question=0)
        import numpy as np
        row = copy.deepcopy(self.row)
        row["question"] = np.int64(1)
        row["extra_info"]["session"] = np.int64(1)
        _, metadata = self.bind(row, question=np.int64(1))
        self.assertIs(type(metadata["question"]), int)

    def test_missing_optional_constraints_have_no_invented_sentences(self):
        self.manifest["goals"][1]["goal_options"] = {}
        self.manifest["goals"][1]["price_upper"] = goal_tools.PRICE_SENTINEL
        self.manifest = goal_tools.make_manifest(self.manifest["goals"], self.manifest["product_prices"], {}, {})
        self.row = fixture_row(self.manifest, 1)
        spec, _ = self.bind()
        self.assertNotIn("price", spec["goal_response"])
        self.assertNotIn("selected options", spec["goal_response"])

    def test_spec_json_duplicates_and_mutable_manifest_tampering_are_rejected(self):
        with self.assertRaisesRegex(goal_tools.ConversionError, "Duplicate"):
            self.bind({**self.row, "webshop_goal_spec": '{"x":1,"x":2}'})
        self.manifest["product_prices"]["FIXTURE1"] = 0
        with self.assertRaisesRegex(goal_tools.ConversionError, "content hash"):
            self.bind()


class FrozenRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.captured = self.base / "captured"
        package = self.captured / "web_agent_site"
        (package / "engine").mkdir(parents=True)
        (package / "__init__.py").write_text("# explicit native fixture\n")
        (package / "engine" / "engine.py").write_text("# fixture source v1\n")
        (self.captured / "products.json").write_text('[{"asin":"FIXTURE0"},{"asin":"FIXTURE1"}]')
        (self.captured / "attributes.json").write_text('{"FIXTURE0":{"attributes":["insulated"]}}')
        (self.captured / "human_attributes.json").write_text('{"FIXTURE1":["fixture instruction"]}')
        provenance = {"webshop_root": str(self.captured),
                      "source_files": [goal_tools.fingerprint(path) for path in sorted(package.rglob("*.py"))],
                      "data_files": {"products": goal_tools.fingerprint(self.captured / "products.json"),
                                     "attributes": goal_tools.fingerprint(self.captured / "attributes.json"),
                                     "human_attributes": goal_tools.fingerprint(self.captured / "human_attributes.json")}}
        config = {"num_products": 2, "human_goals_argument": None, "loaded_product_count": 2,
                  "filter_goals": None, "limit_goals": -1, "show_attrs": False}
        self.manifest = goal_tools.make_manifest([fixture_goal(0), fixture_goal(1)],
                                                {"FIXTURE0": 15.0, "FIXTURE1": 31.25}, config, provenance)
        self.root = self.base / "different-server-path"
        shutil.copytree(self.captured, self.root)
        products = [{"asin": "FIXTURE0"}, {"asin": "FIXTURE1"}]
        self.server = NS(all_products=products, product_item_dict={p["asin"]: p for p in products},
                         goals=[{"stale": True}], product_prices={"FIXTURE0": 999, "FIXTURE1": 888},
                         user_sessions={"constructor-reset": {"goal": "old"}}, assigned_instruction_text="old")
        engine = NS(__file__=str(self.root / "web_agent_site/engine/engine.py"),
                    DEFAULT_ATTR_PATH=str(self.root / "attributes.json"),
                    HUMAN_ATTR_PATH=str(self.root / "human_attributes.json"))
        self.modules = mock.patch.dict(sys.modules, {"web_agent_site.engine.engine": engine})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def install(self, **kwargs):
        options = {"num_products": 2, "human_goals": None, "file_path": self.root / "products.json"}
        options.update(kwargs)
        return goal_tools.install_frozen_manifest(self.server, self.manifest, self.root, **options)

    def test_restore_relocates_by_content_rebuilds_weights_and_discards_constructor_session(self):
        self.install()
        self.assertEqual(self.server.goals, self.manifest["goals"])
        self.assertEqual(self.server.product_prices, self.manifest["product_prices"])
        self.assertEqual(self.server.weights, [1, 2])
        self.assertEqual(self.server.cum_weights, [0, 1, 3])
        self.assertEqual(self.server.user_sessions, {})
        self.assertIsNone(self.server.assigned_instruction_text)
        self.server.goals[0]["attributes"].append("native session mutation")
        self.assertEqual(self.manifest["goals"][0]["attributes"], ["insulated", "reusable"])
        self.server.user_sessions["active"] = {"goal": self.server.goals[1]}
        with mock.patch.object(goal_tools, "fingerprint", side_effect=AssertionError("rehashing")):
            self.install()
        self.assertIn("active", self.server.user_sessions)

    def test_read_cache_is_recursive_readonly_and_does_not_require_native_imports(self):
        path = self.base / "corpus.json.gz"
        goal_tools.json_writer(self.manifest, True)(path)
        frozen = goal_tools.load_frozen_manifest(path)
        self.assertIs(frozen, goal_tools.load_frozen_manifest(path))
        with self.assertRaises(TypeError):
            frozen["product_prices"]["FIXTURE0"] = 3
        with self.assertRaises(TypeError):
            frozen["goals"][0]["attributes"][0] = "mutated"
        self.manifest = frozen
        self.install()
        self.assertIsInstance(self.server.goals, list)
        self.assertIsInstance(self.server.goals[0], dict)
        row = fixture_row(frozen, 1)
        with mock.patch.object(goal_tools, "manifest_hash", side_effect=AssertionError("full rehash")):
            goal_tools.read_parquet_task_goal(row, frozen, self.server.goals[1])

    def test_native_config_and_full_product_coverage_must_match(self):
        for kwargs in ({"num_products": 1}, {"human_goals": False}, {"num_products": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(goal_tools.ConversionError):
                self.install(**kwargs)
        self.server.all_products.append({"asin": "UNPRICED"})
        self.server.product_item_dict["UNPRICED"] = self.server.all_products[-1]
        with self.assertRaisesRegex(goal_tools.ConversionError, "complete native catalog"):
            self.install()
        with self.assertRaisesRegex(goal_tools.ConversionError, "arguments are required"):
            goal_tools.install_frozen_manifest(self.server, self.manifest, self.root)

    def test_changed_source_and_resource_contents_are_rejected_before_mutation(self):
        for relative, message in (("web_agent_site/engine/engine.py", "source"),
                                  ("attributes.json", "attributes"),
                                  ("human_attributes.json", "human_attributes"),
                                  ("products.json", "products")):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(path=relative), self.assertRaisesRegex(goal_tools.ConversionError, message + ".*hash mismatch"):
                self.install()
            self.assertEqual(self.server.goals, [{"stale": True}])
            path.write_bytes(original)
        (self.root / "web_agent_site/new_source.py").write_text("# extra source\n")
        with self.assertRaisesRegex(goal_tools.ConversionError, "source file coverage"):
            self.install()

    def test_missing_provenance_and_switching_corpus_are_rejected(self):
        other = fixture_manifest()
        with self.assertRaises(goal_tools.ConversionError):
            goal_tools.install_frozen_manifest(self.server, other, self.root, num_products=2, human_goals=None)
        self.install()
        with self.assertRaisesRegex(goal_tools.ConversionError, "different frozen manifest"):
            goal_tools.install_frozen_manifest(self.server, other, self.root, num_products=2, human_goals=None)


if __name__ == "__main__":
    unittest.main()
