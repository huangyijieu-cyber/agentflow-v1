#!/usr/bin/env python3
"""Freeze a native WebShop corpus and add task-specific candidate-state GT to Parquet.

Requires pyarrow for conversion. --manifest-in never imports the native environment.
A fresh native initialization does not reconstruct the targets of any historical run.
Manifest hashes cover all fields except created_at_utc and manifest_sha256. Goal hashes
cover only the complete original goal, serialized as sorted, compact UTF-8 JSON.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import math
import numbers
import os
from pathlib import Path
import random
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping
from functools import lru_cache
from itertools import accumulate
from types import MappingProxyType

TEMPLATE = "webshop-candidate-state-v1"
TEMPLATE_VERSION = TEMPLATE
SCORING_MODE = "webshop_state"
MANIFEST_VERSION = "webshop-frozen-corpus-v1"
SPEC_VERSION = "webshop-goal-spec-v1"
DEFAULT_SEED = 233
PRICE_SENTINEL = 1_000_000
FIELDS = {
    "webshop_goal_spec": "string",
    "webshop_goal_sha256": "string",
    "webshop_task_key": "string",
    "webshop_gt_template": "string",
    "webshop_gt_valid": "bool",
    "webshop_gt_error": "string",
    "webshop_manifest_sha256": "string",
}


class ConversionError(ValueError):
    pass


def json_value(value):
    """Normalize native/numpy scalar values without importing numpy."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        value = float(value)
        if not math.isfinite(value):
            raise ConversionError("JSON values must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ConversionError("JSON object keys must be strings")
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    raise ConversionError(f"Unsupported JSON value type: {type(value).__name__}")


def canonical_json(value):
    return json.dumps(json_value(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def manifest_hash(manifest):
    return sha256_json({k: v for k, v in manifest.items()
                        if k not in {"manifest_sha256", "created_at_utc"}})


def make_manifest(goals, product_prices, configuration, provenance):
    """Also usable to export a genuinely captured, already initialized server."""
    goals, product_prices = json_value(goals), json_value(product_prices)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "index_scope": "zero-based indices into this manifest's ordered goals only",
        "configuration": json_value(configuration),
        "provenance": json_value(provenance),
        "goals": goals,
        "product_prices": product_prices,
        "goal_count": len(goals),
        "product_price_count": len(product_prices),
        "goals_sha256": sha256_json(goals),
        "product_prices_sha256": sha256_json(product_prices),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    manifest["manifest_sha256"] = manifest_hash(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if not isinstance(manifest, Mapping) or manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ConversionError("Unsupported or missing manifest_version")
    if manifest.get("manifest_sha256") != manifest_hash(manifest):
        raise ConversionError("Manifest content hash mismatch")
    goals, prices = manifest.get("goals"), manifest.get("product_prices")
    if not isinstance(goals, (list, tuple)) or not goals or any(not isinstance(g, Mapping) for g in goals):
        raise ConversionError("Manifest goals must be a nonempty ordered list of objects")
    if not isinstance(prices, Mapping) or not prices:
        raise ConversionError("Manifest must contain the full product_prices object")
    if any(not isinstance(k, str) or not k or isinstance(v, bool)
           or not isinstance(v, numbers.Real) or not math.isfinite(v) or v < 0
           for k, v in prices.items()):
        raise ConversionError("Manifest product prices must have string keys and finite nonnegative numeric values")
    for index, goal in enumerate(goals):
        asin, weight = goal.get("asin"), goal.get("weight")
        if not isinstance(asin, str) or not asin.strip() or asin not in prices:
            raise ConversionError(f"Manifest goal {index} must reference an asin in product_prices")
        if isinstance(weight, bool) or not isinstance(weight, numbers.Real) or not math.isfinite(weight) or weight <= 0:
            raise ConversionError(f"Manifest goal {index} weight must be a finite positive number")
    for key, values, count in (("goals", goals, "goal_count"),
                               ("product_prices", prices, "product_price_count")):
        if type(manifest.get(count)) is not int or manifest[count] != len(values):
            raise ConversionError(f"Manifest {count} mismatch")
        if manifest.get(key + "_sha256") != sha256_json(values):
            raise ConversionError(f"Manifest {key} table hash mismatch")
    if not isinstance(manifest.get("configuration"), Mapping) or not isinstance(manifest.get("provenance"), Mapping):
        raise ConversionError("Manifest configuration and provenance are required")
    if manifest.get("index_scope") != "zero-based indices into this manifest's ordered goals only":
        raise ConversionError("Manifest index_scope is missing or unsupported")
    return manifest


def read_manifest(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    def reject_constant(value):
        raise ConversionError(f"Non-finite JSON constant: {value}")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ConversionError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with opener(path, "rt", encoding="utf-8") as handle:
        manifest = json.load(handle, parse_constant=reject_constant,
                             object_pairs_hook=unique_object)
    return validate_manifest(manifest)


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def export_native_manifest(args):
    """Use the unmodified SimServer path used by WebAgentTextEnv.__init__."""
    root = Path(args.webshop_root).expanduser().resolve(strict=True)
    if not (root / "web_agent_site" / "__init__.py").is_file():
        raise ConversionError("--webshop-root must contain web_agent_site/__init__.py")
    sys.path.insert(0, str(root))
    # Native imports may log progress; keep the CLI's stdout summary machine-readable.
    with contextlib.redirect_stdout(sys.stderr):
        native = importlib.import_module("web_agent_site.envs.web_agent_text_env")
        utils = importlib.import_module("web_agent_site.utils")
        engine = importlib.import_module("web_agent_site.engine.engine")
    module_paths = {}
    for name, module in list(sys.modules.items()):
        if name == "web_agent_site" or name.startswith("web_agent_site."):
            actual = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
            if actual is None or root not in actual.parents:
                raise ConversionError(f"Wrong WebShop installation imported for {name}: {actual}")
            module_paths[name] = str(actual)
    goal_mode = args.goal_mode or "env-default"
    human_goals = {"env-default": None, "human": True, "synthetic": False}[goal_mode]
    num_products = getattr(args, "num_products", utils.DEBUG_PROD_SIZE)
    file_path = Path(args.file_path or utils.DEFAULT_FILE_PATH).expanduser().resolve(strict=True)
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    provenance = {
        "kind": "fresh-native-initialization",
        "historical_run_reconstructed": False,
        "webshop_root": str(root),
        "module_paths": module_paths,
        "source_files": [fingerprint(p) for p in sorted((root / "web_agent_site").rglob("*.py"))],
        "data_files": {
            "products": fingerprint(file_path),
            "attributes": fingerprint(engine.DEFAULT_ATTR_PATH),
            "human_attributes": fingerprint(engine.HUMAN_ATTR_PATH),
        },
        "python_version": sys.version,
        "packages": {},
    }
    for package in ("gym", "numpy", "spacy", "torch", "pyserini", "thefuzz"):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = None
    random.seed(seed)
    native.np.random.seed(seed)
    with contextlib.redirect_stdout(sys.stderr):
        server = native.SimServer(
            "http://127.0.0.1:3000", str(file_path), filter_goals=None,
            limit_goals=-1, num_products=num_products, human_goals=human_goals,
            show_attrs=False,
        )
    if set(server.product_prices) != {p["asin"] for p in server.all_products}:
        raise ConversionError("Native product_prices does not cover the complete loaded catalog")
    configuration = {
        "goal_mode": goal_mode, "human_goals_argument": human_goals,
        "num_products": num_products, "file_path": str(file_path),
        "seed_before_server_initialization": seed, "seeded_rngs": ["random", "numpy"],
        "filter_goals": None, "limit_goals": -1, "show_attrs": False,
        "native_shuffle": "unchanged; see fingerprinted SimServer source",
        "loaded_product_count": len(server.all_products),
    }
    return make_manifest(server.goals, server.product_prices, configuration, provenance)


def render_goal(goal):
    """Assemble one fixed GT per task from its actual conditions, never from a turn.

    Native WebShop goals require attributes; missing or malformed attributes remain
    invalid. Empty option collections and the native no-price-limit sentinel omit
    their corresponding sentences instead of inventing conditions.
    """
    goal = json_value(goal)
    instruction = goal.get("instruction_text")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ConversionError("invalid_instruction: instruction_text must be a nonempty string")
    for key in ("name", "query", "product_category"):
        if not isinstance(goal.get(key), str):
            raise ConversionError(f"invalid_type_context: native reward requires {key} to be a string")
    attributes = goal.get("attributes")
    if not isinstance(attributes, list) or not attributes or any(
            not isinstance(v, str) or not v.strip() for v in attributes):
        raise ConversionError("invalid_attributes: require a nonempty list of nonempty strings")
    options = goal.get("goal_options")
    if isinstance(options, dict):
        if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
               for k, v in options.items()):
            raise ConversionError("invalid_options: option keys and values must be nonempty strings")
        option_text = "; ".join(f"{json.dumps(k, ensure_ascii=False)} = {json.dumps(options[k], ensure_ascii=False)}"
                                for k in sorted(options))
        option_kind = "named"
    elif isinstance(options, list):
        if any(not isinstance(v, str) or not v.strip() for v in options):
            raise ConversionError("invalid_options: option list values must be nonempty strings")
        option_text = "; ".join(json.dumps(v, ensure_ascii=False) for v in options)
        option_kind = "values_only"
    else:
        raise ConversionError("invalid_options: goal_options must be an object or list")
    price = goal.get("price_upper")
    if isinstance(price, bool) or not isinstance(price, numbers.Real) or not math.isfinite(price) or price <= 0:
        raise ConversionError("invalid_price: price_upper must be a finite positive number")
    constrained = price != PRICE_SENTINEL
    spec = {
        "spec_version": SPEC_VERSION,
        "original_goal": goal,
        **{key: goal.get(key) for key in ("instruction_text", "name", "category", "query", "product_category")},
        "attributes": attributes, "goal_options": options, "option_kind": option_kind,
        "reference_asin": goal.get("asin"), "reference_identity_required": False,
        "type_requirement": "the product type required by the shopping instruction",
        "price_upper": price,
        "price_constraint": ({"enabled": True, "operator": "<=", "currency": "USD", "upper": price}
                             if constrained else {"enabled": False,
                             "reason": "native_price_upper_1000000_sentinel_no_substantive_price_constraint",
                             "native_reward_still_checks_upper": True}),
    }
    sentences = [
        "The current candidate product matches the product type required by the shopping instruction.",
        "It satisfies the required attributes: "
        + "; ".join(json.dumps(v, ensure_ascii=False) for v in attributes) + ".",
    ]
    if constrained:
        sentences.append("Its price satisfies the task's price limit.")
    if option_text:
        sentences.append("The selected options for this product are: " + option_text + ".")
    return "\n".join(sentences), json_value(spec)


def integer_index(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ConversionError(f"invalid_{label}: expected an integer, without string coercion")
    return int(value)


def _readonly(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _readonly(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_readonly(item) for item in value)
    return value


@lru_cache(maxsize=8)
def _cached_manifest(path, size, modified_ns):
    return _readonly(read_manifest(path))


def load_frozen_manifest(path):
    """Read and validate a corpus once, exposing only immutable cached values.

    The cache is invalidated when the manifest's file identity changes. No
    native imports, RNG initialization, or product data are needed here.
    """
    resolved = Path(path).expanduser().resolve(strict=True)
    stat = resolved.stat()
    return _cached_manifest(str(resolved), stat.st_size, stat.st_mtime_ns)


_UNSET = object()


def _native_root(server, webshop_root):
    module = sys.modules.get(type(server).__module__)
    module_path = getattr(module, "__file__", None)
    discovered = None
    if module_path:
        for parent in Path(module_path).resolve().parents:
            if parent.name == "web_agent_site":
                discovered = parent.parent
                break
    if webshop_root is None:
        webshop_root = discovered or os.environ.get("WEBSHOP_ROOT")
    if webshop_root is None:
        raise ConversionError("Cannot locate native WebShop source; provide webshop_root or WEBSHOP_ROOT")
    root = Path(webshop_root).expanduser().resolve(strict=True)
    if not (root / "web_agent_site" / "__init__.py").is_file():
        raise ConversionError("webshop_root must contain web_agent_site/__init__.py")
    if discovered is not None and discovered != root:
        raise ConversionError("Configured WebShop root differs from the actually loaded native source")
    # Checking the loaded modules avoids verifying one tree while running another.
    for name, loaded in list(sys.modules.items()):
        if name == "web_agent_site" or name.startswith("web_agent_site."):
            source = getattr(loaded, "__file__", None)
            if source is not None and root not in Path(source).resolve().parents:
                raise ConversionError(f"Native module {name} is loaded outside webshop_root")
    return root


def _relocated_path(captured_path, captured_root, root):
    try:
        return root / Path(captured_path).relative_to(captured_root)
    except ValueError:
        return Path(captured_path).expanduser()


def _verify_fingerprint(actual, expected, label):
    if not isinstance(expected, Mapping) or not isinstance(expected.get("sha256"), str):
        raise ConversionError(f"Frozen manifest has no verifiable {label} fingerprint")
    try:
        observed = fingerprint(actual)
    except OSError as error:
        raise ConversionError(f"Cannot verify native {label}: {error}") from error
    if (observed["sha256"] != expected["sha256"]
            or observed["size_bytes"] != expected.get("size_bytes")):
        raise ConversionError(f"Native {label} content hash mismatch")


def _verify_native_files(server, manifest, webshop_root, file_path):
    root = _native_root(server, webshop_root)
    provenance = manifest["provenance"]
    old_root = provenance.get("webshop_root")
    sources = provenance.get("source_files")
    resources = provenance.get("data_files")
    if not isinstance(old_root, str) or not isinstance(sources, (list, tuple)) or not sources:
        raise ConversionError("Frozen manifest must fingerprint all native source files")
    if not isinstance(resources, Mapping) or set(resources) != {"products", "attributes", "human_attributes"}:
        raise ConversionError("Frozen manifest must fingerprint product and attribute resources")
    expected_sources = {}
    for entry in sources:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ConversionError("Invalid native source fingerprint")
        try:
            relative = Path(entry["path"]).relative_to(old_root)
        except ValueError as error:
            raise ConversionError("Native source fingerprint is outside the captured WebShop root") from error
        if not relative.parts or relative.parts[0] != "web_agent_site" or relative in expected_sources:
            raise ConversionError("Duplicate or unsupported native source fingerprint")
        expected_sources[relative] = entry
    actual_sources = {source.relative_to(root) for source in (root / "web_agent_site").rglob("*.py")}
    if actual_sources != set(expected_sources):
        raise ConversionError("Native source file coverage differs from the frozen manifest")
    for relative, expected in expected_sources.items():
        _verify_fingerprint(root / relative, expected, f"source {relative}")
    engine = sys.modules.get("web_agent_site.engine.engine")
    for key, constant in (("products", None), ("attributes", "DEFAULT_ATTR_PATH"),
                          ("human_attributes", "HUMAN_ATTR_PATH")):
        expected = resources[key]
        if not isinstance(expected, Mapping) or not isinstance(expected.get("path"), str):
            raise ConversionError(f"Invalid {key} resource fingerprint")
        actual = file_path if key == "products" else getattr(engine, constant, None)
        if actual is None:
            actual = _relocated_path(expected["path"], old_root, root)
        _verify_fingerprint(actual, expected, key)


def install_frozen_manifest(server, manifest, webshop_root=None, *, num_products=_UNSET,
                            human_goals=_UNSET, file_path=None):
    """Restore the ordered goals and exact reward prices before a formal reset.

    SimServer does not retain its constructor's catalog arguments, so the caller
    supplies the actual native env kwargs. Files are verified by content, allowing
    the captured corpus to move between servers. Repeated installation of the
    same corpus on the same server does not clear an active episode or hash the
    catalog again.
    """
    corpus = manifest.get("manifest_sha256") if isinstance(manifest, Mapping) else None
    installed = getattr(server, "_webshop_frozen_manifest_sha256", None)
    if installed is not None:
        if installed != corpus:
            raise ConversionError("A different frozen manifest is already installed on this server")
        return
    validate_manifest(manifest)
    configuration = manifest["configuration"]
    if num_products is _UNSET or human_goals is _UNSET:
        raise ConversionError("Actual native num_products and human_goals arguments are required")
    if num_products is not None:
        num_products = integer_index(num_products, "num_products")
        if num_products <= 0:
            raise ConversionError("Native num_products must be positive or None")
    if human_goals is not None and type(human_goals) is not bool:
        raise ConversionError("Native human_goals must be bool or None")
    if ("num_products" not in configuration or "human_goals_argument" not in configuration
            or canonical_json(num_products) != canonical_json(configuration["num_products"])
            or canonical_json(human_goals) != canonical_json(configuration["human_goals_argument"])):
        raise ConversionError("Native num_products/human_goals configuration mismatch")
    if configuration.get("filter_goals") is not None or configuration.get("limit_goals", -1) != -1:
        raise ConversionError("Frozen runtime requires the complete unfiltered ordered goal table")
    products = getattr(server, "all_products", None)
    items = getattr(server, "product_item_dict", None)
    if not isinstance(products, list) or not isinstance(items, Mapping):
        raise ConversionError("Native server has no complete product catalog")
    asins = [product.get("asin") for product in products if isinstance(product, Mapping)]
    if (len(asins) != len(products) or any(not isinstance(asin, str) or not asin for asin in asins)
            or len(set(asins)) != len(asins) or set(items) != set(asins)
            or set(manifest["product_prices"]) != set(asins)):
        raise ConversionError("Frozen product prices must cover the exact complete native catalog")
    if type(configuration.get("loaded_product_count")) is not int or configuration["loaded_product_count"] != len(asins):
        raise ConversionError("Native loaded_product_count mismatch")
    _verify_native_files(server, manifest, webshop_root, file_path)
    # json_value recursively copies the immutable cache into independently mutable
    # native tables; receive() is allowed to keep a goal reference in its session.
    server.goals = json_value(manifest["goals"])
    server.product_prices = json_value(manifest["product_prices"])
    server.weights = [goal["weight"] for goal in server.goals]
    server.cum_weights = [0] + list(accumulate(server.weights))
    server.user_sessions = {}
    server.assigned_instruction_text = None
    server._webshop_frozen_manifest_sha256 = corpus


def _spec_json(text):
    if not isinstance(text, str) or not text:
        raise ConversionError("Missing webshop_goal_spec JSON string")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ConversionError(f"Duplicate goal-spec JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise ConversionError(f"Non-finite goal-spec JSON constant: {value}")
    try:
        spec = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (TypeError, json.JSONDecodeError) as error:
        raise ConversionError(f"Invalid webshop_goal_spec JSON: {error}") from error
    if not isinstance(spec, dict):
        raise ConversionError("webshop_goal_spec must contain an object")
    return spec


def read_parquet_task_goal(task, manifest, actual_goal, question=None):
    """Bind one converted row to its exact frozen goal and actual native session.

    This is only a target-binding check. It never evaluates the product, computes
    native reward, or declares that a shopping goal has been attained.
    """
    if not isinstance(task, Mapping):
        raise ConversionError("WebShop trace task must be a complete Parquet row object")
    # Cached manifests are recursively immutable and were already validated by
    # load_frozen_manifest. Ad-hoc mutable inputs must still prove their hash.
    if not isinstance(manifest, MappingProxyType):
        validate_manifest(manifest)
    index = integer_index(task.get("question"), "question")
    if question is not None and integer_index(question, "question") != index:
        raise ConversionError("Runtime question does not match the Parquet question")
    goals = manifest.get("goals")
    if not isinstance(goals, (list, tuple)) or not 0 <= index < len(goals):
        raise ConversionError("Runtime question is outside the frozen goal table")
    if task.get("source") != "webshop":
        raise ConversionError("WebShop trace task source must equal 'webshop'")
    extra = task.get("extra_info")
    if extra is not None and not isinstance(extra, Mapping):
        raise ConversionError("WebShop trace extra_info must be an object or null")
    if isinstance(extra, Mapping) and "session" in extra and integer_index(extra["session"], "session") != index:
        raise ConversionError("WebShop trace session does not match question")
    if task.get("webshop_gt_valid") is not True or task.get("webshop_gt_error") != "":
        raise ConversionError("WebShop trace requires a valid, error-free converted GT row")
    goal = json_value(goals[index])
    if not isinstance(actual_goal, Mapping) or canonical_json(actual_goal) != canonical_json(goal):
        raise ConversionError("Actual native goal does not match the frozen question goal")
    response, expected_spec = render_goal(goal)
    corpus = manifest.get("manifest_sha256")
    goal_hash = sha256_json(goal)
    key = f"{MANIFEST_VERSION}:{corpus}:goal:{index}"
    expected_fields = {"webshop_gt_template": TEMPLATE, "webshop_manifest_sha256": corpus,
                       "webshop_goal_sha256": goal_hash, "webshop_task_key": key, "result": response}
    for field, expected in expected_fields.items():
        if task.get(field) != expected:
            raise ConversionError(f"WebShop trace {field} does not match the frozen goal")
    spec = _spec_json(task.get("webshop_goal_spec"))
    if (canonical_json(spec.get("original_goal")) != canonical_json(goal)
            or canonical_json(spec) != canonical_json(expected_spec)):
        raise ConversionError("WebShop trace original_goal or goal spec does not match the frozen goal")
    metadata = {"scoring_mode": SCORING_MODE, "template_version": TEMPLATE,
                "goal_sha256": goal_hash, "manifest_sha256": corpus, "task_key": key,
                "question": index, "goal_check_valid": True}
    spec.update(metadata)
    spec.update({"goal_response": response, "goal_spec_hash": goal_hash})
    return spec, metadata


def convert_table(table, manifest):
    """Reuse original Arrow columns and metadata, replacing only result/generated fields."""
    import pyarrow as pa
    validate_manifest(manifest)
    if len(set(table.column_names)) != len(table.column_names):
        raise ConversionError("Duplicate Parquet column names are unsupported")
    missing = {"id", "question", "result", "source", "extra_info"} - set(table.column_names)
    if missing:
        raise ConversionError(f"Missing required columns: {sorted(missing)}")
    result_index = table.schema.get_field_index("result")
    result_type = table.schema.field(result_index).type
    if not (pa.types.is_string(result_type) or pa.types.is_large_string(result_type)):
        raise ConversionError("result must have Arrow string or large_string type")
    original = {name: table[name].to_pylist() for name in
                {"id", "question", "result", "source", "extra_info"} | (set(FIELDS) & set(table.column_names))}
    output = {name: [] for name in FIELDS}
    results, invalid, samples = [], [], []
    goals, corpus = manifest["goals"], manifest["manifest_sha256"]
    for row in range(table.num_rows):
        old_result = original["result"][row]
        if old_result not in (None, "") and (
                "webshop_gt_template" not in original or original["webshop_gt_template"][row] != TEMPLATE):
            raise ConversionError(f"row {row}: refusing to overwrite a nonempty result not produced by {TEMPLATE}")
        goal_hash = task_key = goal_json = error = ""
        text = ""
        try:
            index = integer_index(original["question"][row], "question")
            if not 0 <= index < len(goals):
                raise ConversionError(f"question_out_of_range: index {index} outside [0, {len(goals)})")
            extra = original["extra_info"][row]
            if extra is not None and not isinstance(extra, dict):
                raise ConversionError("invalid_extra_info: expected an object or null")
            if isinstance(extra, dict) and "session" in extra:
                session = integer_index(extra["session"], "session")
                if session != index:
                    raise ConversionError(f"session_mismatch: session {session} != question {index}")
            if original["source"][row] != "webshop":
                raise ConversionError("invalid_source: source must equal 'webshop'")
            goal = goals[index]
            goal_hash = sha256_json(goal)
            task_key = f"{MANIFEST_VERSION}:{corpus}:goal:{index}"
            goal_json = canonical_json({"spec_version": SPEC_VERSION, "valid": False, "original_goal": goal})
            text, spec = render_goal(goal)
            goal_json = canonical_json(spec)
        except ConversionError as exc:
            error = str(exc)
        values = {
            "webshop_goal_spec": goal_json, "webshop_goal_sha256": goal_hash,
            "webshop_task_key": task_key, "webshop_gt_template": TEMPLATE,
            "webshop_gt_valid": not bool(error), "webshop_gt_error": error,
            "webshop_manifest_sha256": corpus,
        }
        if old_result not in (None, "") and (error or old_result != text or any(
                name not in original or original[name][row] != value for name, value in values.items())):
            raise ConversionError(f"row {row}: nonempty result does not match this version and frozen manifest")
        results.append(text)
        for name, value in values.items():
            output[name].append(value)
        if error:
            invalid.append({"row": row, "question": original["question"][row], "error": error})
        elif len(samples) < 3:
            samples.append({"row": row, "question": index, "result": text})
    converted = table.set_column(result_index, table.schema.field(result_index), pa.array(results, type=result_type))
    for name, kind in FIELDS.items():
        field = pa.field(name, pa.bool_() if kind == "bool" else pa.string())
        values = pa.array(output[name], type=field.type)
        position = converted.schema.get_field_index(name)
        converted = converted.set_column(position, field, values) if position >= 0 else converted.append_column(field, values)
    audit = {"row_count": table.num_rows, "valid_count": table.num_rows - len(invalid),
             "invalid_count": len(invalid), "invalid_rows": invalid, "samples": samples,
             "error_counts": dict(sorted(Counter(r["error"].split(":", 1)[0] for r in invalid).items()))}
    return converted, audit


def publish_files(items):
    """Stage every file, then publish atomically without overwriting a raced destination.

    A hard-link followed by unlink is the no-clobber equivalent of a same-filesystem
    rename. Roll back this batch's published files if any later publication fails.
    """
    staged, published = [], []
    try:
        for destination, writer in items:
            fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
            os.close(fd)
            temporary = Path(temporary)
            staged.append(temporary)
            writer(temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        for (destination, _), temporary in zip(items, staged):
            os.link(temporary, destination)
            published.append((destination, temporary.stat().st_ino))
    except BaseException:
        for destination, inode in reversed(published):
            if destination.exists() and destination.stat().st_ino == inode:
                destination.unlink()
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def json_writer(value, compressed=False):
    payload = (canonical_json(value) + "\n").encode("utf-8")
    def write(path):
        with path.open("wb") as handle:
            if compressed:
                with gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) as zipped:
                    zipped.write(payload)
            else:
                handle.write(payload)
    return write


def output_paths(args):
    source = Path(args.input).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ConversionError("--input must be a file")
    output = Path(args.output).expanduser().resolve() if args.output else source.with_name(source.stem + "_with_gt.parquet")
    manifest = Path(args.manifest_out).expanduser().resolve() if args.manifest_out else output.with_suffix(".manifest.json.gz")
    report = Path(args.report).expanduser().resolve() if args.report else output.with_suffix(".report.json")
    paths = [output, manifest, report]
    protected = [source]
    if args.manifest_in:
        protected.append(Path(args.manifest_in).expanduser().resolve(strict=True))
    if len(set(paths + protected)) != len(paths + protected):
        raise ConversionError("Input, frozen manifest input, and all output paths must be distinct")
    if output.suffix.lower() != ".parquet":
        raise ConversionError("--output must end in .parquet")
    if not (str(manifest).endswith(".json") or str(manifest).endswith(".json.gz")):
        raise ConversionError("--manifest-out must end in .json or .json.gz")
    for path in paths:
        if os.path.lexists(path):
            raise ConversionError(f"Refusing to overwrite existing file: {path}")
        if not path.parent.is_dir():
            raise ConversionError(f"Output parent directory does not exist: {path.parent}")
    return source, output, manifest, report


def positive_products(value):
    if value.lower() == "all":
        return None
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("num-products must be positive or 'all'")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--input", required=True, help="source Parquet; never modified")
    result.add_argument("--output", help="default: INPUT_STEM_with_gt.parquet")
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest-in", help="reuse a validated frozen .json/.json.gz without native imports")
    mode.add_argument("--webshop-root", help="native WebShop root containing web_agent_site/")
    result.add_argument("--goal-mode", choices=("env-default", "human", "synthetic"),
                        help="native only; default env-default passes human_goals=None, matching current adapter")
    result.add_argument("--num-products", type=positive_products, default=argparse.SUPPRESS,
                        help="native only; default utils.DEBUG_PROD_SIZE; use 'all' for None")
    result.add_argument("--seed", type=int, help=f"native only; RNG seed before initialization, default {DEFAULT_SEED}")
    result.add_argument("--file-path", help="native only; override utils.DEFAULT_FILE_PATH")
    result.add_argument("--manifest-out", help="default: OUTPUT_STEM.manifest.json.gz")
    result.add_argument("--report", help="default: OUTPUT_STEM.report.json; also records failed conversions")
    result.add_argument("--expected-rows", type=int, help="fail if input count differs")
    result.add_argument("--allow-invalid", action="store_true", help="retain invalid rows with empty result; default fails")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.manifest_in and (args.goal_mode is not None or args.seed is not None
                             or args.file_path is not None or hasattr(args, "num_products")):
        parser().error("--manifest-in cannot be combined with native initialization options")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser().error("--seed must be in [0, 2**32)")
    if args.expected_rows is not None and args.expected_rows < 0:
        parser().error("--expected-rows must be nonnegative")
    report_path = None
    report = {"report_version": "webshop-gt-report-v1", "status": "failed", "template": TEMPLATE}
    try:
        source, output, manifest_path, report_path = output_paths(args)
        report.update({"input": str(source), "output": str(output), "manifest_output": str(manifest_path),
                       "report": str(report_path), "outputs_written": [], "expected_rows": args.expected_rows})
        import pyarrow.parquet as pq
        table = pq.read_table(source)
        report["row_count"] = table.num_rows
        if args.expected_rows is not None and args.expected_rows != table.num_rows:
            raise ConversionError(f"Expected {args.expected_rows} rows, found {table.num_rows}")
        manifest = read_manifest(args.manifest_in) if args.manifest_in else export_native_manifest(args)
        report.update({"manifest_sha256": manifest["manifest_sha256"], "goal_count": manifest["goal_count"],
                       "product_price_count": manifest["product_price_count"],
                       "manifest_origin": str(Path(args.manifest_in).resolve()) if args.manifest_in else "fresh-native-initialization"})
        converted, audit = convert_table(table, manifest)
        report.update(audit)
        if audit["invalid_count"] and not args.allow_invalid:
            raise ConversionError(f"{audit['invalid_count']} invalid rows; inspect report or explicitly use --allow-invalid")
        report["status"] = "completed_with_invalid" if audit["invalid_count"] else "success"
        report["outputs_written"] = [str(output), str(manifest_path), str(report_path)]
        publish_files([(output, lambda path: pq.write_table(converted, path, compression="zstd")),
                       (manifest_path, json_writer(manifest, str(manifest_path).endswith(".gz"))),
                       (report_path, json_writer(report))])
        print(canonical_json({key: report[key] for key in ("status", "row_count", "valid_count", "invalid_count",
                                                          "output", "manifest_output", "report", "samples")}))
        return 0
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "outputs_written": []})
        if report_path is not None:
            try:
                publish_files([(report_path, json_writer(report))])
            except Exception as report_exc:
                report["report_write_error"] = f"{type(report_exc).__name__}: {report_exc}"
        print(canonical_json({key: report[key] for key in ("status", "error", "report", "report_write_error") if key in report}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
