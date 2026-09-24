#!/usr/bin/env python3
import argparse
import json
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def normalize_entity(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(string.whitespace + string.punctuation + "“”‘’")


def entity_mentioned(alias: Any, observation: Any) -> bool:
    alias_norm = normalize_entity(alias)
    if not alias_norm:
        return False

    if isinstance(observation, str):
        observation_text = observation
    else:
        observation_text = json.dumps(observation, ensure_ascii=False, default=str)

    observation_norm = unicodedata.normalize("NFKC", observation_text).casefold()
    observation_norm = re.sub(r"\s+", " ", observation_norm)

    pattern = r"(?<!\w)" + re.escape(alias_norm) + r"(?!\w)"
    return re.search(pattern, observation_norm) is not None


def compute_search_subreward(result: dict, reward_spec: Any) -> Tuple[float, List[dict]]:
    reward_spec = as_dict(reward_spec)
    subgoals = reward_spec.get("subgoals", []) or []
    count_once = bool(reward_spec.get("count_each_subgoal_once", True))

    hit_subgoals = set()
    hit_details = []
    subreward = 0.0

    memory = result.get("memory", {}) or {}
    if not isinstance(memory, dict):
        return 0.0, []

    for step_name, action in memory.items():
        if not isinstance(action, dict):
            continue

        tool_name = str(action.get("tool_name", ""))
        if "search" not in tool_name.casefold():
            continue

        observation = action.get("result", "")
        turn_match = re.search(r"(\d+)", str(step_name))
        turn = int(turn_match.group(1)) if turn_match else None

        for subgoal in subgoals:
            if not isinstance(subgoal, dict):
                continue

            subgoal_id = str(subgoal.get("id", ""))
            if count_once and subgoal_id in hit_subgoals:
                continue

            aliases = list(subgoal.get("aliases", []) or [])
            if subgoal.get("answer"):
                aliases.append(subgoal["answer"])

            matched_alias = next(
                (alias for alias in aliases if entity_mentioned(alias, observation)),
                None,
            )
            if matched_alias is None:
                continue

            weight = float(subgoal.get("weight", 0.0))
            subreward += weight
            hit_subgoals.add(subgoal_id)
            hit_details.append(
                {
                    "turn": turn,
                    "tool_name": tool_name,
                    "subgoal_id": subgoal_id,
                    "matched_alias": str(matched_alias),
                    "weight": weight,
                }
            )

    return subreward, hit_details


def iter_json_records(path: Path) -> Iterable[dict]:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in {".json", ".jsonl"}:
                yield from iter_json_records(child)
        return

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL: {path}:{line_no}: {e}") from e
                if isinstance(obj, list):
                    for item in obj:
                        if isinstance(item, dict):
                            yield item
                elif isinstance(obj, dict):
                    yield obj
        return

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        yield obj


def sample_id(record: dict) -> str:
    for key in ("id", "sample_id", "source_id"):
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def extract_result(record: dict) -> dict:
    for key in ("total_result", "result", "rollout_result"):
        value = record.get(key)
        if isinstance(value, dict):
            return value

    if isinstance(record.get("memory"), dict):
        return record

    return {}


def extract_reward_spec(sample: dict) -> dict:
    spec = sample.get("reward_spec")
    if spec:
        return as_dict(spec)

    extra_info = as_dict(sample.get("extra_info", {}))
    return as_dict(extra_info.get("reward_spec", {}))


def load_dataset(dataset_path: Path) -> Dict[str, dict]:
    dataset = {}
    for record in iter_json_records(dataset_path):
        sid = sample_id(record)
        if not sid:
            continue
        dataset[sid] = record
    return dataset


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate InfoSeek search-result subgoal reward hit rates."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Original InfoSeek reward JSON/JSONL containing id + reward_spec.",
    )
    parser.add_argument(
        "--rollouts",
        required=True,
        help="Rollout JSON/JSONL file or directory. Each rollout needs id + total_result/result.",
    )
    parser.add_argument(
        "--output",
        default="infoseek_reward_metrics.json",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--details-output",
        default="infoseek_reward_details.jsonl",
        help="Per-rollout detail JSONL output path.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rollouts_path = Path(args.rollouts)
    output_path = Path(args.output)
    details_path = Path(args.details_output)

    dataset = load_dataset(dataset_path)
    if not dataset:
        raise ValueError(f"No valid samples found in dataset: {dataset_path}")

    detail_rows = []
    missing_sample_ids = []
    missing_result = 0
    missing_reward_spec = 0

    total_rollouts = 0
    rollout_hit_count = 0
    rollout_full_hit_count = 0
    total_subreward = 0.0
    total_hit_subgoals = 0
    total_available_subgoals = 0

    sample_stats = defaultdict(
        lambda: {
            "rollouts": 0,
            "hit_rollouts": 0,
            "subrewards": [],
            "hit_subgoals_union": set(),
        }
    )
    hit_turn_counter = Counter()
    subgoal_hit_counter = Counter()

    for rollout_index, rollout_record in enumerate(iter_json_records(rollouts_path)):
        sid = sample_id(rollout_record)
        if not sid or sid not in dataset:
            missing_sample_ids.append(sid or f"<missing-id:{rollout_index}>")
            continue

        result = extract_result(rollout_record)
        if not result:
            missing_result += 1
            continue

        reward_spec = extract_reward_spec(dataset[sid])
        subgoals = reward_spec.get("subgoals", []) or []
        if not reward_spec or not subgoals:
            missing_reward_spec += 1
            continue

        subreward, hits = compute_search_subreward(result, reward_spec)
        hit_ids = {hit["subgoal_id"] for hit in hits}
        n_subgoals = len(subgoals)
        n_hits = len(hit_ids)

        total_rollouts += 1
        total_subreward += subreward
        total_hit_subgoals += n_hits
        total_available_subgoals += n_subgoals

        if n_hits > 0:
            rollout_hit_count += 1
        if n_subgoals > 0 and n_hits == n_subgoals:
            rollout_full_hit_count += 1

        stat = sample_stats[sid]
        stat["rollouts"] += 1
        stat["subrewards"].append(subreward)
        stat["hit_subgoals_union"].update(hit_ids)
        if n_hits > 0:
            stat["hit_rollouts"] += 1

        for hit in hits:
            subgoal_hit_counter[hit["subgoal_id"]] += 1
            if hit["turn"] is not None:
                hit_turn_counter[str(hit["turn"])] += 1

        detail_rows.append(
            {
                "id": sid,
                "rollout_index": rollout_index,
                "subreward": subreward,
                "hit_subgoal_count": n_hits,
                "total_subgoal_count": n_subgoals,
                "hit_subgoal_ids": sorted(hit_ids),
                "hits": hits,
            }
        )

    evaluated_samples = len(sample_stats)
    sample_any_hit_count = sum(
        1 for stat in sample_stats.values() if stat["hit_rollouts"] > 0
    )
    sample_all_zero_count = sum(
        1 for stat in sample_stats.values() if stat["hit_rollouts"] == 0
    )

    summary = {
        "dataset_samples": len(dataset),
        "evaluated_samples": evaluated_samples,
        "evaluated_rollouts": total_rollouts,
        "rollout_hit_rate": safe_rate(rollout_hit_count, total_rollouts),
        "rollout_zero_hit_rate": safe_rate(
            total_rollouts - rollout_hit_count, total_rollouts
        ),
        "rollout_full_hit_rate": safe_rate(
            rollout_full_hit_count, total_rollouts
        ),
        "sample_any_hit_rate": safe_rate(
            sample_any_hit_count, evaluated_samples
        ),
        "sample_all_zero_rate": safe_rate(
            sample_all_zero_count, evaluated_samples
        ),
        "subgoal_hit_rate": safe_rate(
            total_hit_subgoals, total_available_subgoals
        ),
        "avg_subreward": (
            total_subreward / total_rollouts if total_rollouts else 0.0
        ),
        "avg_hit_subgoals_per_rollout": (
            total_hit_subgoals / total_rollouts if total_rollouts else 0.0
        ),
        "first_hit_turn_counts": dict(
            sorted(hit_turn_counter.items(), key=lambda x: int(x[0]))
        ),
        "subgoal_hit_counts": dict(sorted(subgoal_hit_counter.items())),
        "skipped": {
            "missing_sample_id_or_not_in_dataset": len(missing_sample_ids),
            "missing_result": missing_result,
            "missing_reward_spec": missing_reward_spec,
        },
    }

    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== InfoSeek Process Reward Evaluation ===")
    print(f"Dataset samples          : {summary['dataset_samples']}")
    print(f"Evaluated samples        : {summary['evaluated_samples']}")
    print(f"Evaluated rollouts       : {summary['evaluated_rollouts']}")
    print(f"Rollout hit rate         : {summary['rollout_hit_rate']:.2%}")
    print(f"Rollout zero-hit rate    : {summary['rollout_zero_hit_rate']:.2%}")
    print(f"Rollout full-hit rate    : {summary['rollout_full_hit_rate']:.2%}")
    print(f"Sample any-hit rate      : {summary['sample_any_hit_rate']:.2%}")
    print(f"Sample all-zero rate     : {summary['sample_all_zero_rate']:.2%}")
    print(f"Subgoal hit rate         : {summary['subgoal_hit_rate']:.2%}")
    print(f"Average subreward        : {summary['avg_subreward']:.4f}")
    print(
        "Avg hit subgoals/rollout: "
        f"{summary['avg_hit_subgoals_per_rollout']:.4f}"
    )
    print(f"First-hit turn counts    : {summary['first_hit_turn_counts']}")
    print(f"Skipped                  : {summary['skipped']}")
    print(f"Summary saved to         : {output_path}")
    print(f"Details saved to         : {details_path}")


if __name__ == "__main__":
    main()
