"""Rollout-level reward diagnostics for AgentFlow training and validation."""

from collections import defaultdict
from typing import Any, Iterable


def summarize_reward_metrics(
    breakdowns: Iterable[dict[str, Any]], prefix: str, max_tool_steps: int = 5
) -> dict[str, float]:
    """Average once per completed QA rollout, never once per planner turn."""
    records = [item for item in breakdowns if isinstance(item, dict) and "final_reward" in item]
    metrics = {
        f"{prefix}/rollout_count": float(len(records)),
        f"{prefix}/reward_spec_coverage": 0.0,
    }
    if not records:
        return metrics

    metrics[f"{prefix}/final_correct_rate"] = sum(
        float(item["final_reward"]) for item in records
    ) / len(records)
    metrics[f"{prefix}/answer_tag_rate"] = sum(
        bool(item.get("answer_tag_valid", False)) for item in records
    ) / len(records)

    eligible = [item for item in records if int(item.get("n_subgoals", 0)) > 0]
    metrics[f"{prefix}/reward_spec_coverage"] = len(eligible) / len(records)
    metrics[f"{prefix}/reward_spec_count"] = float(len(eligible))
    if not eligible:
        # A dataset without annotated subgoals has no defined subreward rate.
        return metrics

    metrics[f"{prefix}/subreward_activation_rate"] = sum(
        float(item.get("subreward", 0.0)) > 0 for item in eligible
    ) / len(eligible)
    metrics[f"{prefix}/subreward_mean"] = sum(
        float(item.get("subreward", 0.0)) for item in eligible
    ) / len(eligible)

    wrong = [item for item in eligible if float(item["final_reward"]) == 0.0]
    if wrong:
        metrics[f"{prefix}/subreward_mean_when_final_wrong"] = sum(
            float(item.get("subreward", 0.0)) for item in wrong
        ) / len(wrong)
    metrics[f"{prefix}/final_wrong_count"] = float(len(wrong))

    hit_count = sum(
        min(
            len({str(hit.get("subgoal_id", "")) for hit in item.get("subgoal_hits", [])
                 if isinstance(hit, dict)}),
            int(item["n_subgoals"]),
        )
        for item in eligible
    )
    subgoal_count = sum(int(item["n_subgoals"]) for item in eligible)
    metrics[f"{prefix}/subgoal_coverage"] = hit_count / subgoal_count

    search_count = hit_search_count = 0
    count_by_turn = defaultdict(int)
    hits_by_turn = defaultdict(int)
    for item in eligible:
        rewarded_turns = {
            int(turn) for turn, reward in (item.get("turn_process_rewards") or {}).items()
            if float(reward) > 0
        }
        for raw_turn in item.get("search_turn_indices", []) or []:
            turn = int(raw_turn)
            search_count += 1
            hit = turn in rewarded_turns
            hit_search_count += hit
            if 1 <= turn <= max_tool_steps:
                count_by_turn[turn] += 1
                hits_by_turn[turn] += hit

    metrics[f"{prefix}/search_turn_count"] = float(search_count)
    if search_count:
        metrics[f"{prefix}/search_turn_hit_rate"] = hit_search_count / search_count
    for turn in range(1, max_tool_steps + 1):
        count = count_by_turn[turn]
        metrics[f"{prefix}/search_turn_count/turn_{turn}"] = float(count)
        if count:
            metrics[f"{prefix}/search_turn_hit_rate/turn_{turn}"] = hits_by_turn[turn] / count
    return metrics


def summarize_gigpo_groups(
    uids: Iterable[Any], anchors: Iterable[Any], pair_masks: Iterable[bool],
    step_rewards: Iterable[float], trainable_masks: Iterable[bool] | None = None,
) -> dict[str, float]:
    """Measure how often eligible turns have a peer and a nonzero step signal."""
    uids = list(uids)
    anchors = list(anchors)
    pair_masks = list(pair_masks)
    step_rewards = list(step_rewards)
    if trainable_masks is None:
        trainable_masks = [True] * len(uids)
    rows = list(zip(uids, anchors, pair_masks, step_rewards, trainable_masks, strict=True))
    groups = defaultdict(list)
    for uid, anchor, pair_enabled, reward, trainable in rows:
        if pair_enabled:
            groups[(str(uid), str(anchor))].append((float(reward), bool(trainable)))

    eligible = sum(bool(row[2]) and bool(row[4]) for row in rows)
    paired = nonzero = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        mean = sum(reward for reward, _ in members) / len(members)
        for reward, trainable in members:
            if trainable:
                paired += 1
                nonzero += abs(reward - mean) > 1e-8
    return {
        "gigpo/paired_turn_rate": paired / eligible if eligible else 0.0,
        "gigpo/nonzero_step_adv_rate": nonzero / paired if paired else 0.0,
        "gigpo/eligible_turn_count": float(eligible),
        "gigpo/paired_turn_count": float(paired),
    }
