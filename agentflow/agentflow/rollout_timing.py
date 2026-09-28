"""Per-rollout wall-clock timing and validation report generation.

All phase buckets are exclusive. ``executor`` means command generation and
parsing; tool execution is recorded under ``search`` or ``other_tools``.
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PHASES = (
    "planner",
    "executor",
    "search",
    "other_tools",
    "verifier",
    "reward",
    "encoding",
    "other",
)


def profiling_enabled() -> bool:
    return os.environ.get("AGENTFLOW_PROFILE_VAL_TIMING", "").lower() in {"1", "true", "yes"}


def is_search_tool(tool_name: Any) -> bool:
    return "search" in str(tool_name or "").lower()


class RolloutTiming:
    def __init__(self) -> None:
        self.phases = {name: 0.0 for name in PHASES if name != "other"}
        self.planner_detail: dict[str, float] = {}
        self.search_tools: dict[str, float] = {}
        self.tool_calls: dict[str, int] = {}
        self.started = time.perf_counter()

    @contextmanager
    def track(self, phase: str, *, detail: str | None = None, tool: str | None = None):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self.phases[phase] += elapsed
            if phase == "planner" and detail:
                self.planner_detail[detail] = self.planner_detail.get(detail, 0.0) + elapsed
            if phase in {"search", "other_tools"} and tool:
                self.tool_calls[tool] = self.tool_calls.get(tool, 0) + 1
                if phase == "search":
                    self.search_tools[tool] = self.search_tools.get(tool, 0.0) + elapsed

    def snapshot(self) -> dict[str, Any]:
        return {
            "solver_total_s": time.perf_counter() - self.started,
            "phases_s": dict(self.phases),
            "planner_detail_s": dict(self.planner_detail),
            "search_tools_s": dict(self.search_tools),
            "tool_calls": dict(self.tool_calls),
        }


def finish_rollout(
    solver_profile: dict[str, Any] | None,
    total_s: float,
    reward_s: float,
    encoding_s: float,
) -> dict[str, Any]:
    phases = {name: 0.0 for name in PHASES}
    if solver_profile:
        for name, value in solver_profile.get("phases_s", {}).items():
            if name in phases and name != "other":
                phases[name] = float(value)
    phases["reward"] = reward_s
    phases["encoding"] = encoding_s
    phases["other"] = max(0.0, total_s - sum(phases.values()))
    return {
        "total_s": total_s,
        "phases_s": phases,
        "phases_pct": {name: 100.0 * value / total_s if total_s > 0 else 0.0 for name, value in phases.items()},
        "planner_detail_s": (solver_profile or {}).get("planner_detail_s", {}),
        "search_tools_s": (solver_profile or {}).get("search_tools_s", {}),
        "tool_calls": (solver_profile or {}).get("tool_calls", {}),
        "executor_full_s": phases["executor"] + phases["search"] + phases["other_tools"],
    }


def summarize_profiles(profiles: Iterable[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    records = list(profiles)
    totals = [float(record["total_s"]) for record in records]
    phase_stats = {}
    for phase in PHASES:
        seconds = [float(record["phases_s"].get(phase, 0.0)) for record in records]
        percentages = [100.0 * part / total if total > 0 else 0.0 for part, total in zip(seconds, totals)]
        phase_stats[phase] = {
            "mean_s": statistics.mean(seconds) if seconds else 0.0,
            "mean_pct": statistics.mean(percentages) if percentages else 0.0,
            "total_share_pct": 100.0 * sum(seconds) / sum(totals) if sum(totals) > 0 else 0.0,
        }
    search_tools = sorted({name for record in records for name in record.get("search_tools_s", {})})
    planner_parts = sorted({name for record in records for name in record.get("planner_detail_s", {})})
    executor_full = [
        float(record["phases_s"].get("executor", 0.0))
        + float(record["phases_s"].get("search", 0.0))
        + float(record["phases_s"].get("other_tools", 0.0))
        for record in records
    ]
    return {
        "expected_count": expected_count,
        "profiled_count": len(records),
        "complete_expected": len(records) == expected_count,
        "mean_total_s": statistics.mean(totals) if totals else 0.0,
        "phase_stats": phase_stats,
        "executor_full_mean_s": statistics.mean(executor_full) if executor_full else 0.0,
        "executor_full_mean_pct": statistics.mean(
            100.0 * part / total if total > 0 else 0.0
            for part, total in zip(executor_full, totals)
        ) if totals else 0.0,
        "planner_detail_mean_s": {
            name: statistics.mean(float(record.get("planner_detail_s", {}).get(name, 0.0)) for record in records)
            for name in planner_parts
        },
        "search_tool_mean_s": {
            name: statistics.mean(float(record.get("search_tools_s", {}).get(name, 0.0)) for record in records)
            for name in search_tools
        },
    }


def write_validation_report(
    rollouts: Iterable[Any], output_dir: str | Path, expected_count: int, run_name: str
) -> tuple[Path, Path, dict[str, Any]]:
    records = []
    completed_count = 0
    valid_count = 0
    for rollout in rollouts:
        completed_count += 1
        valid = bool(getattr(rollout, "triplets", None))
        valid_count += int(valid)
        metadata = rollout.metadata or {}
        profile = metadata.get("timing_profile")
        if not profile:
            continue
        records.append({"rollout_id": str(rollout.rollout_id), "valid": valid, **profile})

    records.sort(key=lambda item: item["rollout_id"])
    summary = summarize_profiles(records, expected_count)
    summary["completed_count"] = completed_count
    summary["valid_completed_count"] = valid_count
    summary["missing_timing_count"] = completed_count - len(records)
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["hostname"] = socket.gethostname()

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    detail_path = directory / f"{run_name}_rollouts.jsonl"
    summary_path = directory / f"{run_name}_summary.json"
    with detail_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return detail_path, summary_path, summary
