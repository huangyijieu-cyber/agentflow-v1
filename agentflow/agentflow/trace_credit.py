"""TRACE state values, adjacent-turn credit, and SLIC pair selection.

This module intentionally has no distributed-training dependencies.  A state
observation can be useful even when the action taken from it is invalid: action
validity is applied only by :func:`select_pairs`, never by the value functions.
"""

from dataclasses import asdict, dataclass, fields
import math
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = True
    # TRACE section 4.1 / appendix A.1 table 4: training gap offset.
    epsilon: float = 0.1
    k: int = 1
    gamma: float = 0.8
    match_epsilon: float = 0.1
    positive_threshold: float = 0.01
    negative_threshold: float = 0.01
    gap_epsilon: float = 0.02
    length_ratio_max: float = 1.8
    k_pairs_per_group: int = 2
    max_pairs_per_step: int = 256
    answer_instruction: str = "Answer the question directly and concisely."
    max_gt_tokens: int = 256
    gt_alias_strategy: str = "first"

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("trace.enabled must be a boolean")
        for name in ("k", "k_pairs_per_group", "max_pairs_per_step", "max_gt_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"trace.{name} must be a positive integer")
        for name in ("epsilon", "gamma", "match_epsilon", "positive_threshold",
                     "negative_threshold", "gap_epsilon", "length_ratio_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not _finite(value):
                raise ValueError(f"trace.{name} must be finite")
        if self.epsilon <= 0:
            raise ValueError("trace.epsilon must be positive")
        if not 0 <= self.gamma <= 1:
            raise ValueError("trace.gamma must be between 0 and 1")
        for name in ("match_epsilon", "positive_threshold", "negative_threshold", "gap_epsilon"):
            if getattr(self, name) < 0:
                raise ValueError(f"trace.{name} must be nonnegative")
        if self.length_ratio_max < 1:
            raise ValueError("trace.length_ratio_max must be at least 1")
        if not isinstance(self.answer_instruction, str) or not self.answer_instruction.strip():
            raise ValueError("trace.answer_instruction must be a nonempty string")
        if self.gt_alias_strategy != "first":
            raise ValueError("trace.gt_alias_strategy currently supports only 'first'")

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]] = None):
        if isinstance(config, cls):
            return config
        if config is None:
            return cls()
        known_fields = {field.name for field in fields(cls)}
        unknown = set(config) - known_fields
        if unknown:
            raise ValueError(f"Unknown TRACE configuration fields: {sorted(unknown)}")
        return cls(**dict(config))


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


DEFAULT_TRACE_CONFIG = asdict(TraceConfig())


def state_values(logps: Sequence[Optional[float]], epsilon: float = 0.1):
    """Return log((-ell_0 + eps) / (-ell_t + eps)) for one rollout.

    Missing, nonfinite, or positive log probabilities are invalid observations.
    A missing initial observation invalidates the entire rollout's values;
    otherwise individual missing observations preserve their original positions.
    """
    if not _finite(epsilon) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    result = [math.nan] * len(logps)
    if not len(logps) or not _finite(logps[0]) or float(logps[0]) > 0:
        return result
    initial_nll = -float(logps[0]) + epsilon
    for index, logp in enumerate(logps):
        if _finite(logp) and float(logp) <= 0:
            # Subtract logs rather than forming a ratio, avoiding overflow for
            # very small eps and large finite negative log probabilities.
            result[index] = math.log(initial_nll) - math.log(-float(logp) + epsilon)
    return result


def discounted_credits(values: Sequence[Optional[float]],
                       turn_indices: Optional[Sequence[int]] = None,
                       k: int = 1, gamma: float = 0.8):
    """Compute normalized K-step credit without crossing an unobserved turn.

    The input is one rollout in its original state order; it must not be
    prefiltered by action validity.  The final state has no successor and gets
    NaN.  A shortened window at the rollout end is normalized by its own weight
    sum.  Missing observations or discontinuous original turn indices invalidate
    each window that needs their delta, instead of joining distant observations.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not _finite(gamma) or isinstance(gamma, bool) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and between 0 and 1")
    size = len(values)
    if turn_indices is None:
        turn_indices = range(size)
    if len(turn_indices) != size:
        raise ValueError("turn_indices and values must have the same length")
    deltas = [math.nan] * max(0, size - 1)
    for index in range(size - 1):
        left, right = turn_indices[index], turn_indices[index + 1]
        # Checking both the gap and integral indices avoids silently accepting
        # fractional turn numbers or joining duplicated turns.
        continuous = (_finite(left) and _finite(right)
                      and float(left).is_integer() and float(right).is_integer()
                      and float(right) == float(left) + 1)
        if continuous and _finite(values[index]) and _finite(values[index + 1]):
            deltas[index] = float(values[index + 1]) - float(values[index])
    result = [math.nan] * size
    for index in range(size - 1):
        weighted_sum, weight_sum = 0.0, 0.0
        for offset, delta in enumerate(deltas[index:min(index + k, size - 1)]):
            weight = gamma ** offset
            if weight == 0:
                continue
            if not _finite(delta):
                break
            weighted_sum += weight * delta
            weight_sum += weight
        else:
            if weight_sum:
                result[index] = weighted_sum / weight_sum
    return result


def select_pairs(rows: Sequence[Mapping[str, Any]], config=None, metrics=None):
    """Select strict positive/negative pairs within question and decision type.

    Each row supplies ``index``, ``uid``, ``turn_type``, ``value``, ``credit``,
    ``valid``, ``length``, and ``old_per_token``.  The returned indices are those
    supplied in each row, not positions in a filtered list.  Rollout and turn
    numbers do not restrict pairing.  No outcome reward or fallback is used.

    ``metrics`` optionally collects counts for this call, resetting these keys
    while preserving unrelated keys.  Candidate filters form an ordered,
    mutually exclusive funnel: sign, value, credit gap, length, distinct action,
    group cap, then global cap.  Every count is an absolute count, not a rate.
    Group sign-absence counts may overlap (a neutral-only group lacks both).
    """
    cfg = TraceConfig.from_mapping(config)
    counts = dict.fromkeys((
        "rows_total", "rows_eligible", "groups_total", "groups_eligible",
        "groups_no_positive", "groups_no_negative", "groups_both_signs",
        "positive_rows", "negative_rows", "neutral_rows", "candidates_sign",
        "after_value_match", "after_credit_gap", "after_length_filter",
        "after_distinct_action", "rejected_value_match", "rejected_credit_gap",
        "rejected_length", "rejected_same_action", "rejected_group_cap",
        "rejected_global_cap", "selected_unique",
    ), 0)
    counts["rows_total"] = len(rows)
    if metrics is not None:
        metrics.update(counts)
    if not cfg.enabled:
        return []
    groups = defaultdict(list)
    raw_group_keys = set()
    for row in rows:
        # Invalid observations still contribute to the raw group population.
        raw_group_keys.add((row.get("uid"), row.get("turn_type")))
        if (not row.get("valid", True)
                or not all(_finite(row.get(key)) for key in ("value", "credit", "old_per_token", "length"))
                or float(row["length"]) <= 0):
            continue
        groups[(row["uid"], row["turn_type"])].append(row)
        counts["rows_eligible"] += 1

    counts["groups_total"] = len(raw_group_keys)
    counts["groups_eligible"] = len(groups)

    candidates = []
    for group_key, group_rows in groups.items():
        positive = [row for row in group_rows if float(row["credit"]) > cfg.positive_threshold]
        negative = [row for row in group_rows if float(row["credit"]) < -cfg.negative_threshold]
        counts["positive_rows"] += len(positive)
        counts["negative_rows"] += len(negative)
        counts["neutral_rows"] += len(group_rows) - len(positive) - len(negative)
        counts["groups_no_positive"] += not positive
        counts["groups_no_negative"] += not negative
        counts["groups_both_signs"] += bool(positive and negative)
        counts["candidates_sign"] += len(positive) * len(negative)
        for chosen in positive:
            for rejected in negative:
                value_gap = abs(float(chosen["value"]) - float(rejected["value"]))
                credit_gap = float(chosen["credit"]) - float(rejected["credit"])
                length_ratio = max(float(chosen["length"]), float(rejected["length"])) / min(
                    float(chosen["length"]), float(rejected["length"]))
                if value_gap > cfg.match_epsilon:
                    counts["rejected_value_match"] += 1
                    continue
                counts["after_value_match"] += 1
                if credit_gap < cfg.gap_epsilon:
                    counts["rejected_credit_gap"] += 1
                    continue
                counts["after_credit_gap"] += 1
                if length_ratio > cfg.length_ratio_max:
                    counts["rejected_length"] += 1
                    continue
                counts["after_length_filter"] += 1
                chosen_key, rejected_key = chosen.get("action_key"), rejected.get("action_key")
                if chosen_key is not None and rejected_key is not None and chosen_key == rejected_key:
                    counts["rejected_same_action"] += 1
                    continue
                counts["after_distinct_action"] += 1
                priority = (value_gap, -credit_gap,
                            abs(float(chosen["old_per_token"]) - float(rejected["old_per_token"])),
                            chosen["index"], rejected["index"])
                candidates.append((priority, group_key, chosen["index"], rejected["index"]))

    candidates.sort(key=lambda item: item[0])
    per_group = defaultdict(int)
    pairs = []
    for _, group_key, chosen_index, rejected_index in candidates:
        if per_group[group_key] >= cfg.k_pairs_per_group:
            counts["rejected_group_cap"] += 1
            continue
        # Continue counting after the original break point, without accepting
        # another pair or letting rejected candidates consume group capacity.
        if len(pairs) >= cfg.max_pairs_per_step:
            counts["rejected_global_cap"] += 1
            continue
        pairs.append((chosen_index, rejected_index))
        per_group[group_key] += 1
    counts["selected_unique"] = len(set(pairs))
    if metrics is not None:
        metrics.update(counts)
    return pairs
