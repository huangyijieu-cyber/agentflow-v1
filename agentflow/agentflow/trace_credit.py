"""TRACE state values, adjacent-turn credit, and SLIC pair selection.

This module intentionally has no distributed-training dependencies.  Invalid
model actions can still provide state observations; external execution failures
additionally invalidate every tool-credit window that includes the failure.
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
    k: int = 3
    gamma: float = 0.8
    terminal_fill_enabled: bool = True
    terminal_fill_scale: float = 2.0
    outcome_match_value: bool = False
    outcome_success_threshold: float = 0.5
    match_epsilon: float = 0.05
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
        for name in ("enabled", "terminal_fill_enabled", "outcome_match_value"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"trace.{name} must be a boolean")
        for name in ("k", "k_pairs_per_group", "max_pairs_per_step", "max_gt_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"trace.{name} must be a positive integer")
        for name in ("epsilon", "gamma", "match_epsilon", "positive_threshold",
                     "negative_threshold", "gap_epsilon", "length_ratio_max",
                     "terminal_fill_scale", "outcome_success_threshold"):
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
        if self.terminal_fill_scale < 0:
            raise ValueError("trace.terminal_fill_scale must be nonnegative")
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


def annotate_rollout_credits(values: Sequence[Optional[float]],
                             turn_indices: Sequence[int],
                             turn_types: Sequence[str],
                             external_errors: Sequence[bool],
                             final_flags: Sequence[bool], *,
                             outcome_valid: bool,
                             outcome_advantage: Optional[float], config=None):
    """Assign tool credit without crossing missing turns or external failures.

    K counts original decision positions, including the current decision and
    the actual final answer.  Final has no fabricated GT delta: local credit
    averages only real tool transitions.  Outcome fill requires the window to
    contain the final *decision*, not merely its pre-action state.  Thus K=3
    can fill the final two tool decisions.  Analysis never receives credit.

    ``final_flags`` marks an actual terminal answer, never a last-visible tool
    decision.  ``outcome_valid`` additionally certifies its verifier outcome.
    Invalid scoring windows are kept as NaN, not truncated across a failure.
    """
    cfg = TraceConfig.from_mapping(config)
    size = len(values)
    if any(len(items) != size for items in
           (turn_indices, turn_types, external_errors, final_flags)):
        raise ValueError("TRACE rollout credit inputs must have the same length")
    result = {key: [math.nan] * size for key in
              ("local_credit", "terminal_fill", "effective_credit")}
    result.update({key: [False] * size for key in
                   ("window_blocked", "external_error_blocked", "terminal_reached")})
    finals = [i for i, flag in enumerate(final_flags) if bool(flag)]
    final_position = finals[0] if len(finals) == 1 else None

    def valid_index(position):
        value = turn_indices[position]
        return (not isinstance(value, bool) and _finite(value)
                and float(value).is_integer())

    for start in range(size):
        if str(turn_types[start]) != "tool_plan":
            continue
        if len(finals) > 1 or not valid_index(start):
            result["window_blocked"][start] = True
            continue
        if final_position is not None and start >= final_position:
            continue
        last_turn = int(turn_indices[start]) + cfg.k - 1
        weighted_sum = weight_sum = 0.0
        blocked = False
        reached = False
        for position in range(start, size):
            if (not valid_index(position)
                    or (position > start and float(turn_indices[position])
                        != float(turn_indices[position - 1]) + 1)):
                blocked = True
                break
            turn = int(turn_indices[position])
            if turn > last_turn:
                break
            if bool(external_errors[position]):
                result["external_error_blocked"][start] = True
                blocked = True
                break
            if position == final_position:
                reached = True
                break
            if str(turn_types[position]) != "tool_plan":
                blocked = True
                break
            if position + 1 >= size:
                break  # An observed prefix can have no terminal answer.
            if (not valid_index(position + 1)
                    or float(turn_indices[position + 1]) != turn + 1):
                blocked = True
                break
            weight = cfg.gamma ** (turn - int(turn_indices[start]))
            if weight == 0:
                continue
            if not (_finite(values[position]) and _finite(values[position + 1])):
                blocked = True
                break
            delta = float(values[position + 1]) - float(values[position])
            if not _finite(delta):
                blocked = True
                break
            weighted_sum += weight * delta
            weight_sum += weight
        result["terminal_reached"][start] = reached
        if blocked or weight_sum == 0:
            result["window_blocked"][start] = True
            continue
        local = weighted_sum / weight_sum
        if not _finite(local):
            result["window_blocked"][start] = True
            continue
        result["local_credit"][start] = local
        fill = 0.0
        if (cfg.terminal_fill_enabled and cfg.terminal_fill_scale > 0
                and reached and outcome_valid):
            if not _finite(outcome_advantage):
                result["window_blocked"][start] = True
                continue
            distance = int(turn_indices[final_position]) - int(turn_indices[start])
            fill = cfg.terminal_fill_scale * cfg.gamma ** distance * float(outcome_advantage)
        if not _finite(fill) or not _finite(local + fill):
            result["window_blocked"][start] = True
            continue
        result["terminal_fill"][start] = fill
        result["effective_credit"][start] = local + fill
    return result


def _pair_label_source(row):
    """Reward labels are explicit type routes, never a tool-credit fallback."""
    turn_type = str(row.get("turn_type", ""))
    if turn_type == "analysis":
        source = "outcome"
    elif turn_type in ("final_answer", "direct", "base"):
        if not row.get("is_final", False):
            return None
        source = "outcome"
    else:
        source = "trace"
    requested = row.get("label_source")
    return source if requested in (None, source) else None


def _outcome_success(row, config):
    success = row.get("outcome_success")
    if success is not None:
        if _finite(success) and float(success) in (0.0, 1.0):
            return float(success) == 1.0
        return None
    reward = row.get("outcome_reward")
    return float(reward) > config.outcome_success_threshold if _finite(reward) else None


def select_pairs(rows: Sequence[Mapping[str, Any]], config=None, metrics=None):
    """Select type-routed pairs within question, decision type and label source.

    Each row supplies ``index``, ``uid``, ``turn_type``, ``value``, ``credit``,
    ``valid``, ``length``, and ``old_per_token``.  The returned indices are those
    supplied in each row, not positions in a filtered list.  Rollout and turn
    numbers do not restrict pairing.  Analysis and actual final answers use
    trusted verifier outcomes; tools always require strict signed credit.
    Outcome labels never require GT credit and, by default, do not match GT V.

    ``metrics`` optionally collects counts for this call, resetting these keys
    while preserving unrelated keys.  Candidate filters form an ordered,
    mutually exclusive funnel: sign, value, credit gap, length, distinct action,
    group cap, then global cap.  Every count is an absolute count, not a rate.
    Group sign-absence counts may overlap (a neutral-only group lacks both).
    Among eligible pairs, larger credit gaps take priority; state distance and
    old-policy log-probability distance only break ties.  Outcome pairs sort by
    final reward gap and do not apply the TRACE credit-gap threshold.
    """
    cfg = TraceConfig.from_mapping(config)
    counts = dict.fromkeys((
        "groups_eligible",
        "groups_no_positive", "groups_no_negative", "groups_both_signs",
        "candidates_sign", "rejected_value_match", "rejected_credit_gap",
        "rejected_length", "rejected_same_action", "rejected_group_cap",
        "rejected_global_cap", "selected_unique",
        "trace_candidates_sign", "trace_after_value_match",
    ), 0)
    if metrics is not None:
        metrics.update(counts)
    if not cfg.enabled:
        return []
    groups = defaultdict(list)
    scores, success_flags = {}, {}
    for row in rows:
        source = _pair_label_source(row)
        turn_type = row.get("turn_type")
        key = (row.get("uid"), turn_type, source)
        if (source is None or not row.get("valid", True) or row.get("external_error", False)
                or not all(_finite(row.get(key)) for key in ("old_per_token", "length"))
                or float(row["length"]) <= 0):
            continue
        if source == "outcome":
            success = _outcome_success(row, cfg)
            if (not row.get("outcome_valid", False) or success is None
                    or not _finite(row.get("outcome_reward"))
                    or (cfg.outcome_match_value and not _finite(row.get("value")))):
                continue
            score = float(row["outcome_reward"])
            success_flags[row["index"]] = success
        else:
            if not all(_finite(row.get(name)) for name in ("value", "credit")):
                continue
            score = float(row["credit"])
        scores[row["index"]] = score
        groups[key].append(row)

    counts["groups_eligible"] = len(groups)

    candidates = []
    for group_key, group_rows in groups.items():
        outcome = group_key[2] == "outcome"
        if outcome:
            positive = [row for row in group_rows if success_flags[row["index"]]]
            negative = [row for row in group_rows if not success_flags[row["index"]]]
        else:
            positive = [row for row in group_rows if scores[row["index"]] > cfg.positive_threshold]
            negative = [row for row in group_rows if scores[row["index"]] < -cfg.negative_threshold]
        counts["groups_no_positive"] += not positive
        counts["groups_no_negative"] += not negative
        counts["groups_both_signs"] += bool(positive and negative)
        counts["candidates_sign"] += len(positive) * len(negative)
        if not outcome:
            counts["trace_candidates_sign"] += len(positive) * len(negative)
        for chosen in positive:
            for rejected in negative:
                match_value = not outcome or cfg.outcome_match_value
                value_gap = (abs(float(chosen["value"]) - float(rejected["value"]))
                             if match_value else 0.0)
                credit_gap = scores[chosen["index"]] - scores[rejected["index"]]
                length_ratio = max(float(chosen["length"]), float(rejected["length"])) / min(
                    float(chosen["length"]), float(rejected["length"]))
                if match_value and value_gap > cfg.match_epsilon:
                    counts["rejected_value_match"] += 1
                    continue
                if not outcome:
                    counts["trace_after_value_match"] += 1
                if not outcome and credit_gap < cfg.gap_epsilon:
                    counts["rejected_credit_gap"] += 1
                    continue
                if length_ratio > cfg.length_ratio_max:
                    counts["rejected_length"] += 1
                    continue
                chosen_key, rejected_key = chosen.get("action_key"), rejected.get("action_key")
                if chosen_key is not None and rejected_key is not None and chosen_key == rejected_key:
                    counts["rejected_same_action"] += 1
                    continue
                priority = (-credit_gap, value_gap,
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
