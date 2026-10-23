"""Finite, CPU-only diagnostics for TRACE state credit and selected SLIC pairs.

Metrics describe observations and selected pairs; they never alter selection.
The caller must pass selected pairs before any distributed padding or replay.
"""

from collections import defaultdict
import math

import numpy as np


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _numbers(values):
    return [number for value in values if (number := _number(value)) is not None]


def _distribution(values, prefix, *, mean=False, std=False,
                  quantiles=(10, 50, 90), minimum=False, maximum=False):
    """Summarize finite observations, omitting undefined distribution values."""
    finite = _numbers(values)
    metrics = {prefix + "_count": len(finite)}
    if not finite:
        return metrics
    # Scale first so even finite observations near float64's limit cannot
    # overflow the intermediate mean, variance, or percentile interpolation.
    scale = max(abs(value) for value in finite)
    normalized = np.asarray(finite, dtype=float) / (scale or 1.0)
    if mean:
        metrics[prefix + "_mean"] = float(np.mean(normalized)) * scale
    if std:
        metrics[prefix + "_std"] = min(float(np.std(normalized)), 1.0) * scale
    for quantile in quantiles:
        metrics[prefix + f"_p{quantile}"] = float(np.percentile(normalized, quantile)) * scale
    if minimum:
        metrics[prefix + "_min"] = min(finite)
    if maximum:
        metrics[prefix + "_max"] = max(finite)
    return metrics


def _key(value):
    """Keep tuple rollout identities one dimensional, including NumPy input."""
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(_key(item) for item in value)
    return value


def _credit_classes(credits, prefix, config):
    finite = _numbers(credits)
    positive = sum(value > config.positive_threshold for value in finite)
    negative = sum(value < -config.negative_threshold for value in finite)
    neutral = len(finite) - positive - negative
    metrics = {}
    for name, count in (("positive", positive), ("negative", negative), ("neutral", neutral)):
        metrics[prefix + name + "_turns"] = count
        metrics[prefix + name + "_rate"] = count / len(finite) if finite else 0.0
    return metrics


def summarize_states(logps, values, credits, turn_types, rollout_ids, turn_indices, config):
    """Summarize state scores and credit without counting terminal NaNs as loss.

    ``rollout_ids`` must identify a rollout within a question; tuple identities
    such as ``(uid, rollout_id)`` are supported without NumPy reshaping.  A
    rollout's largest original turn index is its terminal state.  Missing
    intermediate scores remain in the credit-coverage denominator.
    """
    size = len(logps)
    if any(len(sequence) != size for sequence in
           (values, credits, turn_types, rollout_ids, turn_indices)):
        raise ValueError("TRACE state metric inputs must have the same length")
    valid_logps = [number for value in logps
                   if (number := _number(value)) is not None and number <= 0]
    metrics = {
        "trace/turns_total": size,
        "trace/score_coverage": len(valid_logps) / size if size else 0.0,
    }
    for name, observations in (("gt_logp", valid_logps), ("value", values), ("credit", credits)):
        metrics.update(_distribution(observations, "trace/" + name, mean=True, std=True))
    metrics.update(_credit_classes(credits, "trace/", config))
    finite_credits = _numbers(credits)
    metrics["trace/unscored_credit_turns"] = size - len(finite_credits)
    absolute = _distribution([abs(value) for value in finite_credits],
                             "trace/credit_abs", quantiles=(90,))
    metrics.update(absolute)

    groups = defaultdict(list)
    for index, (rollout, turn) in enumerate(zip(rollout_ids, turn_indices)):
        groups[_key(rollout)].append((index, _number(turn)))
    terminal = set()
    for observations in groups.values():
        # Invalid turn metadata cannot supply an ordering; use arrival order
        # only to break ties or when no original index can be read.
        terminal.add(max(observations, key=lambda item: (
            item[1] is not None, item[1] if item[1] is not None else 0, item[0]))[0])
    expected = size - len(terminal)
    available = sum(index not in terminal and _number(credit) is not None
                    for index, credit in enumerate(credits))
    metrics.update({
        "trace/terminal_turns": len(terminal),
        "trace/credit_expected_turns": expected,
        "trace/credit_coverage": available / expected if expected else 0.0,
        "trace/credit_missing_nonterminal": expected - available,
    })
    typed = defaultdict(list)
    for turn_type, credit in zip(turn_types, credits):
        typed[str(turn_type)].append(credit)
    for turn_type, observations in sorted(typed.items()):
        prefix = f"trace/{turn_type}/"
        metrics.update(_distribution(observations, prefix + "credit", mean=True))
        metrics.update(_credit_classes(observations, prefix, config))
    return metrics


def summarize_selected_pairs(rows, pairs):
    """Describe unique selected pairs before padding, with optional outcomes.

    ``rows`` have unique ``index`` keys and use the same fields as TRACE pair
    selection.  Outcome reward, when provided, is descriptive only and is never
    consulted by the selector.  No outcome rates are emitted without complete,
    finite outcome observations for at least one selected pair.
    """
    row_map = {row["index"]: row for row in rows}
    if len(row_map) != len(rows):
        raise ValueError("TRACE pair metric rows must have unique index values")
    chosen_indices, rejected_indices, questions = set(), set(), set()
    value_gaps, credit_gaps, chosen_credits, rejected_credits, length_ratios = [], [], [], [], []
    cross_turn = cross_rollout = 0
    outcome_pairs = chosen_failed = rejected_successful = same_outcome = 0
    for chosen_index, rejected_index in pairs:
        chosen, rejected = row_map[chosen_index], row_map[rejected_index]
        chosen_indices.add(chosen_index)
        rejected_indices.add(rejected_index)
        questions.update((_key(chosen["uid"]), _key(rejected["uid"])))
        cross_turn += chosen.get("turn_index") != rejected.get("turn_index")
        cross_rollout += (_key(chosen["uid"]), _key(chosen.get("rollout_id"))) != (
            _key(rejected["uid"]), _key(rejected.get("rollout_id")))
        chosen_value, rejected_value = _number(chosen.get("value")), _number(rejected.get("value"))
        chosen_credit, rejected_credit = _number(chosen.get("credit")), _number(rejected.get("credit"))
        if chosen_value is not None and rejected_value is not None:
            value_gaps.append(abs(chosen_value - rejected_value))
        if chosen_credit is not None:
            chosen_credits.append(chosen_credit)
        if rejected_credit is not None:
            rejected_credits.append(rejected_credit)
        if chosen_credit is not None and rejected_credit is not None:
            credit_gaps.append(chosen_credit - rejected_credit)
        chosen_length, rejected_length = _number(chosen.get("length")), _number(rejected.get("length"))
        if chosen_length is not None and rejected_length is not None and min(chosen_length, rejected_length) > 0:
            length_ratios.append(max(chosen_length, rejected_length) / min(chosen_length, rejected_length))
        chosen_outcome = _number(chosen.get("outcome_reward"))
        rejected_outcome = _number(rejected.get("outcome_reward"))
        if chosen_outcome is not None and rejected_outcome is not None:
            outcome_pairs += 1
            chosen_failed += chosen_outcome <= 0
            rejected_successful += rejected_outcome > 0
            same_outcome += (chosen_outcome > 0) == (rejected_outcome > 0)
    prefix = "trace/pair/"
    count = len(pairs)
    metrics = {
        prefix + "selected_unique": count,
        prefix + "questions_covered": len(questions),
        prefix + "chosen_rows_unique": len(chosen_indices),
        prefix + "rejected_rows_unique": len(rejected_indices),
        prefix + "rows_unique": len(chosen_indices | rejected_indices),
        prefix + "cross_turn_rate": cross_turn / count if count else 0.0,
        prefix + "cross_rollout_rate": cross_rollout / count if count else 0.0,
        prefix + "outcome_pairs_count": outcome_pairs,
    }
    metrics.update(_distribution(value_gaps, prefix + "value_gap", quantiles=(50, 90), maximum=True))
    metrics.update(_distribution(credit_gaps, prefix + "credit_gap", quantiles=(50, 90), minimum=True))
    metrics.update(_distribution(chosen_credits, prefix + "chosen_credit", mean=True, quantiles=()))
    metrics.update(_distribution(rejected_credits, prefix + "rejected_credit", mean=True, quantiles=()))
    metrics.update(_distribution(length_ratios, prefix + "length_ratio", quantiles=(90,)))
    if outcome_pairs:
        metrics[prefix + "chosen_failed_rollout_rate"] = chosen_failed / outcome_pairs
        metrics[prefix + "rejected_successful_rollout_rate"] = rejected_successful / outcome_pairs
        metrics[prefix + "same_outcome_rate"] = same_outcome / outcome_pairs
    return metrics
