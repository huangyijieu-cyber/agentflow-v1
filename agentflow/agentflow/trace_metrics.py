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


def summarize_states(logps, values, credits, turn_types, rollout_ids, turn_indices, config,
                     *, local_credits=None, terminal_fills=None, window_blocked=None,
                     external_errors=None, outcome_valid=None, label_sources=None,
                     credit_expected_mask=None):
    """Summarize state scores and credit without counting terminal NaNs as loss.

    ``rollout_ids`` must identify a rollout within a question; tuple identities
    such as ``(uid, rollout_id)`` are supported without NumPy reshaping.  A
    rollout's largest original turn index is its terminal state.  Missing
    intermediate scores remain in the credit-coverage denominator. New callers
    supply ``credit_expected_mask`` for tool decisions only, so outcome-labelled
    analysis/final rows do not count as missing tool credit. All optional
    observations are diagnostics and cannot change pair selection.
    """
    size = len(logps)
    if any(len(sequence) != size for sequence in
           (values, credits, turn_types, rollout_ids, turn_indices)):
        raise ValueError("TRACE state metric inputs must have the same length")
    optional = {
        "local_credit": local_credits, "terminal_fill": terminal_fills,
        "window_blocked": window_blocked, "external_errors": external_errors,
        "outcome_valid": outcome_valid, "label_sources": label_sources,
        "credit_expected_mask": credit_expected_mask,
    }
    if any(sequence is not None and len(sequence) != size for sequence in optional.values()):
        raise ValueError("TRACE optional state metric inputs must have the same length")
    valid_logps = [number for value in logps
                   if (number := _number(value)) is not None and number <= 0]
    metrics = {
        "trace/turns_total": size,
        "trace/score_coverage": len(valid_logps) / size if size else 0.0,
    }
    for name, observations in (("gt_logp", valid_logps), ("value", values), ("credit", credits)):
        metrics.update(_distribution(observations, "trace/" + name, mean=True, std=True))
    if local_credits is not None:
        metrics.update(_distribution(local_credits, "trace/local_credit", mean=True, std=True))
    if terminal_fills is not None:
        metrics.update(_distribution(terminal_fills, "trace/terminal_fill", mean=True, std=True))
        metrics["trace/terminal_fill_nonzero_turns"] = sum(
            value != 0 for value in _numbers(terminal_fills))
    if local_credits is not None or terminal_fills is not None:
        metrics.update(_distribution(credits, "trace/effective_credit", mean=True, std=True))
    if window_blocked is not None:
        metrics["trace/credit_windows_blocked_external"] = sum(bool(value) for value in window_blocked)
    if external_errors is not None:
        metrics["trace/external_error_turns"] = sum(bool(value) for value in external_errors)
    if outcome_valid is not None:
        metrics["trace/outcome_invalid_turns"] = sum(not bool(value) for value in outcome_valid)
        rollout_outcomes = defaultdict(bool)
        for rollout, valid in zip(rollout_ids, outcome_valid):
            # Outputs after the actual evaluated final intentionally have no
            # outcome label; they cannot invalidate a trusted rollout prefix.
            rollout_outcomes[_key(rollout)] |= bool(valid)
        metrics["trace/outcome_invalid_rollouts"] = sum(
            not valid for valid in rollout_outcomes.values())
    if label_sources is not None:
        sources = defaultdict(int)
        for source in label_sources:
            sources[str(source)] += 1
        for source, count in sorted(sources.items()):
            metrics[f"trace/label_source/{source}/turns"] = count
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
    expected_mask = ([index not in terminal for index in range(size)]
                     if credit_expected_mask is None else [bool(value) for value in credit_expected_mask])
    expected = sum(expected_mask)
    available = sum(expected_mask[index] and _number(credit) is not None
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
    selection. Explicit ``label_source='outcome'`` rows are summarized separately
    from TRACE credit/value gaps; reward labels are never reported as GT credit.
    Outcome agreement is descriptive and prefers verified ``outcome_success``
    flags. Legacy rows without a source retain their historical reward > 0
    diagnostic convention.
    """
    row_map = {row["index"]: row for row in rows}
    if len(row_map) != len(rows):
        raise ValueError("TRACE pair metric rows must have unique index values")
    chosen_indices, rejected_indices, questions = set(), set(), set()
    value_gaps, credit_gaps, chosen_credits, rejected_credits, length_ratios = [], [], [], [], []
    outcome_reward_gaps = []
    source_counts = defaultdict(int)
    source_outcomes = defaultdict(lambda: [0, 0, 0, 0])
    source_mismatch = 0
    cross_turn = cross_rollout = 0
    outcome_pairs = chosen_failed = rejected_successful = same_outcome = 0
    for chosen_index, rejected_index in pairs:
        chosen, rejected = row_map[chosen_index], row_map[rejected_index]
        chosen_source = str(chosen.get("label_source", "trace"))
        rejected_source = str(rejected.get("label_source", "trace"))
        same_source = chosen_source == rejected_source
        source_mismatch += not same_source
        if same_source:
            source_counts[chosen_source] += 1
        chosen_indices.add(chosen_index)
        rejected_indices.add(rejected_index)
        questions.update((_key(chosen["uid"]), _key(rejected["uid"])))
        cross_turn += chosen.get("turn_index") != rejected.get("turn_index")
        cross_rollout += (_key(chosen["uid"]), _key(chosen.get("rollout_id"))) != (
            _key(rejected["uid"]), _key(rejected.get("rollout_id")))
        chosen_value, rejected_value = _number(chosen.get("value")), _number(rejected.get("value"))
        chosen_credit, rejected_credit = _number(chosen.get("credit")), _number(rejected.get("credit"))
        trace_pair = same_source and chosen_source == "trace"
        if trace_pair and chosen_value is not None and rejected_value is not None:
            value_gaps.append(abs(chosen_value - rejected_value))
        if trace_pair and chosen_credit is not None:
            chosen_credits.append(chosen_credit)
        if trace_pair and rejected_credit is not None:
            rejected_credits.append(rejected_credit)
        if trace_pair and chosen_credit is not None and rejected_credit is not None:
            credit_gaps.append(chosen_credit - rejected_credit)
        chosen_length, rejected_length = _number(chosen.get("length")), _number(rejected.get("length"))
        if chosen_length is not None and rejected_length is not None and min(chosen_length, rejected_length) > 0:
            length_ratios.append(max(chosen_length, rejected_length) / min(chosen_length, rejected_length))
        chosen_outcome = _number(chosen.get("outcome_reward"))
        rejected_outcome = _number(rejected.get("outcome_reward"))
        if (same_source and chosen_source == "outcome"
                and chosen_outcome is not None and rejected_outcome is not None):
            outcome_reward_gaps.append(chosen_outcome - rejected_outcome)
        chosen_success = _outcome_success(chosen)
        rejected_success = _outcome_success(rejected)
        if chosen_success is not None and rejected_success is not None:
            outcome_pairs += 1
            chosen_failed += not chosen_success
            rejected_successful += rejected_success
            same_outcome += chosen_success == rejected_success
            if same_source:
                summary = source_outcomes[chosen_source]
                summary[0] += 1
                summary[1] += not chosen_success
                summary[2] += rejected_success
                summary[3] += chosen_success == rejected_success
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
        prefix + "trace_selected_unique": source_counts["trace"],
        prefix + "outcome_selected_unique": source_counts["outcome"],
        prefix + "label_source_mismatch_pairs": source_mismatch,
    }
    metrics.update(_distribution(value_gaps, prefix + "value_gap", quantiles=(50, 90), maximum=True))
    metrics.update(_distribution(credit_gaps, prefix + "credit_gap", quantiles=(50, 90), minimum=True))
    metrics.update(_distribution(chosen_credits, prefix + "chosen_credit", mean=True, quantiles=()))
    metrics.update(_distribution(rejected_credits, prefix + "rejected_credit", mean=True, quantiles=()))
    metrics.update(_distribution(length_ratios, prefix + "length_ratio", quantiles=(90,)))
    metrics.update(_distribution(outcome_reward_gaps, prefix + "outcome_reward_gap",
                                 quantiles=(50, 90), minimum=True))
    if outcome_pairs:
        metrics[prefix + "chosen_failed_rollout_rate"] = chosen_failed / outcome_pairs
        metrics[prefix + "rejected_successful_rollout_rate"] = rejected_successful / outcome_pairs
        metrics[prefix + "same_outcome_rate"] = same_outcome / outcome_pairs
    for source, (observed, failed, successful, same) in sorted(source_outcomes.items()):
        source_prefix = prefix + source + "/"
        metrics[source_prefix + "outcome_pairs_count"] = observed
        metrics[source_prefix + "chosen_failed_rollout_rate"] = failed / observed
        metrics[source_prefix + "rejected_successful_rollout_rate"] = successful / observed
        metrics[source_prefix + "same_outcome_rate"] = same / observed
    return metrics


def _outcome_success(row):
    """Use explicit verifier truth without treating format rewards as success."""
    if not bool(row.get("outcome_valid", "label_source" not in row)):
        return None
    if "outcome_success" in row:
        value = row["outcome_success"]
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        number = _number(value)
        return bool(number) if number in (0, 1) else None
    reward = _number(row.get("outcome_reward"))
    if reward is None:
        return None
    if "label_source" in row:
        # New paths require verified truth; do not infer success from arbitrary
        # positive process/format rewards when metadata is incomplete.
        return None
    return reward > 0
