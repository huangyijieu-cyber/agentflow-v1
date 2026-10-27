"""Finite, CPU-only diagnostics for TRACE state credit and selected SLIC pairs.

Metrics describe observations and selected pairs; they never alter selection.
The caller must pass selected pairs before any distributed padding or replay.
"""

import math
from collections import defaultdict

import numpy as np


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _numbers(values):
    return [number for value in values if (number := _number(value)) is not None]


def _distribution(
    values,
    prefix,
    *,
    mean=False,
    count=True,
    quantiles=(10, 50, 90),
    minimum=False,
    maximum=False,
):
    """Summarize finite observations, omitting undefined distribution values."""
    finite = _numbers(values)
    metrics = {prefix + "_count": len(finite)} if count else {}
    if not finite:
        return metrics
    # Scale first so even finite observations near float64's limit cannot
    # overflow the intermediate mean or percentile interpolation.
    scale = max(abs(value) for value in finite)
    normalized = np.asarray(finite, dtype=float) / (scale or 1.0)
    if mean:
        metrics[prefix + "_mean"] = float(np.mean(normalized)) * scale
    for quantile in quantiles:
        metrics[prefix + f"_p{quantile}"] = (
            float(np.percentile(normalized, quantile)) * scale
        )
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
    return {
        prefix + "positive_rate": positive / len(finite) if finite else 0.0,
        prefix + "negative_rate": negative / len(finite) if finite else 0.0,
    }


def summarize_states(
    logps,
    values,
    credits,
    turn_types,
    rollout_ids,
    turn_indices,
    config,
    *,
    local_credits=None,
    terminal_fills=None,
    window_blocked=None,
    external_errors=None,
    outcome_valid=None,
    label_sources=None,
    credit_expected_mask=None,
    include_credit=True,
):
    """Summarize state scores and credit without counting terminal NaNs as loss.

    ``rollout_ids`` must identify a rollout within a question; tuple identities
    such as ``(uid, rollout_id)`` are supported without NumPy reshaping.  A
    rollout's largest original turn index is its terminal state.  Missing
    intermediate scores remain in the credit-coverage denominator. New callers
    supply ``credit_expected_mask`` for tool decisions only, so outcome-labelled
    analysis/final rows do not count as missing tool credit. All optional
    observations are diagnostics and cannot change pair selection. IGPO callers
    use ``include_credit=False`` to keep state health metrics without presenting
    a different advantage as TRACE local credit.
    """
    size = len(logps)
    if any(
        len(sequence) != size
        for sequence in (values, credits, turn_types, rollout_ids, turn_indices)
    ):
        raise ValueError("TRACE state metric inputs must have the same length")
    optional = {
        "local_credit": local_credits,
        "terminal_fill": terminal_fills,
        "window_blocked": window_blocked,
        "external_errors": external_errors,
        "outcome_valid": outcome_valid,
        "label_sources": label_sources,
        "credit_expected_mask": credit_expected_mask,
    }
    if any(
        sequence is not None and len(sequence) != size for sequence in optional.values()
    ):
        raise ValueError("TRACE optional state metric inputs must have the same length")
    valid_logps = [
        number
        for value in logps
        if (number := _number(value)) is not None and number <= 0
    ]
    metrics = {
        "trace/turns_total": size,
        "trace/score_coverage": len(valid_logps) / size if size else 0.0,
    }
    metrics.update(_distribution(valid_logps, "trace/gt_logp", mean=True, quantiles=()))
    if external_errors is not None:
        metrics["trace/external_error_turns"] = sum(
            bool(value) for value in external_errors
        )
    if outcome_valid is not None:
        rollout_outcomes = defaultdict(bool)
        for rollout, valid in zip(rollout_ids, outcome_valid):
            # Outputs after the actual evaluated final intentionally have no
            # outcome label; they cannot invalidate a trusted rollout prefix.
            rollout_outcomes[_key(rollout)] |= bool(valid)
        metrics["trace/outcome_invalid_rollouts"] = sum(
            not valid for valid in rollout_outcomes.values()
        )
    if not include_credit:
        return metrics
    metrics.update(_distribution(credits, "trace/credit", mean=True))
    if local_credits is not None:
        metrics.update(_distribution(local_credits, "trace/local_credit", mean=True))
    if terminal_fills is not None:
        metrics.update(_distribution(terminal_fills, "trace/terminal_fill", mean=True))
        metrics["trace/terminal_fill_nonzero_turns"] = sum(
            value != 0 for value in _numbers(terminal_fills)
        )
    if window_blocked is not None:
        metrics["trace/credit_windows_blocked_external"] = sum(
            bool(value) for value in window_blocked
        )
    metrics.update(_credit_classes(credits, "trace/", config))

    if credit_expected_mask is None:
        groups = defaultdict(list)
        for index, (rollout, turn) in enumerate(zip(rollout_ids, turn_indices)):
            groups[_key(rollout)].append((index, _number(turn)))
        terminal = set()
        for observations in groups.values():
            # Invalid turn metadata cannot supply an ordering; use arrival order
            # only to break ties or when no original index can be read.
            terminal.add(
                max(
                    observations,
                    key=lambda item: (
                        item[1] is not None,
                        item[1] if item[1] is not None else 0,
                        item[0],
                    ),
                )[0]
            )
        expected_mask = [index not in terminal for index in range(size)]
    else:
        expected_mask = [bool(value) for value in credit_expected_mask]
    expected = sum(expected_mask)
    available = sum(
        expected_mask[index] and _number(credit) is not None
        for index, credit in enumerate(credits)
    )
    metrics.update(
        {
            "trace/credit_expected_turns": expected,
            "trace/credit_coverage": available / expected if expected else 0.0,
        }
    )
    return metrics


def _config_value(config, name, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(name, default)
    return getattr(config, name, default)


def _correlation(left, right):
    """Finite Pearson correlation, including empty and constant populations."""
    if len(left) < 2:
        return 0.0
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    # Scale before centering to avoid overflow for large finite observations.
    left = left / (float(np.max(np.abs(left))) or 1.0)
    right = right / (float(np.max(np.abs(right))) or 1.0)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    correlation = float(np.dot(left, right) / denominator)
    return min(1.0, max(-1.0, correlation)) if math.isfinite(correlation) else 0.0


def summarize_igpo(rows, annotated, config=None):
    """Describe IGPO rewards and returns without changing eligibility.

    The pure credit helper supplies input-row-aligned arrays. Return components
    already include their configured weights, so they sum to ``advantage``.
    Trusted outcome z-scores are counted once per rollout, while raw IG samples
    retain zero-valued observations. Helper counters are merged under the same
    ``trace/igpo/`` namespace. This function must run before distributed padding.
    """
    fields = (
        "raw_ig",
        "normalized_ig",
        "outcome_z",
        "raw_ig_return",
        "ig_return",
        "terminal_return",
        "advantage",
        "valid",
        "blocked",
        "external_blocked",
        "remaining_turns",
    )
    size = len(rows)
    if any(name not in annotated or len(annotated[name]) != size for name in fields):
        raise ValueError("IGPO metric arrays must align with input rows")
    prefix = "trace/igpo/"
    metrics = {}
    for name, value in annotated.get("metrics", {}).items():
        key = (
            name
            if name.startswith("trace/")
            else "trace/" + name if name.startswith("igpo/") else prefix + name
        )
        number = _number(value)
        if number is not None:
            metrics[key] = number
    tool_rows = [
        index
        for index, row in enumerate(rows)
        if str(row.get("turn_type")) == "tool_plan"
        and bool(row.get("within_final_prefix", True))
    ]
    valid_rows = [
        index
        for index in tool_rows
        if bool(annotated["valid"][index])
        and _number(annotated["advantage"][index]) is not None
    ]
    metrics[prefix + "advantage_coverage"] = (
        len(valid_rows) / len(tool_rows) if tool_rows else 0.0
    )
    for field in ("raw_ig", "normalized_ig"):
        metrics.update(
            _distribution(
                [annotated[field][index] for index in tool_rows],
                prefix + field,
                mean=True,
            )
        )
    for field in ("ig_return", "terminal_return", "advantage"):
        metrics.update(
            _distribution(
                [annotated[field][index] for index in valid_rows],
                prefix + field,
                mean=True,
            )
        )
    outcome_by_rollout = {}
    for index, row in enumerate(rows):
        if (
            bool(row.get("is_final"))
            and bool(row.get("outcome_valid"))
            and _number(annotated["outcome_z"][index]) is not None
        ):
            key = (_key(row.get("uid")), _key(row.get("rollout_id")))
            outcome_by_rollout[key] = annotated["outcome_z"][index]
    metrics.update(
        _distribution(outcome_by_rollout.values(), prefix + "outcome_z", mean=True)
    )

    compared, flipped = 0, 0
    remaining, advantages = [], []
    for index in valid_rows:
        raw_ig = _number(annotated["raw_ig"][index])
        advantage = _number(annotated["advantage"][index])
        # Zero is retained in IG distributions but has no directional sign to
        # flip; only opposite nonzero signs contribute to this diagnostic.
        if raw_ig is not None and raw_ig != 0 and advantage != 0:
            compared += 1
            flipped += (raw_ig > 0) != (advantage > 0)
        distance = _number(annotated["remaining_turns"][index])
        if distance is not None:
            remaining.append(distance)
            advantages.append(advantage)
    metrics[prefix + "sign_compared_turns"] = compared
    metrics[prefix + "raw_ig_advantage_sign_flip_rate"] = (
        flipped / compared if compared else 0.0
    )
    metrics[prefix + "advantage_remaining_turns_corr"] = _correlation(
        advantages, remaining
    )
    return metrics


def summarize_selected_pairs(rows, pairs, config=None):
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
    all_questions = {_key(row["uid"]) for row in rows}
    questions = set()
    value_gaps, credit_gaps, igpo_gaps = [], [], []
    source_counts = defaultdict(int)
    cross_turn = 0
    outcome_pairs = chosen_failed = rejected_successful = same_outcome = 0
    for chosen_index, rejected_index in pairs:
        chosen, rejected = row_map[chosen_index], row_map[rejected_index]
        chosen_source = str(chosen.get("label_source", "trace"))
        rejected_source = str(rejected.get("label_source", "trace"))
        same_source = chosen_source == rejected_source
        if same_source:
            source_counts[chosen_source] += 1
        questions.update((_key(chosen["uid"]), _key(rejected["uid"])))
        cross_turn += chosen.get("turn_index") != rejected.get("turn_index")
        trace_pair = same_source and chosen_source == "trace"
        if trace_pair:
            chosen_value, rejected_value = _number(chosen.get("value")), _number(
                rejected.get("value")
            )
            if chosen_value is not None and rejected_value is not None:
                value_gaps.append(abs(chosen_value - rejected_value))
            mode = _config_value(
                config, "pair_score_mode", chosen.get("pair_score_mode", "trace_fill")
            )
            if mode == "igpo":
                chosen_advantage = _number(chosen.get("igpo_advantage"))
                rejected_advantage = _number(rejected.get("igpo_advantage"))
                if (
                    chosen.get("igpo_valid", False)
                    and rejected.get("igpo_valid", False)
                    and chosen_advantage is not None
                    and rejected_advantage is not None
                ):
                    igpo_gaps.append(chosen_advantage - rejected_advantage)
            else:
                chosen_credit = _number(chosen.get("credit"))
                rejected_credit = _number(rejected.get("credit"))
                if chosen_credit is not None and rejected_credit is not None:
                    credit_gaps.append(chosen_credit - rejected_credit)
            chosen_success = _outcome_success(chosen)
            rejected_success = _outcome_success(rejected)
            if chosen_success is not None and rejected_success is not None:
                outcome_pairs += 1
                chosen_failed += not chosen_success
                rejected_successful += rejected_success
                same_outcome += chosen_success == rejected_success
    prefix = "trace/pair/"
    count = len(pairs)
    metrics = {
        prefix + "selected_unique": count,
        prefix
        + "question_coverage": (
            len(questions) / len(all_questions) if all_questions else 0.0
        ),
        prefix + "cross_turn_rate": cross_turn / count if count else 0.0,
        prefix + "trace_selected_unique": source_counts["trace"],
        prefix + "outcome_selected_unique": source_counts["outcome"],
        prefix + "trace/outcome_pairs_count": outcome_pairs,
    }
    metrics.update(
        _distribution(
            value_gaps,
            prefix + "value_gap",
            count=False,
            quantiles=(50, 90),
            maximum=True,
        )
    )
    metrics.update(
        _distribution(
            credit_gaps,
            prefix + "credit_gap",
            count=False,
            quantiles=(50, 90),
            minimum=True,
        )
    )
    metrics.update(
        _distribution(
            igpo_gaps,
            prefix + "igpo_advantage_gap",
            count=False,
            quantiles=(50, 90),
            minimum=True,
        )
    )
    if outcome_pairs:
        metrics[prefix + "trace/chosen_failed_rollout_rate"] = (
            chosen_failed / outcome_pairs
        )
        metrics[prefix + "trace/rejected_successful_rollout_rate"] = (
            rejected_successful / outcome_pairs
        )
        metrics[prefix + "trace/same_outcome_rate"] = same_outcome / outcome_pairs
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
