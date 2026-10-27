"""Pure IGPO-style rewards and full-suffix advantages for SLIC pair selection.

The frozen-reference GT scores are supplied by the trainer.  This module does
not alter GRPO advantages, losses, or rollout generation.  Rewards are grouped
by question, with one outcome observation per trusted rollout and one IG
observation per unique tool transition whose future suffix is complete.
"""

from collections import defaultdict
import math
from collections.abc import Mapping


_FINAL_TYPES = {"final_answer", "direct", "base"}
_NUMBER_FIELDS = ("gt_logp", "value", "outcome_reward")
_BOOLEAN_FIELDS = ("valid", "external_error", "is_final", "outcome_valid")


def _finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _turn(value):
    if _finite(value) and float(value).is_integer() and float(value) >= 0:
        return int(value)
    return None


def _config_value(config, name, default):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _freeze(value):
    """Compare metadata without depending on array truth-value semantics."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, "tolist"):
        return _freeze(value.tolist())
    if _finite(value):
        return float(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return repr(value)


def _signature(row):
    # Physical row indices differ for distributed padding copies. Everything
    # that can affect the reward, quality, or generated action must agree.
    fields = [(name, _freeze(row.get(name))) for name in _NUMBER_FIELDS]
    fields.extend((name, bool(row.get(name, name == "valid"))) for name in _BOOLEAN_FIELDS)
    fields.extend((name, _freeze(row.get(name)))
                  for name in ("turn_type", "action_key", "response_key"))
    return tuple(fields)


def _stats(values):
    """Population statistics, avoiding an overflowing sum before division."""
    count = len(values)
    if not count:
        return math.nan, math.nan
    try:
        mean = math.fsum(value / count for value in values)
        differences = [value - mean for value in values]
        scale = max(abs(value) for value in differences)
        std = (scale * math.sqrt(math.fsum((value / scale) ** 2
                                         for value in differences) / count)
               if scale else 0.0)
    except (OverflowError, ValueError, ZeroDivisionError):
        return math.nan, math.nan
    return mean, std


def compute_igpo_advantages(rows, config=None):
    """Return row-aligned IG rewards and full-future advantages.

    Required row metadata is described in ``docs/trace-2b-design.md``. A trusted
    final requires its real final flag, final decision type, response quality,
    explicit outcome validity and a finite verifier reward. ``valid`` on input
    is action/response quality, independent of credit sign or pair thresholds.

    Duplicate physical copies of the same (uid, rollout, original turn) count
    once. Conflicting copies invalidate that turn and dependent prefixes. A
    missing/invalid transition is never removed to join the surrounding states.
    Analysis receives no IG reward or advantage; an intervening valid analysis
    still occupies its original position for discount distances.

    ``ig_return`` and ``terminal_return`` include their respective configured
    weights. Their sum is ``advantage``. ``raw_ig_return`` is the unnormalized,
    unweighted discounted IG sum for an optional conservative pair guard.
    """
    gamma = _config_value(config, "igpo_gamma", 1.0)
    ig_weight = _config_value(config, "igpo_ig_weight", 1.0)
    outcome_weight = _config_value(config, "igpo_outcome_weight", 1.0)
    epsilon = _config_value(config, "igpo_norm_epsilon", 1e-6)
    std_min = _config_value(config, "igpo_ig_std_min", 1e-3)
    min_rollouts = _config_value(config, "igpo_min_rollouts", 2)
    for name, value in (("igpo_gamma", gamma), ("igpo_ig_weight", ig_weight),
                        ("igpo_outcome_weight", outcome_weight),
                        ("igpo_norm_epsilon", epsilon), ("igpo_ig_std_min", std_min)):
        if not _finite(value):
            raise ValueError(f"trace.{name} must be finite")
    if (not 0 < gamma <= 1 or min(ig_weight, outcome_weight) < 0
            or max(ig_weight, outcome_weight) == 0 or std_min <= 0 or epsilon <= 0):
        raise ValueError("Invalid IGPO discount, weight, standard-deviation floor, or epsilon")
    if isinstance(min_rollouts, bool) or not isinstance(min_rollouts, int) or min_rollouts < 2:
        raise ValueError("trace.igpo_min_rollouts must be an integer of at least two")

    size = len(rows)
    numeric_names = ("raw_ig", "normalized_ig", "outcome_z", "raw_ig_return",
                     "ig_return", "terminal_return", "advantage", "remaining_turns")
    result = {name: [math.nan] * size for name in numeric_names}
    result.update({name: [False] * size for name in ("valid", "blocked", "external_blocked")})
    metrics = dict.fromkeys(("groups_total", "groups_valid", "groups_insufficient_rollouts",
                            "groups_insufficient_ig", "groups_low_ig_std", "groups_invalid_stats",
                            "rollouts_total", "rollouts_trusted", "duplicate_rows",
                            "conflicting_turns", "returns_blocked", "returns_blocked_external",
                            "valid_returns"), 0)
    result["metrics"] = metrics
    grouped = defaultdict(list)
    for position, row in enumerate(rows):
        if "uid" not in row or "rollout_id" not in row:
            raise ValueError("IGPO rows require uid and rollout_id")
        grouped[(str(row["uid"]), str(row["rollout_id"]))].append(position)
    metrics["rollouts_total"] = len(grouped)
    tool_uids = {str(row["uid"]) for row in rows if row.get("turn_type") == "tool_plan"}
    metrics["groups_total"] = len(tool_uids)
    populations = defaultdict(list)
    terminal_populations = defaultdict(list)
    rollout_info = {}

    for key, positions in grouped.items():
        by_turn = defaultdict(list)
        malformed = False
        for position in positions:
            original_turn = _turn(rows[position].get("turn_index"))
            if original_turn is None:
                malformed = True
            else:
                by_turn[original_turn].append(position)
        if malformed:
            for position in positions:
                result["blocked"][position] = rows[position].get("turn_type") == "tool_plan"
            continue
        canonical, conflicts, repeated = [], set(), set()
        for original_turn, copies in sorted(by_turn.items()):
            canonical.append(copies[0])
            metrics["duplicate_rows"] += len(copies) - 1
            if any(_signature(rows[position]) != _signature(rows[copies[0]]) for position in copies[1:]):
                conflicts.add(copies[0])
                metrics["conflicting_turns"] += 1
            for position in copies[1:]:
                result["blocked"][position] = rows[position].get("turn_type") == "tool_plan"

        # Only successful, valid attempts enter action dedup. A network-failed
        # attempt cannot make its recovered retry a duplicate.
        seen_actions = set()
        for position in canonical:
            row = rows[position]
            if (row.get("turn_type") != "tool_plan" or position in conflicts
                    or not row.get("valid", True) or row.get("external_error", False)):
                continue
            action = row.get("action_key")
            if action is not None:
                identity = _freeze(action)
                if identity in seen_actions:
                    repeated.add(position)
                else:
                    seen_actions.add(identity)

        final_rows = [position for position in canonical if rows[position].get("is_final", False)]
        final = final_rows[0] if len(final_rows) == 1 else None
        if final is not None and rows[final].get("turn_type") not in _FINAL_TYPES:
            final = None
        final_offset = canonical.index(final) if final is not None else len(canonical)
        trusted = (final is not None and final not in conflicts
                   and rows[final].get("valid", True)
                   and not rows[final].get("external_error", False)
                   and rows[final].get("outcome_valid", False)
                   and _finite(rows[final].get("outcome_reward")))
        if trusted:
            terminal_populations[key[0]].append((key, float(rows[final]["outcome_reward"])))
            metrics["rollouts_trusted"] += 1

        # Capture real immediate rewards even for prefixes blocked later. They
        # remain diagnostic values and never enter the normalization population.
        for offset, position in enumerate(canonical[:final_offset]):
            row = rows[position]
            if row.get("turn_type") != "tool_plan" or offset + 1 >= len(canonical):
                continue
            following = canonical[offset + 1]
            successor = rows[following]
            local_valid = (position not in conflicts and following not in conflicts
                           and position not in repeated and row.get("valid", True)
                           and not row.get("external_error", False)
                           and _turn(successor.get("turn_index")) == _turn(row.get("turn_index")) + 1
                           and all(_finite(item.get("gt_logp")) and float(item["gt_logp"]) <= 0
                                   and _finite(item.get("value")) for item in (row, successor)))
            if local_valid:
                delta = float(successor["gt_logp"]) - float(row["gt_logp"])
                if _finite(delta):
                    result["raw_ig"][position] = delta

        clean_suffix = {}
        raw_tail = 0.0
        clean = bool(trusted)
        external = bool(final is not None and rows[final].get("external_error", False))
        for offset in range(final_offset - 1, -1, -1):
            position = canonical[offset]
            row = rows[position]
            external = external or bool(row.get("external_error", False))
            following = canonical[offset + 1] if offset + 1 < len(canonical) else None
            continuous = (following is not None and _turn(rows[following].get("turn_index"))
                          == _turn(row.get("turn_index")) + 1)
            quality = (row.get("valid", True) and position not in conflicts
                       and position not in repeated and not row.get("external_error", False))
            kind = row.get("turn_type")
            clean = bool(clean and continuous and quality
                         and (kind == "analysis" or (kind == "tool_plan"
                                                    and _finite(result["raw_ig"][position]))))
            if kind == "tool_plan":
                result["blocked"][position] = not clean
                result["external_blocked"][position] = bool(external)
                if final is not None:
                    result["remaining_turns"][position] = _turn(rows[final]["turn_index"]) - _turn(row["turn_index"])
                if clean:
                    raw_tail = float(result["raw_ig"][position]) + gamma * raw_tail
                    if not _finite(raw_tail):
                        clean = False
                        result["blocked"][position] = True
                    else:
                        result["raw_ig_return"][position] = raw_tail
                        populations[key[0]].append((key, position))
            elif clean:
                raw_tail *= gamma
            clean_suffix[position] = clean
        rollout_info[key] = (canonical, final, final_offset, bool(trusted), clean_suffix)

    # Outcome normalization uses all trusted outcomes, including rollouts with
    # no tools. Its one-vote-per-rollout denominator never depends on length.
    outcome_z = {}
    for uid, observations in terminal_populations.items():
        mean, std = _stats([reward for _, reward in observations])
        if not (_finite(mean) and _finite(std)):
            continue
        for key, reward in observations:
            score = (reward - mean) / (std + epsilon) if std > 0 else 0.0
            if _finite(score):
                outcome_z[key] = score
                canonical, _, final_offset, _, _ = rollout_info[key]
                for position in canonical[:final_offset + 1]:
                    result["outcome_z"][position] = score

    valid_groups = set()
    for uid in tool_uids:
        observations = populations[uid]
        if len({key for key, _ in observations}) < min_rollouts:
            metrics["groups_insufficient_rollouts"] += 1
            continue
        if len(observations) < 2:
            metrics["groups_insufficient_ig"] += 1
            continue
        mean, std = _stats([result["raw_ig"][position] for _, position in observations])
        if not (_finite(mean) and _finite(std)):
            metrics["groups_invalid_stats"] += 1
            continue
        if std <= 0 or std < std_min:
            metrics["groups_low_ig_std"] += 1
            continue
        normalized = [(position, (result["raw_ig"][position] - mean) / (std + epsilon))
                      for _, position in observations]
        if not all(_finite(score) for _, score in normalized):
            metrics["groups_invalid_stats"] += 1
            continue
        valid_groups.add(uid)
        metrics["groups_valid"] += 1
        for position, score in normalized:
            result["normalized_ig"][position] = score

    for key, (canonical, final, final_offset, trusted, clean_suffix) in rollout_info.items():
        if not trusted or key[0] not in valid_groups or key not in outcome_z:
            for position in canonical[:final_offset]:
                if rows[position].get("turn_type") == "tool_plan":
                    result["blocked"][position] = True
            continue
        ig_tail = 0.0
        terminal_tail = outcome_weight * outcome_z[key]
        for position in reversed(canonical[:final_offset]):
            if not clean_suffix.get(position, False):
                ig_tail = terminal_tail = math.nan
                continue
            row = rows[position]
            if row.get("turn_type") == "analysis":
                ig_tail *= gamma
                terminal_tail *= gamma
                continue
            ig_tail = ig_weight * result["normalized_ig"][position] + gamma * ig_tail
            terminal_tail *= gamma
            advantage = ig_tail + terminal_tail
            if not all(_finite(value) for value in (ig_tail, terminal_tail, advantage)):
                result["blocked"][position] = True
                ig_tail = terminal_tail = math.nan
                continue
            result["ig_return"][position] = ig_tail
            result["terminal_return"][position] = terminal_tail
            result["advantage"][position] = advantage
            result["valid"][position] = True

    metrics["returns_blocked"] = sum(result["blocked"])
    metrics["returns_blocked_external"] = sum(result["external_blocked"])
    metrics["valid_returns"] = sum(result["valid"])
    return result
