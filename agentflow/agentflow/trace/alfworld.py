"""Deterministic ALFWorld PDDL goals and full-state scoring views.

This module intentionally uses only the standard library. It never translates
rules with a language model and never updates facts from a generated action.
The loaded game's problem and domain are authoritative; traj_data only selects
and validates the supported task family.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any


TEMPLATE_VERSION = "alfworld-pddl-controlled-english-v1"
DEFAULT_ANSWER_INSTRUCTION = (
    "Based on the applicable environment rules, the initial state, and the "
    "current full environment state above, describe the current state in "
    "relation to the task goal. Desired goal conditions are not evidence that "
    "they currently hold."
)

PREDICATE_ARITIES = {
    "atlocation": 2, "receptacleatlocation": 2, "objectatlocation": 2,
    "openable": 1, "opened": 1, "inreceptacle": 2,
    "isreceptacleobject": 1, "inreceptacleobject": 2,
    "isreceptacleobjectfull": 1, "wasinreceptacle": 2, "checked": 1,
    "examined": 1, "receptacletype": 2, "cancontain": 2, "objecttype": 2,
    "holds": 2, "holdsany": 1, "holdsanyreceptacleobject": 1, "full": 1,
    "isclean": 1, "cleanable": 1, "ishot": 1, "heatable": 1,
    "iscool": 1, "coolable": 1, "pickupable": 1, "moveable": 1,
    "toggleable": 1, "ison": 1, "istoggled": 1, "sliceable": 1,
    "issliced": 1,
}
TASK_MARKERS = {
    "pick_and_place_simple": set(),
    "pick_two_obj_and_place": {"="},
    "pick_clean_then_place_in_recep": {"cleanable", "isclean"},
    "pick_heat_then_place_in_recep": {"heatable", "ishot"},
    "pick_cool_then_place_in_recep": {"coolable", "iscool"},
    "look_at_obj_in_light": {"holds", "toggleable", "receptacleatlocation", "atlocation"},
}
STATE_MARKERS = {"isclean", "ishot", "iscool"}


class GoalConversionError(ValueError):
    """A source cannot be faithfully represented by the supported converter."""


def normalize_gamefile(value: Any) -> Path:
    """Unpack exactly one real game path; never guess it from an episode index."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise GoalConversionError("extra.gamefile must identify exactly one game")
        value = value[0]
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise GoalConversionError("Missing actual extra.gamefile")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise GoalConversionError(f"Game file does not exist: {path}")
    return path


def parse_pddl(text: str):
    """Parse PDDL S-expressions, preserving symbols while ignoring comments."""
    if not isinstance(text, str) or not text.strip():
        raise GoalConversionError("PDDL source must be nonempty text")
    tokens = re.findall(r"\(|\)|[^\s(),]+", re.sub(r";[^\n]*", "", text))
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            node = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise GoalConversionError("Unbalanced PDDL closing parenthesis")
            stack.pop()
        else:
            if not stack:
                raise GoalConversionError("Unexpected text outside a PDDL expression")
            stack[-1].append(token.lower())
    if stack or len(roots) != 1:
        raise GoalConversionError("PDDL must contain one balanced root expression")
    return roots[0]


def pddl_text(node) -> str:
    return "(" + " ".join(pddl_text(x) if isinstance(x, list) else str(x) for x in node) + ")"


def _sections(root, expected):
    if not isinstance(root, list) or len(root) < 2 or root[0] != "define":
        raise GoalConversionError(f"Expected a PDDL {expected} definition")
    if not isinstance(root[1], list) or len(root[1]) != 2 or root[1][0] != expected:
        raise GoalConversionError(f"Expected a PDDL {expected} definition")
    result = {}
    for node in root[2:]:
        if not isinstance(node, list) or not node:
            raise GoalConversionError("Malformed PDDL section")
        if node[0] == ":action":
            result.setdefault(":actions", []).append(node)
        elif node[0] in result:
            raise GoalConversionError(f"Duplicate PDDL section {node[0]}")
        else:
            result[node[0]] = node[1:]
    return result


def _typed(items, *, variables=False):
    if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
        raise GoalConversionError("Unsupported typed declaration")
    result, pending = [], []
    i = 0
    while i < len(items):
        token = items[i]
        if token in ("-", "#"):
            if not pending or i + 1 >= len(items) or items[i + 1].startswith("?"):
                raise GoalConversionError("Malformed PDDL type annotation")
            result.extend((name, items[i + 1]) for name in pending)
            pending = []
            i += 2
        else:
            if variables != token.startswith("?"):
                raise GoalConversionError(f"Invalid {'variable' if variables else 'object'} {token}")
            pending.append(token)
            i += 1
    result.extend((name, "object") for name in pending)
    names = [name for name, _ in result]
    if len(names) != len(set(names)):
        raise GoalConversionError("Duplicate name in typed declaration")
    return result


def _validate_expr(node, bound=None, *, effect=False):
    bound = set(bound or ())
    if not isinstance(node, list):
        raise GoalConversionError("Expected a predicate or logical expression")
    if not node:
        return
    head = node[0]
    if not isinstance(head, str):
        raise GoalConversionError("A PDDL expression must begin with an operator or predicate name")
    if head in ("and", "or"):
        for child in node[1:]:
            _validate_expr(child, bound, effect=effect)
    elif head == "not":
        if len(node) != 2:
            raise GoalConversionError("not must have exactly one operand")
        _validate_expr(node[1], bound, effect=effect)
    elif head in ("exists", "forall"):
        if len(node) != 3:
            raise GoalConversionError(f"Malformed {head}")
        declarations = _typed(node[1], variables=True)
        _validate_expr(node[2], bound | {x for x, _ in declarations}, effect=effect)
    elif head == "imply":
        if len(node) != 3 or effect:
            raise GoalConversionError("Unsupported implication")
        _validate_expr(node[1], bound)
        _validate_expr(node[2], bound)
    elif head == "when" and effect:
        if len(node) != 3:
            raise GoalConversionError("Malformed conditional effect")
        _validate_expr(node[1], bound)
        _validate_expr(node[2], bound, effect=True)
    elif head == "increase" and effect:
        if len(node) != 3 or node[1] != ["total-cost"] or not isinstance(node[2], str):
            raise GoalConversionError("Only numeric total-cost increases are supported")
        try:
            float(node[2])
        except ValueError as error:
            raise GoalConversionError("Non-numeric action cost") from error
    else:
        arity = 2 if head == "=" else PREDICATE_ARITIES.get(head)
        if arity is None or len(node) != arity + 1 or any(not isinstance(x, str) for x in node[1:]):
            raise GoalConversionError(f"Unsupported predicate or arity: {pddl_text(node)}")
        for argument in node[1:]:
            if argument.startswith("?") and argument not in bound:
                raise GoalConversionError(f"Unbound variable {argument}")


def _alpha_goal(node, mapping=None, counters=None):
    mapping, counters = dict(mapping or {}), counters if counters is not None else {}
    if not node:
        return []
    head = node[0]
    if head in ("exists", "forall"):
        declarations = []
        for variable, kind in _typed(node[1], variables=True):
            counters[kind] = counters.get(kind, 0) + 1
            stem = {"object": "o", "receptacle": "r", "agent": "a", "location": "l"}.get(kind, kind)
            renamed = f"?{stem}{counters[kind]}"
            mapping[variable] = renamed
            declarations.extend([renamed, "-", kind])
        return [head, declarations, _alpha_goal(node[2], mapping, counters)]
    return [head] + [_alpha_goal(child, mapping, counters) if isinstance(child, list)
                     else mapping.get(child, child) for child in node[1:]]


def _pretty_symbol(symbol):
    if symbol.startswith("?"):
        return symbol[1:].upper()
    if symbol.endswith("type"):
        return symbol[:-4]
    return symbol


def _atom_english(node):
    head, args = node[0], [_pretty_symbol(x) for x in node[1:]]
    patterns = {
        "atlocation": "{0} is at location {1}",
        "receptacleatlocation": "receptacle {0} is at location {1}",
        "objectatlocation": "object {0} is at location {1}",
        "inreceptacle": "{0} is in or on receptacle {1}",
        "inreceptacleobject": "{0} is inside movable receptacle object {1}",
        "receptacletype": "{0} is a receptacle of type {1}",
        "objecttype": "{0} is an object of type {1}",
        "holds": "agent {0} is holding {1}",
        "cancontain": "receptacles of type {0} can contain objects of type {1}",
        "wasinreceptacle": "the prior-placement flag for {0} and {1} is set",
        "openable": "{0} is openable", "opened": "{0} is open",
        "isreceptacleobject": "{0} can act as a movable receptacle",
        "isreceptacleobjectfull": "the movable-receptacle-full flag of {0} is set",
        "checked": "the checked flag of {0} is set", "examined": "the examined flag of {0} is set",
        "holdsany": "agent {0} is holding an object",
        "holdsanyreceptacleobject": "agent {0} is holding a movable receptacle object",
        "full": "receptacle {0} is full", "isclean": "{0} is clean",
        "cleanable": "{0} is cleanable", "ishot": "{0} is hot",
        "heatable": "{0} is heatable", "iscool": "{0} is cool",
        "coolable": "{0} is coolable", "pickupable": "{0} is pickupable",
        "moveable": "{0} is moveable", "toggleable": "{0} is switchable",
        "ison": "{0} is on", "istoggled": "the toggle flag of {0} is set",
        "sliceable": "{0} is sliceable", "issliced": "{0} is sliced",
        "=": "{0} and {1} are the same entity",
    }
    if head not in patterns:
        raise GoalConversionError(f"Unsupported English predicate {head}")
    return patterns[head].format(*args)


def describe_expression(node, *, effect=False):
    if not node:
        return "true"
    head = node[0]
    if head in ("and", "or"):
        if not node[1:]:
            return "true" if head == "and" else "false"
        if len(node) == 2:
            return describe_expression(node[1], effect=effect)
        joiner = "; and " if head == "and" else "; or "
        content = joiner.join(describe_expression(x, effect=effect) for x in node[1:])
        return "(" + content + ")" if head == "and" else "at least one of (" + content + ")"
    if head == "not":
        if node[1] and node[1][0] == "=":
            a, b = map(_pretty_symbol, node[1][1:])
            return f"{a} and {b} are different entities"
        return "it is not the case that (" + describe_expression(node[1]) + ")"
    if head in ("exists", "forall"):
        bindings = _typed(node[1], variables=True)
        names = ", ".join(f"{kind} {_pretty_symbol(name)}" for name, kind in bindings)
        phrase = "there exists " if head == "exists" else "for every "
        return phrase + names + " such that (" + describe_expression(node[2], effect=effect) + ")"
    if head == "imply":
        return "if (" + describe_expression(node[1]) + "), then (" + describe_expression(node[2]) + ")"
    if head == "when" and effect:
        return "when the pre-action state satisfies (" + describe_expression(node[1]) + "), set the post-action state so that (" + describe_expression(node[2], effect=True) + ")"
    if head == "increase" and effect:
        return f"increase the action-cost counter by {node[2]}"
    return _atom_english(node)


def _positive_atoms(node, polarity=True):
    if not node:
        return []
    head = node[0]
    if head == "not":
        return _positive_atoms(node[1], not polarity)
    if head in ("exists", "forall"):
        return _positive_atoms(node[2], polarity)
    if head in ("and", "or", "imply"):
        return [atom for child in node[1:] for atom in _positive_atoms(child, polarity)]
    return [node] if polarity else []


def _all_atoms(node):
    if not node:
        return []
    head = node[0]
    if head == "not":
        return _all_atoms(node[1])
    if head in ("exists", "forall"):
        return _all_atoms(node[2])
    if head in ("and", "or", "imply"):
        return [atom for child in node[1:] for atom in _all_atoms(child)]
    return [node]


def _validate_metadata(task_type, params, goal):
    if task_type not in TASK_MARKERS:
        raise GoalConversionError(f"Unsupported ALFWorld task type: {task_type}")
    if not isinstance(params, dict):
        raise GoalConversionError("traj_data.pddl_params must be an object")
    required = ["object_target"] + (["toggle_target"] if task_type == "look_at_obj_in_light" else ["parent_target"])
    for key in required:
        if not isinstance(params.get(key), str) or not params[key].strip():
            raise GoalConversionError(f"Missing pddl_params.{key}")
    if params.get("mrecep_target"):
        raise GoalConversionError("Movable-receptacle goals are outside the six supported task families")
    positive = _positive_atoms(goal)
    predicates = {a[0] for a in positive}
    all_predicates = {a[0] for a in _all_atoms(goal)}
    required_markers = TASK_MARKERS[task_type]
    if not required_markers <= all_predicates:
        raise GoalConversionError(f"Actual Goal does not match task type {task_type}: missing {sorted(required_markers-all_predicates)}")
    expected_state = {"pick_clean_then_place_in_recep": "isclean", "pick_heat_then_place_in_recep": "ishot", "pick_cool_then_place_in_recep": "iscool"}.get(task_type)
    if all_predicates & STATE_MARKERS != ({expected_state} if expected_state else set()):
        raise GoalConversionError("Goal state predicates disagree with task_type")
    wanted_object = params["object_target"].lower() + "type"
    object_variables = {a[1] for a in positive if a[0] == "objecttype" and a[2] == wanted_object}
    if not object_variables:
        raise GoalConversionError("object_target does not occur in the actual Goal")
    if task_type == "look_at_obj_in_light":
        wanted_toggle = params["toggle_target"].lower() + "type"
        toggle_variables = {a[1] for a in positive if a[0] == "objecttype" and a[2] == wanted_toggle}
        if not toggle_variables or not any(a[0] == "holds" and a[2] in object_variables for a in positive):
            raise GoalConversionError("Look goal does not bind the target object to holding")
        if not any(a[0] in ("istoggled", "ison") and a[1] in toggle_variables for a in positive):
            raise GoalConversionError("Look goal has no matching toggle/on condition")
        allowed_types = {wanted_object, wanted_toggle}
    else:
        wanted_parent = params["parent_target"].lower() + "type"
        parent_variables = {a[1] for a in positive if a[0] == "receptacletype" and a[2] == wanted_parent}
        placements = [a for a in positive if a[0] == "inreceptacle" and a[1] in object_variables and a[2] in parent_variables]
        if not placements:
            raise GoalConversionError("parent_target or placement relation disagrees with the actual Goal")
        if expected_state and not any([expected_state, a[1]] in positive for a in placements):
            raise GoalConversionError("Processing and placement must concern the same object")
        if task_type == "pick_two_obj_and_place":
            unequal = []
            def collect(node):
                if isinstance(node, list) and node:
                    if node[0] == "not" and len(node) == 2 and isinstance(node[1], list) and node[1][:1] == ["="]:
                        unequal.append(set(node[1][1:]))
                    for child in node[1:]:
                        if isinstance(child, list):
                            collect(child)
            collect(goal)
            shared = any(a[1] != b[1] and a[2] == b[2] and {a[1], b[1]} in unequal for a in placements for b in placements)
            if not shared:
                raise GoalConversionError("Pick-two Goal must bind distinct objects to the same receptacle")
        allowed_types = {wanted_object}
    if {a[2] for a in positive if a[0] == "objecttype"} - allowed_types:
        raise GoalConversionError("Actual Goal contains object types absent from task metadata")
    sliced = params.get("object_sliced", False)
    if not isinstance(sliced, bool):
        raise GoalConversionError("object_sliced must be a boolean")
    if ("issliced" in all_predicates) != sliced:
        raise GoalConversionError("Slice condition differs between actual Goal and metadata")


def _fact_tuple(value):
    if hasattr(value, "name") and hasattr(value, "arguments"):
        return tuple([str(value.name).lower()] + [str(getattr(x, "name", x)).lower() for x in value.arguments])
    if isinstance(value, dict):
        head = value.get("predicate", value.get("name"))
        args = value.get("arguments", value.get("args"))
        if not isinstance(head, str) or not isinstance(args, (list, tuple)):
            raise GoalConversionError("Unsupported serialized environment fact")
        return tuple([head.lower()] + [str(x.get("name") if isinstance(x, dict) else x).lower() for x in args])
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith("("):
            match = re.fullmatch(r"([^\s(]+)\((.*)\)", text)
            if not match:
                raise GoalConversionError(f"Unsupported fact string: {text}")
            return tuple([match[1].lower()] + [x.strip().split(":", 1)[0].lower() for x in match[2].split(",") if x.strip()])
        node = parse_pddl(text)
        if not node or any(not isinstance(x, str) for x in node):
            raise GoalConversionError("Expected a ground positive fact")
        return tuple(node)
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, str) for x in value):
        return tuple(x.lower() for x in value)
    raise GoalConversionError(f"Unsupported environment fact: {type(value).__name__}")


def normalize_facts(facts) -> list[str]:
    """Canonicalize complete environment facts without inventing missing facts."""
    if facts is None:
        raise GoalConversionError("Full environment facts are missing")
    if hasattr(facts, "tolist"):
        facts = facts.tolist()
    if isinstance(facts, str):
        facts = [facts]
    if not isinstance(facts, (list, tuple, set, frozenset)):
        raise GoalConversionError("Environment facts must be a collection")
    if len(facts) == 1 and isinstance(next(iter(facts)), (list, tuple)):
        child = next(iter(facts))
        if not child or not isinstance(child[0], str) or child[0].strip().startswith("("):
            facts = child
    result = set()
    for fact in facts:
        normalized = _fact_tuple(fact)
        if normalized[0].startswith("new-axiom@"):
            # Internal compiler propositions are not user-domain state facts.
            continue
        arity = PREDICATE_ARITIES.get(normalized[0])
        if arity is None or len(normalized) != arity + 1:
            raise GoalConversionError(f"Unsupported environment predicate: {normalized}")
        if any(x.startswith("?") for x in normalized):
            raise GoalConversionError("Environment facts cannot contain variables")
        result.add(pddl_text(list(normalized)))
    return sorted(result)


def _checked_facts(facts, objects) -> list[str]:
    """Reject facts from a different game instead of silently treating them as evidence.

    TextWorld's compiler-only propositions are removed by normalize_facts;
    every argument of a retained domain proposition must belong to the loaded
    problem or domain constants.
    """
    canonical = normalize_facts(facts)
    for fact in canonical:
        for argument in parse_pddl(fact)[1:]:
            if argument not in objects:
                raise GoalConversionError(f"Environment fact refers to undeclared entity: {argument}")
    return canonical


def _domain_actions(sections):
    actions = {}
    for node in sections.get(":actions", []):
        if len(node) < 2 or not isinstance(node[1], str) or node[1] in actions:
            raise GoalConversionError("Malformed or duplicate domain action")
        fields = {}
        i = 2
        while i < len(node):
            if i + 1 >= len(node) or node[i] in fields:
                raise GoalConversionError("Malformed action field")
            fields[node[i]] = node[i + 1]
            i += 2
        if set(fields) != {":parameters", ":precondition", ":effect"}:
            raise GoalConversionError("Unsupported action fields")
        parameters = _typed(fields[":parameters"], variables=True)
        bound = {name for name, _ in parameters}
        _validate_expr(fields[":precondition"], bound)
        _validate_expr(fields[":effect"], bound, effect=True)
        actions[node[1]] = {"parameters": parameters, "precondition": fields[":precondition"], "effect": fields[":effect"]}
    return actions


def _rules_context(actions):
    lines = [
        "Applicable environment rules (read from this game's PDDL domain):",
        "Only successful environment transitions establish action effects. A proposed or failed action establishes none.",
        "All conditional effects test the PRE-ACTION state; add/delete effects describe the resulting state, not an invented action history.",
        "A set flag may already be true initially. Do not infer when it was set. In particular, the toggle flag is distinct from the current on/off state.",
    ]
    for name, action in sorted(actions.items()):
        parameters = ", ".join(f"{_pretty_symbol(v)}:{kind}" for v, kind in action["parameters"])
        lines.append(f"{name}({parameters}): requires {describe_expression(action['precondition'])}; on success, {describe_expression(action['effect'], effect=True)}.")
    return "\n".join(lines)


def _entity_aliases(objects):
    """Use stable short entity IDs instead of long THOR coordinate identifiers.

    The complete raw-to-alias mapping remains in the saved spec. Type constants
    keep their original identifiers. No entities or facts are discarded.
    """
    result, counters, used = {}, {}, set(objects)
    for name, kind in sorted(objects.items()):
        if kind in ("otype", "rtype") or ("_bar_" not in name and len(name) <= 24):
            result[name] = name
            continue
        stem = name.split("_bar_", 1)[0] if "_bar_" in name else kind
        stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or kind
        if "basin" in name and not stem.endswith("basin"):
            stem += "basin"
        counters[stem] = counters.get(stem, 0) + 1
        alias = f"{stem}_{counters[stem]}"
        while alias in used:
            counters[stem] += 1
            alias = f"{stem}_{counters[stem]}"
        result[name] = alias
        used.add(alias)
    return result


def _alias_expression(node, aliases):
    return [_alias_expression(item, aliases) if isinstance(item, list) else aliases.get(item, item)
            for item in node]


def load_goal_spec(gamefile) -> dict:
    """Load and validate an actual game and produce a fixed English goal."""
    game_path = normalize_gamefile(gamefile)
    traj_path = game_path.with_name("traj_data.json")
    if not traj_path.is_file():
        raise GoalConversionError(f"Missing goal metadata: {traj_path}")
    game_bytes, traj_bytes = game_path.read_bytes(), traj_path.read_bytes()
    try:
        game, trajectory = json.loads(game_bytes), json.loads(traj_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GoalConversionError("Expected JSON game.tw-pddl and traj_data.json") from error
    if not isinstance(game, dict) or not isinstance(trajectory, dict):
        raise GoalConversionError("Game and trajectory metadata must be JSON objects")
    domain_text, problem_text = game.get("pddl_domain"), game.get("pddl_problem")
    domain_root, problem_root = parse_pddl(domain_text), parse_pddl(problem_text)
    domain = _sections(domain_root, "domain")
    problem = _sections(problem_root, "problem")
    if set(domain) - {":requirements", ":types", ":constants", ":predicates", ":functions", ":actions"}:
        raise GoalConversionError("Unsupported domain section")
    if set(problem) - {":domain", ":objects", ":init", ":goal", ":metric"}:
        raise GoalConversionError("Unsupported problem section")
    if problem.get(":domain") != [domain_root[1][1]]:
        raise GoalConversionError("Problem and domain names do not match")
    for declaration in domain.get(":predicates", []):
        if not isinstance(declaration, list) or not declaration:
            raise GoalConversionError("Malformed predicate declaration")
        arity = PREDICATE_ARITIES.get(declaration[0])
        if arity is None or len(_typed(declaration[1:], variables=True)) != arity:
            raise GoalConversionError(f"Unsupported declared predicate: {pddl_text(declaration)}")
    goal_section = problem.get(":goal")
    if not goal_section or len(goal_section) != 1:
        raise GoalConversionError("Problem must have exactly one Goal expression")
    goal = goal_section[0]
    _validate_expr(goal)
    goal = _alpha_goal(goal)
    task_type, params = trajectory.get("task_type"), trajectory.get("pddl_params")
    _validate_metadata(task_type, params, goal)
    objects = dict(_typed(problem.get(":objects", [])))
    for symbol, kind in _typed(domain.get(":constants", [])):
        if symbol in objects and objects[symbol] != kind:
            raise GoalConversionError("Conflicting object/constant type")
        objects[symbol] = kind
    for atom in _all_atoms(goal):
        for argument in atom[1:]:
            if not argument.startswith("?") and argument not in objects:
                raise GoalConversionError(f"Goal refers to undeclared constant: {argument}")
    types = dict(_typed(domain.get(":types", [])))
    initial, initial_negative, initial_numeric = [], [], []
    if ":init" not in problem:
        raise GoalConversionError("Problem is missing its initial state")
    for node in problem[":init"]:
        if isinstance(node, list) and node[:1] == ["="] and len(node) == 3 and isinstance(node[1], list):
            initial_numeric.append(pddl_text(node))
        elif isinstance(node, list) and node[:1] == ["not"]:
            _validate_expr(node)
            initial_negative.append(pddl_text(node[1]))
        else:
            _validate_expr(node)
            initial.append(pddl_text(node))
    initial = _checked_facts(initial, objects)
    initial_negative = _checked_facts(initial_negative, objects)
    if set(initial) & set(initial_negative):
        raise GoalConversionError("Initial state contains contradictory facts")
    actions = _domain_actions(domain)
    aliases = _entity_aliases(objects)
    clause = describe_expression(_alias_expression(goal, aliases))
    # A declarative target is scored under the current state. A normative
    # sentence such as "success requires ..." would remain true even when the
    # required state is not supported by the evidence.
    response = clause[0].upper() + clause[1:] + "."
    spec = {
        "schema_version": 1, "template_version": TEMPLATE_VERSION,
        "gamefile": str(game_path), "goal_source": str(traj_path), "traj_data_path": str(traj_path),
        "goal_task_type": task_type, "task_type": task_type, "pddl_params": params,
        "goal_response": response, "goal_ast": goal, "raw_goal": pddl_text(goal_section[0]),
        "objects": objects, "types": types, "entity_aliases": aliases,
        "initial_facts": initial, "initial_negative_facts": sorted(initial_negative),
        "initial_numeric_assignments": sorted(initial_numeric), "domain_actions": actions,
        "source_sha256": {"gamefile": hashlib.sha256(game_bytes).hexdigest(),
                          "traj_data": hashlib.sha256(traj_bytes).hexdigest(),
                          "pddl_domain": hashlib.sha256(domain_text.encode()).hexdigest(),
                          "pddl_problem": hashlib.sha256(problem_text.encode()).hexdigest()},
        "scoring_mode": "full_state", "goal_validation_status": "actual_pddl_goal_and_metadata_checked",
        "answer_instruction": DEFAULT_ANSWER_INSTRUCTION,
    }
    spec["rules_context"] = _rules_context(actions)
    spec["initial_context"] = format_facts(initial, aliases=aliases)
    stable = {"template_version": TEMPLATE_VERSION, "sources": spec["source_sha256"], "goal": goal, "response": response}
    spec["goal_spec_hash"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return spec


def _is_type(kind, expected, types):
    seen = set()
    while kind not in seen:
        if kind == expected:
            return True
        seen.add(kind)
        if kind not in types or types[kind] == kind:
            return False
        kind = types[kind]
    return False


def evaluate_goal(spec, facts) -> bool:
    """Evaluate the retained formal Goal on complete, closed-world facts."""
    true_facts = {tuple(parse_pddl(item)) for item in _checked_facts(facts, spec["objects"])}
    objects, types = spec["objects"], spec.get("types", {})

    def evaluate(node, bindings):
        if not node:
            return True
        head = node[0]
        if head == "and":
            return all(evaluate(x, bindings) for x in node[1:])
        if head == "or":
            return any(evaluate(x, bindings) for x in node[1:])
        if head == "not":
            return not evaluate(node[1], bindings)
        if head == "imply":
            return not evaluate(node[1], bindings) or evaluate(node[2], bindings)
        if head in ("exists", "forall"):
            variables = _typed(node[1], variables=True)
            choices = [[name for name, kind in objects.items() if _is_type(kind, expected, types)] for _, expected in variables]
            def outcomes():
                for assignment in itertools.product(*choices):
                    extended = dict(bindings)
                    extended.update(zip((name for name, _ in variables), assignment))
                    yield evaluate(node[2], extended)
            return any(outcomes()) if head == "exists" else all(outcomes())
        grounded = [bindings.get(x, x) for x in node[1:]]
        if head == "=":
            return grounded[0] == grounded[1]
        return tuple([head] + grounded) in true_facts

    return bool(evaluate(spec["goal_ast"], {}))


def format_facts(facts, *, aliases=None) -> str:
    """Compact, canonical ground facts; the enclosing view explains semantics."""
    grouped, aliases = {}, aliases or {}
    for text in normalize_facts(facts):
        predicate, *arguments = parse_pddl(text)
        grouped.setdefault(predicate, []).append("(" + ",".join(aliases.get(x, x) for x in arguments) + ")")
    return "\n".join(f"{predicate}: " + "; ".join(values) for predicate, values in sorted(grouped.items())) or "No positive domain facts."


def render_full_state(spec, facts, *, observation=None, action_history=None,
                      history=None, initial_facts=None) -> str:
    """Render the selected privileged/full-state diagnostic scoring view."""
    current = _checked_facts(facts, spec["objects"])
    initial = _checked_facts(spec["initial_facts"] if initial_facts is None else initial_facts, spec["objects"])
    aliases = spec.get("entity_aliases", {})
    added, removed = sorted(set(current) - set(initial)), sorted(set(initial) - set(current))
    lines = [
        "Scoring view: FULL ENVIRONMENT STATE (privileged prior information; not the actor's observation-only input).",
        spec["rules_context"],
        "Ground facts below use predicate(argument,...) tuples. Stable short IDs identify concrete entities, not existential variables in the target description. The state is complete for supported domain predicates; absent ground facts are false. Numeric action costs are not goal conditions.",
        "Initial state (before this episode's first action):\n" + format_facts(initial, aliases=aliases),
        "Current full environment state is exactly the initial state with the following additions and removals. Every other initial fact is unchanged; this is a lossless state representation, not a recent-history window.",
        "Currently true facts added to the initial state:\n" + (format_facts(added, aliases=aliases) if added else "None."),
        "Initial facts that are now false:\n" + (format_facts(removed, aliases=aliases) if removed else "None."),
    ]
    if observation is not None:
        lines.append("Current observation (additional textual evidence):\n" + str(observation))
    selected_history = action_history if action_history is not None else history
    if selected_history:
        lines.append("Recorded environment action history:\n" + (selected_history if isinstance(selected_history, str) else json.dumps(selected_history, ensure_ascii=False, sort_keys=True)))
    lines.append("The desired goal is not an observed achievement. A true initial flag does not prove an action happened during this episode.")
    return "\n\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="Convert one actual game without loading ALFWorld or a model")
    convert.add_argument("--gamefile", required=True)
    convert.add_argument("--output")
    args = parser.parse_args(argv)
    spec = load_goal_spec(args.gamefile)
    content = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    try:
        main()
    except GoalConversionError as error:
        raise SystemExit(str(error))
