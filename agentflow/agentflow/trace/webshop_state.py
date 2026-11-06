"""Read-only scoring views of native WebShop state, without native reward calls.

Catalog privilege is restricted to the current item. Search-result rows are read
from the actual HTML in its product-link order; they never borrow catalog facts.
The actor's observation, available actions, and memory are not rewritten here.
"""
from __future__ import annotations

from collections.abc import Mapping
from html.parser import HTMLParser
import math
import numbers
import re
from urllib.parse import urlsplit

from .webshop_goal import ConversionError, SCORING_MODE, TEMPLATE, canonical_json, json_value, sha256_json


STATE_TEMPLATE_VERSION = "webshop-native-state-v1"
RECENT_ACTION_HISTORY_LENGTH = 4
DEFAULT_ANSWER_INSTRUCTION = (
    "Describe the current candidate product state in relation to the shopping task. "
    "Required conditions are desired conditions, and are not evidence that they currently hold."
)
_NEUTRAL_TERMINAL_OBSERVATION = "The checkout page is displayed."
_OUTCOME_TEXT = re.compile(r"\b(?:reward|rewards|score|scores|success|successful|verbose_info)\b", re.I)
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


class _Node:
    def __init__(self, tag="document", attrs=(), parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children = []


class _PageParser(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = self.current = _Node()
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, attrs, self.current)
        self.current.children.append(node)
        if tag not in _VOID_TAGS:
            self.current = node

    def handle_startendtag(self, tag, attrs):
        self.current.children.append(_Node(tag, attrs, self.current))

    def handle_endtag(self, tag):
        node = self.current
        while node.parent is not None:
            if node.tag == tag:
                self.current = node.parent
                return
            node = node.parent

    def handle_data(self, data):
        self.current.children.append(data)


def _hidden(node):
    style = re.sub(r"\s+", "", str(node.attrs.get("style", "")).lower())
    identity = str(node.attrs.get("id", "")).lower()
    return (node.tag in {"head", "script", "style", "title", "meta"}
            or "hidden" in node.attrs or str(node.attrs.get("aria-hidden", "")).lower() == "true"
            or "display:none" in style or "visibility:hidden" in style
            or identity == "instruction-text" or "reward" in identity
            or identity in {"goal", "goal-asin", "goal-options", "goal-attrs", "goal-price",
                            "goal-instruction-text", "goal-category", "goal-pc", "goal-query"})


def _visible(node):
    while node is not None:
        if _hidden(node):
            return False
        node = node.parent
    return True


def _nodes(node):
    yield node
    for child in node.children:
        if isinstance(child, _Node):
            yield from _nodes(child)


def _text(node):
    if _hidden(node):
        return ""
    pieces = []
    for child in node.children:
        text = _text(child) if isinstance(child, _Node) else child.strip()
        if text:
            pieces.append(text)
    return " ".join(pieces)


def _classes(node):
    return str(node.attrs.get("class", "")).split()


def _class_text(node, name):
    return next((_text(child) for child in _nodes(node)
                 if name in _classes(child) and _visible(child)), "")


def _visible_candidates(root):
    candidates = []
    for link in _nodes(root):
        if "product-link" not in _classes(link) or not _visible(link):
            continue
        identity = _text(link)
        if not identity:
            continue
        card = link.parent
        while card is not None and not ({"searched-product", "list-group-item"} & set(_classes(card))):
            card = card.parent
        # Native cards have searched-product. A link without such a card still
        # establishes its visible ID, but cannot borrow fields from other rows.
        candidate = {"candidate_id": identity}
        if card is not None:
            for name, css in (("title", "product-title"), ("display_price", "product-price")):
                value = _class_text(card, css)
                if value:
                    candidate[name] = value
        candidates.append(candidate)
    return candidates


def sanitize_observation(observation, *, terminal=False):
    """Remove outcome-bearing pages instead of leaking a score through text.

    Native terminal HTML also contains hidden target identity and comparison
    details. A neutral terminal description makes all of those absent by design.
    Clean pre-action native observations are otherwise kept verbatim in history.
    """
    if terminal:
        return _NEUTRAL_TERMINAL_OBSERVATION
    if not isinstance(observation, str):
        raise ConversionError("Native WebShop observation must be a string")
    if re.search(r"<\s*(?:html|body|div|h[1-6]|p|a|button|input)\b", observation, re.I):
        observation = _text(_PageParser(observation).root)
    if _OUTCOME_TEXT.search(observation):
        return _NEUTRAL_TERMINAL_OBSERVATION
    return observation


def _page_kind(url):
    pieces = urlsplit(url).path.strip("/").split("/")
    return pieces[0] if pieces[0] in {"search_results", "item_page", "item_sub_page", "done"} else "search"


def _selected_options(session):
    options = session.get("options", {})
    if not isinstance(options, Mapping):
        raise ConversionError("Native selected options must be an object")
    return json_value(options)


def _current_item(server, identity, selected):
    items, prices = server.product_item_dict, server.product_prices
    if identity not in items:
        raise ConversionError("Native current item is absent from product_item_dict")
    product = items[identity]
    price = prices.get(identity)
    if isinstance(price, bool) or not isinstance(price, numbers.Real) or not math.isfinite(price) or price < 0:
        raise ConversionError("Native current item has no finite exact frozen price")
    result = {
        "candidate_id": identity,
        "title": product.get("Title", product.get("name", "")),
        "attributes": product.get("Attributes", []),
        "bullet_points": product.get("BulletPoints", []),
        "description": product.get("Description", ""),
        "available_options": product.get("options", {}),
        "selected_options": selected,
        "price_usd": price,
    }
    for native, output in (("Price", "display_price"), ("category", "category"),
                           ("query", "query"), ("product_category", "product_category")):
        if native in product:
            result[output] = product[native]
    return json_value(result)


def capture_webshop_state(native_env, *, previous_snapshot=None):
    """Deep-copy authoritative native state after reset or one completed action.

    Passing a prior snapshot only supports a terminal wrapper that discarded the
    last item fields. A search or navigation state always uses its own cleared
    selection, never a stale previous item.
    """
    native_env = getattr(native_env, "unwrapped", native_env)
    server = native_env.server
    try:
        session = server.user_sessions[native_env.session]
    except (AttributeError, KeyError, TypeError) as error:
        raise ConversionError("Cannot locate the actual native WebShop session") from error
    if not isinstance(session, Mapping):
        raise ConversionError("Native WebShop session must be an object")
    state = native_env.state
    if not isinstance(state, Mapping) or not isinstance(state.get("html"), str) or not isinstance(state.get("url"), str):
        raise ConversionError("Native WebShop state must include html and url strings")
    kind = _page_kind(state["url"])
    terminal = kind == "done" or session.get("done") is True
    selected = _selected_options(session)
    item = None
    identity = session.get("asin")
    if kind in {"item_page", "item_sub_page", "done"} or terminal:
        if isinstance(identity, str) and identity:
            item = _current_item(server, identity, selected)
        elif terminal and isinstance(previous_snapshot, Mapping) and previous_snapshot.get("current_item") is not None:
            item = json_value(previous_snapshot["current_item"])
            if "options" not in session:
                selected = json_value(item["selected_options"])
            item["selected_options"] = json_value(selected)
    if terminal and item is None:
        raise ConversionError("Terminal WebShop state has no purchased candidate snapshot")
    page = _PageParser(state["html"])
    snapshot = {
        "state_template_version": STATE_TEMPLATE_VERSION,
        "page_kind": "done" if terminal else kind,
        "terminal": terminal,
        "search_keywords": session.get("keywords"),
        "search_page": session.get("page"),
        "visible_candidates": _visible_candidates(page.root) if kind == "search_results" and not terminal else [],
        "current_item": item,
        "selected_options": selected if item is not None else {},
        "observation": sanitize_observation(state["html"], terminal=terminal),
    }
    return json_value(snapshot)


def format_actor_history(records, history_length=RECENT_ACTION_HISTORY_LENGTH):
    """Keep native pre-action observations, parsed actions, and global numbering."""
    if records is None:
        records = []
    if not isinstance(records, (list, tuple)):
        raise ConversionError("WebShop history must contain native memory records")
    if isinstance(history_length, str):
        value = history_length.strip()
        if value.lower() in {"none", "null"}:
            history_length = None
        elif re.fullmatch(r"\+?[0-9]+", value):
            history_length = int(value)
    if history_length is not None and (type(history_length) is not int or history_length < 0):
        raise ConversionError("WebShop history_length must be None or a nonnegative integer")
    for record in records:
        if (not isinstance(record, Mapping) or not isinstance(record.get("text_obs"), str)
                or not isinstance(record.get("action"), str)):
            raise ConversionError("WebShop history requires native text_obs and parsed action strings")
    recent = records if history_length is None else records[-history_length:] if history_length else []
    start = len(records) - len(recent)
    return "\n".join(
        f"[Observation {start + index + 1}: '{sanitize_observation(record['text_obs'])}', "
        f"Action {start + index + 1}: '{record['action']}']"
        for index, record in enumerate(recent)
    )


def render_webshop_state(spec, snapshot, *, actor_history=None, history_length=RECENT_ACTION_HISTORY_LENGTH):
    """Render a scoring-only context, never injecting reference identity/title."""
    if not isinstance(spec, Mapping) or not isinstance(snapshot, Mapping):
        raise ConversionError("WebShop scoring requires a bound goal spec and native state snapshot")
    requirements = {"shopping_instruction": spec["instruction_text"], "required_attributes": spec["attributes"]}
    if spec.get("price_constraint", {}).get("enabled"):
        requirements["price_limit_usd"] = spec["price_upper"]
    if spec.get("goal_options"):
        requirements["required_selected_options"] = spec["goal_options"]
    observation = sanitize_observation(snapshot.get("observation", ""), terminal=snapshot.get("terminal") is True)
    sections = [
        "WebShop candidate-state assessment.",
        "All required conditions must hold for the same current candidate product. "
        "Task requirements describe desired conditions, not observed facts. "
        "Search-result candidates are separate from the current item. "
        "Available option values are possibilities; selected_options records the actual choices. "
        "Checkout status is interface state and does not establish any product condition.",
        "Shopping task requirements (desired conditions):\n" + canonical_json(requirements),
        "Current page: " + str(snapshot["page_kind"]),
        "Current search-result candidates (visible page information only, in page order):\n"
        + canonical_json(snapshot.get("visible_candidates", [])),
        "Current candidate item (native product facts and exact price; null means no item is open):\n"
        + canonical_json(snapshot.get("current_item")),
        "Current page observation:\n" + observation,
    ]
    history = format_actor_history(actor_history, history_length)
    if history:
        sections.append("Recent native pre-action observations and parsed actions:\n" + history)
    sections.append(DEFAULT_ANSWER_INSTRUCTION)
    return "\n\n".join(sections)


def scoring_metadata(spec, binding_metadata=None):
    metadata = dict(binding_metadata or {})
    for key in ("goal_sha256", "manifest_sha256", "task_key"):
        if key not in metadata and key in spec:
            metadata[key] = spec[key]
    metadata.update({"scoring_mode": SCORING_MODE, "template_version": TEMPLATE,
                     "state_template_version": STATE_TEMPLATE_VERSION})
    return metadata


def state_id(snapshot):
    """Hash the captured state without random native session IDs or raw URLs."""
    return sha256_json(snapshot)


def action_key(action):
    if not isinstance(action, str) or not action.strip():
        raise ConversionError("A parsed native action string is required")
    return action.strip()
