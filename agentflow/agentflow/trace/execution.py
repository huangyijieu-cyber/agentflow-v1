"""Pure execution-status and verified-terminal metadata for TRACE pairing.

Only execution envelopes and known adapter error fields are inspected. Retrieved
page prose is not searched for arbitrary error words.
"""

import math
import re
import json

_RESULT_FIELDS = {
    "results",
    "pages",
    "relevant_pages",
    "relevant_pages (to the query)",
    "retrieved_information",
    "execution_results",
}
_ERROR_PREFIXES = (
    "error:",
    "error in execute_tool_command:",
    "error importing tool '",
    "execution timed out after ",
    "no execution captured from block:",
    "no result was generated because the tool was not found.",
    "error fetching url:",
    "error extracting text:",
    "error processing web rag:",
    "error processing page:",
    "error generating response:",
    "yibu brave search gateway request failed.",
    "google search failed to get a valid response",
    "connectionerror:",
    "apiconnectionerror:",
    "timeouterror:",
    "apitimeouterror:",
    "ratelimiterror:",
)


def _status(failed=False, external=False, kind="", source=""):
    return {
        "failed": bool(failed),
        "external_error": bool(external),
        "error_type": kind,
        "error_source": source,
    }


def _failure(message, *, source="", kind="", status_code=None):
    text = str(message).lower()
    if text.startswith(
        ("no results found", "no search results", "no matching results")
    ):
        return _status()
    if source in ("model", "policy"):
        return _status(True, False, kind or "invalid_action", "model")
    if source != "external" and re.search(
        r"invalid (?:tool|argument|parameter)|unexpected keyword|syntaxerror|"
        r"nameerror|invalidurl|missingschema|tool was not found|tool.*not available",
        text,
    ):
        return _status(True, False, kind or "invalid_action", "model")
    http_failure = re.search(
        r"(?:http(?:/\d(?:\.\d)?)?[ =:]+|status(?:_code)?[ =:]+|"
        r"error code:[ ]*|last error:[ ]*)(408|429|500|502|503|504)\b",
        text,
    )
    if (
        source == "external"
        or http_failure
        or status_code in (408, 429, 500, 502, 503, 504)
    ):
        if (
            status_code == 429
            or "ratelimit" in text
            or "rate limit" in text
            or "429" in text
        ):
            return _status(True, True, kind or "rate_limit", "external")
        return _status(True, True, kind or "service_unavailable", "external")
    if re.search(
        r"service unavailable|upstream unavailable|bad gateway|gateway timeout", text
    ):
        return _status(True, True, kind or "service_unavailable", "external")
    if re.search(
        r"api_timeout|apitimeouterror|readtimeout|connecttimeout|deadlineexceeded|"
        r"(?:request|connection|network|http|socket|read)\s+timed?\s*out|"
        r"timeout(?:error|exception)?\b",
        text,
    ) or (text.startswith("error fetching url:") and "timed out" in text):
        # A bare executor wall-time limit may also be caused by generated code.
        if not text.startswith("execution timed out after ") or re.search(
            r"api_timeout|phase=(?:search|web|wiki)|last_upstream_error|"
            r"readtimeout|connecttimeout|apitimeouterror",
            text,
        ):
            return _status(True, True, kind or "network_timeout", "external")
    if re.search(
        r"api_connection_error|connectionerror|apiconnectionerror|"
        r"connecterror|connection (?:reset|refused|aborted)|"
        r"networkerror|remoteprotocolerror|proxyerror|api_proxy_error|"
        r"api_tls_error|sslerror|dns|gaierror|name or service not known|"
        r"temporary failure in name resolution|failed to resolve|"
        r"could not resolve|unable to connect",
        text,
    ):
        return _status(True, True, kind or "connection_error", "external")
    if re.search(r"ratelimiterror|rate.limit|too many requests", text):
        return _status(True, True, kind or "rate_limit", "external")
    return _status(True, False, kind or "tool_error", source or "tool")


def classify_tool_result(result, _depth=0):
    """Classify final tool results, preserving normal empty/no-result searches.

    Explicit final success overrides intermediate retry diagnostics. Nested
    adapter result fields are inspected, but ``content``/``text`` prose is not.
    Unknown processing failures remain separate from external infrastructure.
    """
    if _depth > 8:
        return _status(True, False, "invalid_result", "tool")
    if result is None:
        return _status(True, False, "missing_result", "tool")
    if isinstance(result, dict):
        final_success = result.get("success") is True or result.get("status") in (
            "ok",
            "success",
            "succeeded",
        )
        errors = [
            result.get(key)
            for key in ("error", "exception", "error_message")
            if result.get(key)
        ]
        is_failed = result.get("success") is False or result.get("status") in (
            "error",
            "failed",
            "failure",
            "timeout",
        )
        if (errors or is_failed) and not final_success:
            message = (
                errors[0]
                if errors
                else result.get("message", result.get("status", "tool failed"))
            )
            status = _failure(
                message,
                source=result.get("error_source", ""),
                kind=result.get("error_type", ""),
                status_code=result.get("status_code", result.get("http_status")),
            )
            if status["failed"]:
                return status
        failures = [
            classify_tool_result(value, _depth + 1)
            for key, value in result.items()
            if key in _RESULT_FIELDS
        ]
        return next(
            (item for item in failures if item["external_error"]),
            next((item for item in failures if item["failed"]), _status()),
        )
    if isinstance(result, (list, tuple)):
        failures = [classify_tool_result(item, _depth + 1) for item in result]
        return next(
            (item for item in failures if item["external_error"]),
            next((item for item in failures if item["failed"]), _status()),
        )
    if isinstance(result, str):
        text = result.strip()
        lowered = text.lower()
        if not text:
            return _status(True, False, "missing_result", "tool")
        retry_failure = (
            lowered.startswith("yibu brave search tried ")
            or lowered.startswith("google search tried ")
        ) and " but failed. last error:" in lowered
        if lowered.startswith(_ERROR_PREFIXES) or retry_failure:
            return _failure(text)
        return _status()
    if isinstance(result, BaseException):
        return _failure(type(result).__name__ + ": " + str(result))
    return _status()


def environment_scalar(value):
    """Unpack a scalar from an explicit single-environment batch."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    return value


def environment_flag(value):
    value = environment_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def annotate_alfworld_step(log, state_context, done, info):
    """Adapt a real environment decision to the unchanged QA type routes.

    A verified environment budget failure is a terminal decision; a model
    response cut short by token limits or an external crash is not a trusted
    outcome. The runner performs the final finish_reason check separately.
    """
    info = info if isinstance(info, dict) else {}
    ended = environment_flag(done) is True
    external = environment_flag(info.get("env_error")) is True
    truncated = (environment_flag(info.get("truncated")) is True
                 or environment_flag(info.get("truncated_by_max_step")) is True)
    terminated = environment_flag(info.get("terminated")) is True
    terminal = ended and (terminated or truncated) and not external
    invalid = any(environment_flag(info.get(key)) is True
                  for key in ("format_error", "invalid_action"))
    goal_valid = info.get("trace_goal_check_valid", log.get("trace_goal_check_valid")) is True
    before_valid = log.get("trace_goal_check_valid_before", goal_valid) is True
    state_valid = (log.get("trace_scoring_valid", before_valid) is True
                   and info.get("trace_scoring_valid", goal_valid) is True
                   and before_valid and goal_valid)
    action = info.get("trace_env_action")
    log.update({
        "trace_task": "alfworld", "trace_environment_step": True,
        "trace_state_context": state_context,
        "trace_env_done": ended, "trace_env_terminated": terminated,
        "trace_env_truncated": truncated, "trace_env_won": environment_flag(info.get("won")),
        "trace_env_error": external, "trace_env_terminal": terminal,
        "trace_turn_type": "base" if terminal else "tool_plan",
        "trace_action_key": json.dumps([state_context, action], ensure_ascii=False, sort_keys=True),
        "trace_scoring_valid": state_valid,
        "extraction_valid": log.get("extraction_valid", True) is not False and not invalid and state_valid,
        "trace_tool_failed": invalid or external,
        "trace_external_error": external,
        "trace_error_type": info.get("env_error_type", "environment_error") if external else "",
        "trace_error_source": "external" if external else "",
    })
    for key in ("trace_goal_check", "trace_goal_check_valid"):
        if key in info:
            log[key] = info[key]


def _annotate_alfworld_outcome(planner_logs, result, reward_value):
    """Use the environment verifier, then reuse QA's base/final pair route."""
    candidates = [index for index, item in enumerate(planner_logs)
                  if item.get("trace_task") == "alfworld" and item.get("trace_env_terminal") is True]
    terminal = candidates[0] if len(candidates) == 1 else None
    try:
        finite_reward = math.isfinite(float(reward_value))
    except (TypeError, ValueError, OverflowError):
        finite_reward = False
    selected = planner_logs[terminal] if terminal is not None else {}
    output = result.get("base_response")
    completed = bool(
        terminal is not None and isinstance(output, str) and output.strip()
        and selected.get("response") == output
        and selected.get("finish_reason") in ("stop", "eos", "end_turn")
        and selected.get("trace_env_done") is True
        and not selected.get("trace_env_error", False)
    )
    success = selected.get("trace_env_won")
    prefix = planner_logs[:terminal + 1] if terminal is not None else []
    # Both the actual goal evaluation and the real environment verdict must
    # agree. Missing facts never become a failure label or a zero state value.
    trustworthy = bool(
        completed and finite_reward and isinstance(success, bool)
        and prefix and all(item.get("trace_task") == "alfworld"
                           and item.get("trace_goal_check_valid") is True
                           and item.get("trace_scoring_valid") is True
                           and not item.get("trace_external_error", False)
                           for item in prefix)
        and selected.get("trace_goal_check") is success
    )
    for index, item in enumerate(planner_logs):
        item["trace_is_final"] = bool(completed and index == terminal)
        item["trace_outcome_valid"] = bool(trustworthy and index <= terminal)
        item["trace_outcome_success"] = success if item["trace_outcome_valid"] else None
        item["trace_outcome_reward"] = float(reward_value) if finite_reward else None
    return {"trace_outcome_valid": trustworthy,
            "trace_outcome_success": success if trustworthy else None,
            "trace_final_turn": terminal if completed else None}


def annotate_trace_outcome(planner_logs, result, reward_value, verifier, task="qa"):
    """Mark the exact answer whose verifier result produced the rollout reward.

    The runner's existing base/direct/final output priority is retained. Only
    its attributable prefix receives outcome labels; later optional outputs do
    not become another verified terminal. Legacy metadata defaults are invalid.
    """
    if task == "alfworld":
        return _annotate_alfworld_outcome(planner_logs, result, reward_value)
    selected_key = next(
        (
            key
            for key in ("base_response", "direct_output", "final_output")
            if result.get(key)
        ),
        None,
    )
    kind = {
        "base_response": "base",
        "direct_output": "direct",
        "final_output": "final_answer",
    }.get(selected_key)
    output = result.get(selected_key) if selected_key else None
    selected = [
        index
        for index, log in enumerate(planner_logs)
        if log.get("trace_turn_type", log.get("format_stage")) == kind
        and log.get("response") == output
    ]
    terminal = selected[-1] if selected else None
    try:
        finite_reward = math.isfinite(float(reward_value))
    except (TypeError, ValueError, OverflowError):
        finite_reward = False
    success = verifier.get("success") if isinstance(verifier, dict) else None
    valid_verifier = (
        isinstance(verifier, dict)
        and verifier.get("valid") is True
        and isinstance(success, bool)
        and finite_reward
    )
    completed = (
        task == "qa"
        and terminal is not None
        and isinstance(output, str)
        and bool(output.strip())
        and planner_logs[terminal].get("finish_reason") in ("stop", "eos", "end_turn")
    )
    unresolved = set()
    for index, log in enumerate(planner_logs):
        response_status = classify_tool_result(log.get("response"))
        if response_status["external_error"]:
            log["trace_external_error"] = True
            log["trace_error_type"] = response_status["error_type"]
            log["trace_error_source"] = response_status["error_source"]
        log.setdefault("trace_external_error", False)
        log.setdefault("trace_error_type", "")
        log.setdefault("trace_error_source", "")
        if terminal is not None and index <= terminal:
            key = log.get("trace_action_key") or ("turn", index)
            if log["trace_external_error"]:
                unresolved.add(key)
            elif log.get(
                "trace_turn_type", log.get("format_stage")
            ) == "tool_plan" and not log.get("trace_tool_failed", False):
                unresolved.discard(key)
    outcome_valid = bool(completed and valid_verifier and (success or not unresolved))
    for index, log in enumerate(planner_logs):
        log["trace_is_final"] = bool(completed and index == terminal)
        log["trace_outcome_valid"] = bool(outcome_valid and index <= terminal)
        log["trace_outcome_success"] = success if log["trace_outcome_valid"] else None
        log["trace_outcome_reward"] = float(reward_value) if finite_reward else None
    return {
        "trace_outcome_valid": outcome_valid,
        "trace_outcome_success": success if outcome_valid else None,
        "trace_final_turn": terminal if completed else None,
    }
