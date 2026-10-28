"""Teacher-forced fixed-response scoring for saved ALFWorld TRACE probes.

Importing this module needs only the Python standard library. Model dependencies
are loaded only when a real scoring backend is requested. No environment or
trajectory-generation code is imported here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.request


class ProbeScoringError(RuntimeError):
    """A backend cannot provide the requested teacher-forcing probabilities."""


_SCORERS: dict[str, Any] = {}


def _trace_module():
    name = "_alfworld_probe_trace_credit"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "agentflow/agentflow/trace/credit.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ProbeScoringError(f"Cannot load TRACE credit implementation: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _number_or_none(value):
    return float(value) if _finite(value) else None


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_tokenizer(config):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ProbeScoringError(
            "Scoring requires transformers. Install it in the model runtime, "
            "and set scorer.tokenizer to the scoring model's tokenizer path/name."
        ) from exc
    source = config.get("tokenizer") or config.get("model")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("scorer.tokenizer or scorer.model must identify a tokenizer")
    return AutoTokenizer.from_pretrained(
        source, trust_remote_code=bool(config.get("trust_remote_code", False)),
    )


def _validate_logps(values, expected):
    if not isinstance(values, list) or len(values) != expected:
        raise ProbeScoringError(
            f"Expected {expected} response-token log probabilities, got "
            f"{len(values) if isinstance(values, list) else type(values).__name__}"
        )
    if not all(_finite(value) and value <= 0 for value in values):
        raise ProbeScoringError("Response LogP must be finite and <= 0 for every target token")
    return [float(value) for value in values]


class _VLLMScorer:
    """Use echoed prompt probabilities, never generated-answer probabilities."""

    def __init__(self, config):
        self.config = dict(config)
        self.tokenizer = _load_tokenizer(config)
        self.model = config.get("model")
        self.base_url = str(config.get("base_url", "http://127.0.0.1:19998/v1")).rstrip("/")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("scorer.model is required for vllm scoring")

    def score(self, prompt_ids, target_ids):
        # The *entire* fixed target is part of the supplied prompt. Echoed
        # prompt-token probabilities therefore condition on the correct target
        # prefix. max_tokens=0 asks vLLM for echo without a sampled completion.
        full_ids = list(prompt_ids) + list(target_ids)
        body = {
            "model": self.model,
            "prompt": full_ids,
            "echo": True,
            "logprobs": 1,
            "max_tokens": 0,
            "temperature": 1.0,
            "stream": False,
        }
        key = os.environ.get(self.config.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
        request = urllib.request.Request(
            self.base_url + "/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.config.get("timeout", 120))) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not echo arbitrary server bodies, which may contain credentials
            # or a very large prompt. The explicit interface requirement helps
            # distinguish an unsupported scoring endpoint from generation.
            raise ProbeScoringError(
                f"Scoring HTTP {exc.code}. The vllm backend requires /completions "
                "with integer-token prompts, echo=true, logprobs=1 and max_tokens=0. "
                "A chat-only generation endpoint is insufficient; use backend=hf if needed."
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProbeScoringError(f"Scoring request failed: {type(exc).__name__}: {exc}") from exc
        choices = payload.get("choices") or []
        if len(choices) != 1:
            raise ProbeScoringError("Scoring endpoint did not return exactly one completion choice")
        logprobs = choices[0].get("logprobs") or {}
        echoed = logprobs.get("token_logprobs")
        if not isinstance(echoed, list) or len(echoed) < len(full_ids):
            raise ProbeScoringError(
                "Scoring endpoint omitted echoed prompt token_logprobs. "
                "Generated completion logprobs cannot score the fixed response."
            )
        prompt_count = (payload.get("usage") or {}).get("prompt_tokens")
        if prompt_count is not None and prompt_count != len(full_ids):
            raise ProbeScoringError(
                "Server prompt token count differs from supplied token IDs; "
                "verify the serving model and scorer tokenizer match."
            )
        start = len(prompt_ids)
        return _validate_logps(echoed[start:start + len(target_ids)], len(target_ids))


class _HFScorer:
    """Local frozen causal-model forward, with response-only target masking."""

    def __init__(self, config):
        self.tokenizer = _load_tokenizer(config)
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise ProbeScoringError("The hf backend requires torch and transformers") from exc
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            config.get("model"), torch_dtype="auto",
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
        device = config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

    def score(self, prompt_ids, target_ids):
        torch = self.torch
        full = torch.tensor([list(prompt_ids) + list(target_ids)], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            outputs = self.model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False)
            # Causal shift: token at index n is predicted by logits[n-1].
            start = len(prompt_ids) - 1
            logits = outputs.logits[0, start:start + len(target_ids)].float()
            targets = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
            chosen = logits.gather(1, targets[:, None]).squeeze(1)
            result = (chosen - torch.logsumexp(logits, dim=-1)).cpu().tolist()
        return _validate_logps(result, len(target_ids))


def _get_scorer(config):
    key = json.dumps(config, sort_keys=True, ensure_ascii=False)
    if key not in _SCORERS:
        backend = config.get("backend", "vllm")
        if backend == "vllm":
            _SCORERS[key] = _VLLMScorer(config)
        elif backend == "hf":
            _SCORERS[key] = _HFScorer(config)
        else:
            raise ValueError("scorer.backend must be 'vllm' or 'hf'")
    return _SCORERS[key]


def _render(template, fields):
    # Deliberately not str.format(): JSON braces and arbitrary editable prose
    # remain literal. Only these documented placeholders are substituted.
    return re.sub(r"\{(goal|remaining_steps|turn_index)\}", lambda match: str(fields[match.group(1)]), template)


def _invalid_state(state, response, reason, scoring_prompt=None):
    return {
        **state, "scoring_prompt": scoring_prompt, "response": response,
        "scoring_rendered_prompt": None, "mean_logp": None,
        "token_logps": [], "response_token_ids": [], "response_token_count": 0,
        "prompt_token_count": 0, "value": None, "valid": False, "error": reason,
    }


def score_snapshots(snapshot: dict, probe: dict, scorer_config: dict, *, progress=None) -> dict:
    """Rescore saved states and return JSON-safe state/turn diagnostics.

    ``snapshot['trajectories']`` contains ``states`` (T+1) and ``turns`` (T).
    The fixed response may substitute {goal}; state-dependent response targets
    are rejected because their LogP differences are not a fixed-target value.
    No final reward is injected into the model's LogP or derived state value.
    """
    started = time.monotonic()
    instruction = probe.get("prompt")
    response_template = probe.get("response")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("probe.prompt must be a nonempty string")
    if not isinstance(response_template, str) or not response_template.strip():
        raise ValueError("probe.response must be a nonempty fixed target string")
    if re.search(r"\{(?:remaining_steps|turn_index)\}", response_template):
        raise ValueError("probe.response must stay fixed across states; only {goal} is allowed")
    epsilon = probe.get("epsilon", 0.1)
    k = _positive_int(probe.get("k", 3), "probe.k")
    gamma = probe.get("gamma", 0.8)
    max_response = _positive_int(probe.get("max_response_tokens", 256), "probe.max_response_tokens")
    max_context = _positive_int(scorer_config.get("max_context_tokens", 32768), "scorer.max_context_tokens")
    trace = _trace_module()
    trace.TraceConfig(epsilon=epsilon, k=k, gamma=gamma, terminal_fill_enabled=False)
    trajectories = snapshot.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("The snapshot must contain at least one saved trajectory")
    scorer = _get_scorer(scorer_config)
    tokenizer = scorer.tokenizer
    cache = {}
    target_cache = {}
    result_trajectories = []
    counters = {"states": 0, "turns": 0, "valid_scored_states": 0,
                "valid_credit_turns": 0, "invalid_scored_states": 0,
                "external_error_windows": 0}
    for trajectory in trajectories:
        states = trajectory.get("states", [])
        turns = trajectory.get("turns", [])
        if not states or len(states) != len(turns) + 1:
            raise ValueError("Each saved trajectory needs T actions and T+1 real states")
        goal = trajectory.get("goal", "")
        response = _render(response_template, {"goal": goal})
        if response not in target_cache:
            target_cache[response] = list(tokenizer.encode(response, add_special_tokens=False))
        target_ids = target_cache[response]
        scored_states = []
        if progress:
            progress(f"Scoring {trajectory.get('trajectory_id', '?')}: {len(states)} saved states")
        for position, state in enumerate(states):
            context = state.get("state_context")
            fields = {"goal": goal, "remaining_steps": state.get("remaining_steps", "unknown"),
                      "turn_index": state.get("state_index", position)}
            prompt = _render(instruction, fields)
            content = context.strip() + "\n\n" + prompt if isinstance(context, str) and context.strip() else None
            if state.get("scoring_valid") is False:
                # An environment crash can return an error observation rather
                # than an actual successor state. Keep it for inspection, but
                # never turn that synthetic error text into a value probe.
                env_info = state.get("env_info") or {}
                reason = ("external_environment_error_state" if env_info.get("env_error")
                          else "state_marked_scoring_invalid")
                row = _invalid_state(state, response, reason, content)
            elif not content:
                row = _invalid_state(state, response, "missing_state_context")
            elif not target_ids or len(target_ids) > max_response:
                row = _invalid_state(state, response, "invalid_response_token_length", content)
                row["response_token_count"] = len(target_ids)
            else:
                prompt_ids = list(tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                ))
                if not prompt_ids:
                    row = _invalid_state(state, response, "empty_chat_template", content)
                elif len(prompt_ids) + len(target_ids) > max_context:
                    row = _invalid_state(state, response, "context_too_long_no_truncation", content)
                    row["prompt_token_count"] = len(prompt_ids)
                    row["response_token_count"] = len(target_ids)
                else:
                    key = (tuple(prompt_ids), tuple(target_ids))
                    if key not in cache:
                        logps = _validate_logps(scorer.score(prompt_ids, target_ids), len(target_ids))
                        cache[key] = {"token_logps": logps, "mean_logp": math.fsum(logps) / len(logps)}
                    score = cache[key]
                    row = {
                        **state, "scoring_prompt": content, "response": response,
                        "scoring_rendered_prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False),
                        "mean_logp": score["mean_logp"], "token_logps": list(score["token_logps"]),
                        "response_token_ids": list(target_ids), "response_token_count": len(target_ids),
                        "prompt_token_count": len(prompt_ids), "value": None, "valid": True, "error": None,
                    }
            scored_states.append(row)
        state_indices = [state.get("state_index", i) for i, state in enumerate(states)]
        values = trace.state_values([state["mean_logp"] for state in scored_states], epsilon=epsilon)
        credits = trace.discounted_credits(values, state_indices, k=k, gamma=gamma)
        for state, value in zip(scored_states, values):
            state["value"] = _number_or_none(value)
        scored_turns = []
        for i, turn in enumerate(turns):
            current, following = scored_states[i:i + 2]
            continuous = (all(_finite(index) and float(index).is_integer() for index in state_indices[i:i + 2])
                          and state_indices[i + 1] == state_indices[i] + 1)
            external = bool(turn.get("external_error", False))
            delta = (values[i + 1] - values[i]
                     if continuous and not external and _finite(values[i]) and _finite(values[i + 1]) else None)
            # Failure transitions remain in their original positions. Never
            # delete them and join states across the resulting gap.
            external_window = any(bool(item.get("external_error", False)) for item in turns[i:min(i + k, len(turns))])
            credit = None if external_window else _number_or_none(credits[i])
            if external_window:
                reason = "external_error_in_credit_window"
                counters["external_error_windows"] += 1
            elif credit is None:
                reason = ("invalid_initial_state_value" if not _finite(values[0])
                          else "missing_or_invalid_state_or_turn_gap")
            else:
                reason = None
            scored_turns.append({
                **turn, "scoring_prompt": current["scoring_prompt"], "response": response,
                "mean_logp": current["mean_logp"], "token_logps": current["token_logps"],
                "value": current["value"], "next_value": following["value"],
                "delta": _number_or_none(delta), "credit": credit, "local_credit": credit,
                "terminal_fill": 0.0 if credit is not None else None,
                "credit_valid": credit is not None, "credit_error": reason,
            })
        counters["states"] += len(scored_states)
        counters["turns"] += len(scored_turns)
        counters["valid_scored_states"] += sum(state["valid"] for state in scored_states)
        counters["invalid_scored_states"] += sum(not state["valid"] for state in scored_states)
        counters["valid_credit_turns"] += sum(turn["credit_valid"] for turn in scored_turns)
        result_trajectories.append({**trajectory, "states": scored_states, "turns": scored_turns})
    counters.update({"trajectories": len(result_trajectories), "unique_scoring_requests": len(cache),
                     "scoring_seconds": time.monotonic() - started})
    fingerprint = hashlib.sha256(json.dumps({"snapshot_id": snapshot.get("snapshot_id"), "probe": probe,
                                             "scorer": scorer_config}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        "snapshot_id": snapshot.get("snapshot_id"), "scoring_fingerprint": fingerprint,
        "trajectories": result_trajectories, "summary": counters,
        "formula": {
            "mean_logp": "mean response-token log P_ref(response | chat(state_context + prompt)), temperature=1",
            "value": "log((-mean_logp_initial + epsilon) / (-mean_logp_state + epsilon))",
            "delta": "value_after_action - value_before_action",
            "credit": "sum(gamma**j * delta[t+j]) / sum(gamma**j), up to K real actions",
            "epsilon": epsilon, "k": k, "gamma": gamma, "terminal_fill_enabled": False,
            "terminal_state": "Real post-action state is scored; no synthetic final response or reward override",
            "invalid": "null, never zero; no truncation or bridging missing/error transitions",
        },
    }
