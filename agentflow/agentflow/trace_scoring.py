"""Prepare teacher-forced GT probes without importing distributed runtimes."""
import math
from collections import Counter

import torch


def first_ground_truth(value):
    """Use a fixed first nonempty alias, never the alias easiest for a state."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for alias in value:
            answer = first_ground_truth(alias)
            if answer is not None:
                return answer
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value) if math.isfinite(value) else None
    return None


def encode_state_requests(contexts, ground_truths, tokenizer, *, max_prompt_length,
                          max_gt_tokens, answer_instruction):
    """Return unique probes and their row mapping; never truncate a state or GT."""
    if len(contexts) != len(ground_truths):
        raise ValueError("TRACE contexts and ground truths must be aligned")
    requests, row_requests, seen, target_cache = [], [], {}, {}
    counts = Counter({key: 0 for key in (
        "missing_gt", "missing_state", "invalid_gt_length", "overlong_state",
        "deduplicated_states", "scoring_requests",
    )})
    for context, raw_gt in zip(contexts, ground_truths):
        answer = first_ground_truth(raw_gt)
        if answer is None:
            counts["missing_gt"] += 1
            row_requests.append(None)
            continue
        if not isinstance(context, str) or not context.strip():
            counts["missing_state"] += 1
            row_requests.append(None)
            continue
        if answer not in target_cache:
            target_cache[answer] = tokenizer.encode(answer, add_special_tokens=False)
        target_ids = target_cache[answer]
        if not target_ids or len(target_ids) > max_gt_tokens:
            counts["invalid_gt_length"] += 1
            row_requests.append(None)
            continue
        content = context.strip() + "\n\n" + answer_instruction
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True,
        )
        if not prompt_ids or len(prompt_ids) > max_prompt_length:
            counts["overlong_state"] += 1
            row_requests.append(None)
            continue
        key = (tuple(prompt_ids), tuple(target_ids))
        if key not in seen:
            seen[key] = len(requests)
            requests.append({"prompt_ids": list(prompt_ids), "target_ids": list(target_ids)})
        else:
            counts["deduplicated_states"] += 1
        row_requests.append(seen[key])
    counts["scoring_requests"] = len(requests)
    return requests, row_requests, dict(counts)


def build_score_tensors(requests, pad_token_id):
    if not requests:
        raise ValueError("Cannot build an empty TRACE scoring batch")
    prompt_width = max(len(r["prompt_ids"]) for r in requests)
    target_width = max(len(r["target_ids"]) for r in requests)
    prompts = torch.full((len(requests), prompt_width), pad_token_id, dtype=torch.long)
    responses = torch.full((len(requests), target_width), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros_like(prompts)
    response_mask = torch.zeros_like(responses)
    for row, request in enumerate(requests):
        p, r = request["prompt_ids"], request["target_ids"]
        prompts[row, -len(p):] = torch.tensor(p, dtype=torch.long)
        responses[row, :len(r)] = torch.tensor(r, dtype=torch.long)
        prompt_mask[row, -len(p):] = 1
        response_mask[row, :len(r)] = 1
    attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
    return {
        "prompts": prompts, "responses": responses,
        "input_ids": torch.cat((prompts, responses), dim=-1),
        "attention_mask": attention_mask,
        "position_ids": (attention_mask.cumsum(dim=-1) - 1).clamp(min=0),
        "response_mask": response_mask,
    }


def mean_gt_logps(logps, response_mask):
    """Average only GT tokens, rejecting invalid scores even in mixed batches."""
    if logps.shape != response_mask.shape:
        raise ValueError("TRACE log probabilities and GT mask have different shapes")
    mask = response_mask.bool().to(logps.device)
    scores = logps.float().masked_fill(~mask, 0.0)
    valid = ((torch.isfinite(scores) & (scores <= 0)) | ~mask).all(dim=-1)
    lengths = mask.sum(dim=-1)
    means = scores.sum(dim=-1) / lengths.clamp(min=1)
    return means.masked_fill(~valid | (lengths == 0), float("nan")).cpu().tolist()
