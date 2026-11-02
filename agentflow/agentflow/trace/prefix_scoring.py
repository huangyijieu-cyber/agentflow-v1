"""Exact-prefix tree scoring for teacher-forced TRACE targets.

Each physical token node is evaluated once. A four-dimensional additive mask
allows a node to attend only to its own ancestor chain, so branches share their
prefix computation without seeing other branches or future observations.

This adapts the attention-mask construction described by IGPO's vectorized GT
implementation. Unlike a single growing trajectory, ALFWorld snapshots can
change earlier text; an exact-token trie handles those independent snapshots.
No training/distributed dependency is imported until tensors are requested.

Reference: GuoqingWang1/IGPO, commit 64165e2741ed8801f977948c8128080ce87b4101,
scrl/llm_agent/vectorized_gt_logprob.py (the custom-attention implementation).
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real


class PrefixScoringError(ValueError):
    """The requested tree cannot be represented safely or scored exactly."""


@dataclass
class PrefixTreePack:
    input_ids: list
    parents: list
    position_ids: list
    request_indices: list
    target_parent_indices: list
    target_ids: list
    original_tokens: int

    @property
    def tree_tokens(self):
        return len(self.input_ids)

    @property
    def saved_tokens(self):
        return self.original_tokens - self.tree_tokens


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise PrefixScoringError(f"{name} must be a positive integer")
    return int(value)


def _tokens(value, name):
    try:
        result = tuple(value)
    except TypeError as error:
        raise PrefixScoringError(f"{name} must be a nonempty token sequence") from error
    if not result or any(isinstance(token, bool) or not isinstance(token, Integral)
                         or token < 0 for token in result):
        raise PrefixScoringError(f"{name} must contain nonnegative integer token IDs")
    return tuple(int(token) for token in result)


class _TreeBuilder:
    def __init__(self):
        self.children = [{}]  # The virtual root has no model token or position.
        self.tokens, self.parents, self.depths = [], [], []
        self.rows, self.target_parents, self.targets = [], [], []
        self.original_tokens = 0
        self.original_attention_cells = 0

    def additional_nodes(self, tokens):
        node = 0
        for offset, token in enumerate(tokens):
            child = self.children[node].get(token)
            if child is None:
                return len(tokens) - offset
            node = child
        return 0

    def add(self, row, prompt, target):
        node = 0
        path = []
        for token in prompt + target:
            child = self.children[node].get(token)
            if child is None:
                child = len(self.children)
                self.children[node][token] = child
                self.children.append({})
                self.tokens.append(token)
                self.parents.append(node - 1)
                self.depths.append(0 if node == 0 else self.depths[node - 1] + 1)
            path.append(child - 1)
            node = child
        self.rows.append(row)
        # A target token is predicted by its logical parent, not the previous
        # physical token. In particular GT[0] uses the last PROMPT node.
        self.target_parents.append(path[len(prompt) - 1:len(prompt) + len(target) - 1])
        self.targets.append(list(target))
        self.original_tokens += len(prompt) + len(target)
        self.original_attention_cells += (len(prompt) + len(target)) ** 2

    def finish(self):
        return PrefixTreePack(self.tokens, self.parents, self.depths, self.rows,
                              self.target_parents, self.targets, self.original_tokens)


def build_prefix_packs(requests, *, max_tree_tokens, max_mask_bytes, mask_element_bytes=4,
                       max_attention_cost_ratio=1.0):
    """Return bounded tree packs and counters without changing request order IDs.

    Requests use the same ``prompt_ids``/``target_ids`` as build_score_tensors.
    Packs may reorder requests to keep common prefixes together; each pack's
    request_indices maps results to the original request list. Limits apply to
    the dense additive mask, not to model activation or logits memory.

    Dense attention evaluates an L-by-L tree even for masked branches. By
    default each pack must cost no more attention cells than its independent
    requests combined. This avoids merging unrelated snapshots into a larger
    quadratic attention problem. None disables only this cost guard, for
    explicit low-level mask/correctness tests; token/mask bounds still apply.

    A single request that exceeds a limit raises rather than truncating it or
    inventing a score. The caller can explicitly choose its original scorer.
    """
    max_tree_tokens = _positive_integer(max_tree_tokens, "max_tree_tokens")
    max_mask_bytes = _positive_integer(max_mask_bytes, "max_mask_bytes")
    mask_element_bytes = _positive_integer(mask_element_bytes, "mask_element_bytes")
    if (max_attention_cost_ratio is not None
            and (isinstance(max_attention_cost_ratio, bool)
                 or not isinstance(max_attention_cost_ratio, Real)
                 or not math.isfinite(max_attention_cost_ratio)
                 or max_attention_cost_ratio < 1)):
        raise PrefixScoringError("max_attention_cost_ratio must be finite and at least 1, or None")
    limit = min(max_tree_tokens, math.isqrt(max_mask_bytes // mask_element_bytes))
    records = []
    for row, request in enumerate(requests):
        try:
            prompt = _tokens(request["prompt_ids"], "prompt_ids")
            target = _tokens(request["target_ids"], "target_ids")
        except (TypeError, KeyError) as error:
            raise PrefixScoringError("Each request needs prompt_ids and target_ids") from error
        if len(prompt) + len(target) > limit:
            raise PrefixScoringError(
                f"Request {row} needs {len(prompt) + len(target)} tokens; tree/mask limit allows {limit}")
        records.append((prompt + target, row, prompt, target))
    # Lexicographic ordering makes common prefixes contiguous, avoiding a pack
    # per rollout when identical rules/initial facts are shared across rollouts.
    records.sort(key=lambda record: (record[0], record[1]))
    packs, current = [], _TreeBuilder()
    for tokens, row, prompt, target in records:
        extra = current.additional_nodes(tokens)
        tree_size = len(current.tokens) + extra
        independent_cells = current.original_attention_cells + len(tokens) ** 2
        excessive_attention = (max_attention_cost_ratio is not None
                               and tree_size ** 2 > max_attention_cost_ratio * independent_cells)
        if current.rows and (tree_size > limit or excessive_attention):
            packs.append(current.finish())
            current = _TreeBuilder()
        current.add(row, prompt, target)
    if current.rows:
        packs.append(current.finish())
    original_tokens = sum(pack.original_tokens for pack in packs)
    tree_tokens = sum(pack.tree_tokens for pack in packs)
    original_attention_cells = sum(len(record[0]) ** 2 for record in records)
    tree_attention_cells = sum(pack.tree_tokens ** 2 for pack in packs)
    return packs, {
        "requests": len(requests), "packs": len(packs),
        "original_tokens": original_tokens, "tree_tokens": tree_tokens,
        "saved_tokens": original_tokens - tree_tokens,
        "original_attention_cells": original_attention_cells,
        "tree_attention_cells": tree_attention_cells,
        "attention_cost_ratio": (tree_attention_cells / original_attention_cells
                                 if original_attention_cells else 0.0),
        "largest_tree_tokens": max((pack.tree_tokens for pack in packs), default=0),
        "largest_mask_bytes": max((pack.tree_tokens ** 2 * mask_element_bytes for pack in packs), default=0),
    }


def build_tree_tensors(pack, *, device=None, mask_dtype=None):
    """Construct one [1,L] tree and its [1,1,L,L] ancestor-only additive mask."""
    import numpy as np
    import torch

    size = pack.tree_tokens
    if not size or len(pack.parents) != size or len(pack.position_ids) != size:
        raise PrefixScoringError("Tree pack tokens, parents and positions must be nonempty and aligned")
    if mask_dtype is None:
        mask_dtype = torch.float32
    if not mask_dtype.is_floating_point:
        raise PrefixScoringError("Tree attention mask must use a floating dtype")
    # Building rows on the CPU avoids one NPU kernel launch per tree node. A
    # child's visible ancestors are exactly its parent's visible ancestors.
    mask = np.full((size, size), -np.inf, dtype=np.float32)
    for node, parent in enumerate(pack.parents):
        if parent < -1 or parent >= node:
            raise PrefixScoringError("Tree parents must precede their children")
        expected_depth = 0 if parent == -1 else pack.position_ids[parent] + 1
        if pack.position_ids[node] != expected_depth:
            raise PrefixScoringError("Tree positions must equal logical prefix depths")
        if parent >= 0:
            mask[node, :node] = mask[parent, :node]
        mask[node, node] = 0.0
    return {
        "input_ids": torch.tensor([pack.input_ids], dtype=torch.long, device=device),
        "position_ids": torch.tensor([pack.position_ids], dtype=torch.long, device=device),
        "attention_mask": torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device=device, dtype=mask_dtype),
    }


def compute_tree_logps(model, pack, *, device=None, mask_dtype=None, temperature=1.0,
                       logits_to_keep=False):
    """Score every requested GT token from its logical parent's logits.

    Returns a list of one-dimensional float32 tensors in pack.request_indices
    order. ``model`` must be in eval mode, support arbitrary additive 4D masks,
    and use the supplied position IDs without replacing the tree with causal
    physical-position attention. Autocast, FSDP and device placement belong to
    the caller. The optional HF logits_to_keep optimization must only be used
    when that model's forward explicitly supports tensor position selection.
    """
    import torch

    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or temperature <= 0:
        raise PrefixScoringError("temperature must be finite and positive")
    if getattr(model, "training", False):
        raise PrefixScoringError("Tree reference scoring requires eval mode")
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
    tensors = build_tree_tensors(pack, device=device, mask_dtype=mask_dtype)
    parents = [parent for row in pack.target_parent_indices for parent in row]
    targets = [token for row in pack.target_ids for token in row]
    lengths = [len(row) for row in pack.target_ids]
    if (len(pack.request_indices) != len(lengths)
            or len(pack.target_parent_indices) != len(lengths)
            or any(len(indices) != length for indices, length in zip(pack.target_parent_indices, lengths))
            or not parents or any(parent < 0 or parent >= pack.tree_tokens for parent in parents)):
        raise PrefixScoringError("GT target IDs and parent-logit indices must be aligned")
    parent_tensor = torch.tensor(parents, dtype=torch.long, device=device)
    if logits_to_keep:
        keep, inverse = torch.unique(parent_tensor, sorted=True, return_inverse=True)
        tensors["logits_to_keep"] = keep
    with torch.no_grad():
        output = model(**tensors, use_cache=False)
        logits = output.logits
        expected_width = len(keep) if logits_to_keep else pack.tree_tokens
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != expected_width:
            raise PrefixScoringError("Model logits do not match tree positions; cannot score exact GT tokens")
        selected = logits[0].index_select(0, inverse if logits_to_keep else parent_tensor).float()
        selected = selected / temperature
        target_tensor = torch.tensor(targets, dtype=torch.long, device=selected.device)
        logps = selected.log_softmax(dim=-1).gather(1, target_tensor[:, None]).squeeze(1)
    return list(logps.split(lengths))
