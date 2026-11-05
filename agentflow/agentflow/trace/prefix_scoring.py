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

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import inspect
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


# Reference-worker integration and collective coordination.
def _get(config, name, default=None):
    return config.get(name, default) if hasattr(config, "get") else getattr(config, name, default)


def model_configs(model):
    """Return distinct module configs, including the unwrapped HF config."""
    configs, seen = [], set()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            configs.append(config)
    return configs


@contextmanager
def tree_attention(model):
    """Switch only this reference's configs; never patch a model class globally."""
    configs = model_configs(model)
    saved = [(config, getattr(config, "_attn_implementation", None)) for config in configs]
    try:
        for config, _ in saved:
            config._attn_implementation = "eager"
        yield
    finally:
        for config, implementation in saved:
            config._attn_implementation = implementation


def unsupported_reason(worker, model, data):
    """Return a stable reason code before any rank enters a tree forward."""
    if not data.meta_info.get("trace_gt_scoring", False):
        return "not_gt"
    if getattr(worker, "_trace_prefix_disabled", False):
        return "validation_disabled"
    if getattr(worker, "_is_lora", False) or not getattr(worker, "_is_ref", False):
        return "unsupported_model"
    if data.non_tensor_batch.get("multi_modal_inputs") is not None:
        return "unsupported_model"
    policy = getattr(worker, "ref_policy", None)
    if getattr(policy, "use_fused_kernels", False):
        return "unsupported_model"
    config = getattr(model, "config", None)
    if config is None or _get(config, "model_type") not in {"qwen2", "qwen3"}:
        return "unsupported_model"
    model_options = _get(getattr(worker, "config", None), "model", {})
    if _get(model_options, "use_liger", False):
        return "unsupported_model"
    for config in model_configs(model):
        if (_get(config, "vision_config") is not None
                or _get(config, "use_sliding_window", False)
                or "sliding_attention" in (_get(config, "layer_types", []) or [])):
            return "unsupported_model"
        rope = _get(config, "rope_scaling")
        if rope and _get(rope, "rope_type", _get(rope, "type", "default")) != "default":
            return "unsupported_model"
    # Older transformers select a fixed attention class at construction time;
    # toggling a config does not make such a FlashAttention class accept 4D.
    for module in model.modules():
        if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
            name = type(module).__name__.lower()
            if "flashattention" in name or "sdpaattention" in name:
                return "unsupported_model"
    return None


def requests_from_data(data):
    """Recover exact unpadded prompt and GT token sequences from score tensors."""
    import torch

    batch = data.batch
    inputs = batch["input_ids"].detach().cpu()
    targets = batch["responses"].detach().cpu()
    masks = batch["attention_mask"].detach().cpu().bool()
    target_masks = batch["response_mask"].detach().cpu().bool()
    positions = batch["position_ids"].detach().cpu()
    if (inputs.ndim != 2 or targets.ndim != 2 or masks.shape != inputs.shape
            or positions.shape != inputs.shape or target_masks.shape != targets.shape
            or len(inputs) != len(targets)):
        raise ValueError("TRACE prefix score tensors must have aligned 2D shapes")
    prompt_width = inputs.shape[1] - targets.shape[1]
    if prompt_width <= 0 or not torch.equal(inputs[:, prompt_width:], targets):
        raise ValueError("TRACE prefix scoring needs prompt followed by the original GT")
    requests = []
    for row in range(len(inputs)):
        prompt_mask = masks[row, :prompt_width]
        target_mask = target_masks[row]
        if (not prompt_mask.any() or not target_mask.any()
                or not torch.equal(masks[row, prompt_width:], target_mask)
                or not torch.equal(prompt_mask, torch.arange(prompt_width) >= prompt_width - int(prompt_mask.sum()))
                or not torch.equal(target_mask, torch.arange(len(target_mask)) < int(target_mask.sum()))):
            raise ValueError("TRACE prefix scoring requires contiguous prompt/GT masks")
        valid_positions = positions[row][masks[row]]
        if not torch.equal(valid_positions, torch.arange(len(valid_positions), dtype=positions.dtype)):
            raise ValueError("TRACE prefix scoring requires unchanged logical token positions")
        requests.append({"prompt_ids": inputs[row, :prompt_width][prompt_mask].tolist(),
                         "target_ids": targets[row][target_mask].tolist()})
    return requests


class _Consensus:
    """World-wide agreement also covers every FSDP hybrid-sharding subgroup."""
    def __init__(self, device):
        import torch.distributed as dist
        self.dist, self.device = dist, device
        self.active = dist.is_available() and dist.is_initialized()

    def integer(self, value, operation):
        import torch
        if not self.active:
            return int(value)
        tensor = torch.tensor(int(value), dtype=torch.int64, device=self.device)
        self.dist.all_reduce(tensor, op=getattr(self.dist.ReduceOp, operation))
        return int(tensor.item())

    def maximum(self, value):
        import torch
        if not self.active:
            return float(value)
        tensor = torch.tensor(float(value), dtype=torch.float32, device=self.device)
        self.dist.all_reduce(tensor, op=self.dist.ReduceOp.MAX)
        return float(tensor.item())


_REASONS = ("not_gt", "unsupported_model", "validation_disabled", "planning", "no_sharing", "validation")


def compute_reference_prefix(worker, data, proto_cls, *, device_name, device_id,
                             reshard=None, consensus=None):
    """Return native-shaped GT scores plus globally identical prefix metrics.

    Unsupported input and limits fall back before any tree forward. Numerical
    validation failure is agreed across ranks, returns the already computed
    native result and disables this worker's tree path for subsequent batches.
    Runtime/model errors during a forward propagate; ranks never independently
    switch attention/collective protocols after another rank has started.
    """
    import torch

    device = torch.device(device_id) if isinstance(device_id, (str, torch.device)) else torch.device(
        device_name if device_name == "cpu" else f"{device_name}:{device_id}")
    sync = consensus or _Consensus(device)
    model = getattr(getattr(worker, "ref_policy", None), "actor_module", None)
    reason = unsupported_reason(worker, model, data) if model is not None else "unsupported_model"
    packs, stats = [], {}
    if reason is None:
        try:
            requests = requests_from_data(data)
            packs, stats = build_prefix_packs(
                requests,
                max_tree_tokens=data.meta_info.get("trace_prefix_max_tokens", 8192),
                max_mask_bytes=data.meta_info.get("trace_prefix_max_mask_bytes", 256 * 1024 ** 2),
            )
        except (PrefixScoringError, ValueError, KeyError, TypeError):
            reason = "planning"
    # Reduce one flag per reason so all ranks report exactly the same fallback.
    flags = {name: sync.integer(reason == name, "MAX") for name in _REASONS}
    metrics = {"trace/prefix_used": 0, "trace/prefix_validation_checked": 0,
               "trace/prefix_validation_passed": 0, "trace/prefix_validation_max_diff": 0.,
               "trace/prefix_forward_rounds": 0, "trace/prefix_saved_tokens": 0,
               "trace/prefix_tree_tokens": 0, "trace/prefix_original_tokens": 0,
               "trace/prefix_packs": 0, "trace/prefix_dummy_forwards": 0,
               "trace/prefix_original_attention_cells": 0, "trace/prefix_tree_attention_cells": 0,
               "trace/prefix_attention_cost_ratio": 0.,
               "trace/prefix_sp_replication": int(getattr(worker, "ulysses_sequence_parallel_size", 1))}

    def finish(output):
        metrics.update({"trace/prefix_fallback_" + key: value for key, value in flags.items()})
        output.meta_info["trace_prefix_metrics"] = metrics
        return output

    if any(flags.values()):
        return finish(worker.compute_ref_log_prob(data))

    saved = sync.integer(stats["saved_tokens"], "SUM")
    if saved <= 0:
        flags["no_sharing"] = 1
        return finish(worker.compute_ref_log_prob(data))
    rounds = sync.integer(len(packs), "MAX")
    sp = max(1, int(getattr(worker, "ulysses_sequence_parallel_size", 1)))
    metrics.update({
        "trace/prefix_forward_rounds": rounds,
        "trace/prefix_saved_tokens": saved // sp,
        "trace/prefix_tree_tokens": sync.integer(stats["tree_tokens"], "SUM") // sp,
        "trace/prefix_original_tokens": sync.integer(stats["original_tokens"], "SUM") // sp,
        "trace/prefix_packs": sync.integer(len(packs), "SUM") // sp,
        "trace/prefix_dummy_forwards": sync.integer(rounds - len(packs), "SUM") // sp,
    })
    original_cells = sync.integer(stats["original_attention_cells"], "SUM") // sp
    tree_cells = sync.integer(stats["tree_attention_cells"], "SUM") // sp
    metrics.update({"trace/prefix_original_attention_cells": original_cells,
                    "trace/prefix_tree_attention_cells": tree_cells,
                    "trace/prefix_attention_cost_ratio": tree_cells / original_cells if original_cells else 0.})
    verify = bool(data.meta_info.get("trace_prefix_verify", True))
    verify = bool(sync.integer(verify and not getattr(worker, "_trace_prefix_verified", False), "MAX"))
    native = worker.compute_ref_log_prob(data) if verify else None
    if native is not None:
        output_dtype = native.batch["ref_log_prob"].dtype
    else:
        output_dtype = getattr(worker, "_trace_prefix_output_dtype", next(model.parameters()).dtype)
    values = torch.zeros(data.batch["responses"].shape, dtype=output_dtype, device="cpu")
    dummy_token = int(data.batch["input_ids"][0, 0])
    dummy = build_prefix_packs([{"prompt_ids": [dummy_token], "target_ids": [dummy_token]}],
                               max_tree_tokens=2, max_mask_bytes=16)[0][0]
    # The standard Qwen HF forward supports selecting LM-head token positions.
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    keep_logits = "logits_to_keep" in inspect.signature(unwrapped.forward).parameters
    autocast = (torch.autocast(device_type=device_name, dtype=torch.bfloat16)
                if device_name in {"cuda", "npu"} else nullcontext())
    model.eval()
    try:
        with tree_attention(model), torch.no_grad(), autocast:
            for index in range(rounds):
                pack = packs[index] if index < len(packs) else dummy
                scores = compute_tree_logps(model, pack, device=device, mask_dtype=torch.float32,
                                            temperature=1., logits_to_keep=keep_logits)
                if index < len(packs):
                    for row, score in zip(pack.request_indices, scores):
                        values[row, :len(score)] = score.detach().to(device="cpu", dtype=output_dtype)
    finally:
        if reshard is not None:
            reshard(model)
    if native is not None:
        mask = data.batch["response_mask"].detach().cpu().bool()
        expected = native.batch["ref_log_prob"].detach().cpu()
        actual_tokens, expected_tokens = values[mask].float(), expected[mask].float()
        finite = torch.isfinite(actual_tokens) & torch.isfinite(expected_tokens)
        difference = float((actual_tokens[finite] - expected_tokens[finite]).abs().max()) if finite.any() else 0.
        passed = torch.allclose(actual_tokens, expected_tokens, rtol=1e-3, atol=1e-3, equal_nan=True)
        passed = bool(sync.integer(passed, "MIN"))
        metrics["trace/prefix_validation_checked"] = 1
        metrics["trace/prefix_validation_passed"] = int(passed)
        metrics["trace/prefix_validation_max_diff"] = sync.maximum(difference)
        if not passed:
            worker._trace_prefix_disabled = True
            flags["validation"] = 1
            return finish(native)
        worker._trace_prefix_verified = True
        worker._trace_prefix_output_dtype = output_dtype
    metrics["trace/prefix_used"] = 1
    return finish(proto_cls.from_dict(tensors={"ref_log_prob": values}))
