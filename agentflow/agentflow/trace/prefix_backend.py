"""Reference-only integration of exact-prefix tree attention.

The actor's packed/SP forward is not changed. Tree forwards use the same frozen
FSDP reference with dense ancestor masks and full sequences on each SP rank.
All ranks agree on eligibility and execute the same number of FSDP forwards.
"""

from contextlib import contextmanager, nullcontext
import inspect


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
    from .prefix_scoring import build_prefix_packs, compute_tree_logps, PrefixScoringError

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
