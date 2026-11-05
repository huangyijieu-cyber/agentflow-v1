"""Copy colocated actor shards into the frozen reference at step boundaries.

No weights cross the trainer/Ray driver and no optimizer state is copied. The
strict layout check intentionally rejects differently wrapped models instead
of silently assigning equal-sized but unrelated flat parameter shards.
"""

from weakref import WeakValueDictionary
from pathlib import Path


_ACTORS = WeakValueDictionary()


def register_actor(worker):
    """Workers for different roles share one process for each distributed rank."""
    if getattr(worker, "_is_actor", False):
        _ACTORS[int(worker.rank)] = worker


def colocated_actor(reference):
    if not getattr(reference, "_is_ref", False) or getattr(reference, "_is_lora", False):
        raise ValueError("Reference refresh requires a dedicated non-LoRA reference worker")
    config = reference.config
    model_path = config.model.path
    separate_model = config.ref.get("model", None)
    reference_path = model_path if separate_model is None else separate_model.get("path", model_path)
    if str(model_path) != str(reference_path) and (
            Path(str(model_path)).expanduser().resolve() != Path(str(reference_path)).expanduser().resolve()):
        raise ValueError("Reference refresh requires the same initial actor/reference model path; "
                         "a separately configured reference model may have incompatible model semantics")
    actor = _ACTORS.get(int(reference.rank))
    if actor is None:
        raise RuntimeError("Reference refresh requires actor and reference roles colocated on each rank")
    if getattr(actor, "_is_lora", False):
        raise ValueError("Reference refresh does not support LoRA/ref_in_actor")
    if int(actor.world_size) != int(reference.world_size):
        raise ValueError("Reference refresh requires matching actor/reference world sizes")
    if _mesh_signature(actor.device_mesh) != _mesh_signature(reference.device_mesh):
        raise ValueError("Reference refresh requires matching actor/reference device meshes")
    return actor


def _mesh_signature(mesh):
    return (tuple(mesh.mesh.shape), tuple(mesh.mesh.detach().cpu().reshape(-1).tolist()),
            tuple(mesh.mesh_dim_names or ()), str(mesh.device_type))


def _local_tensor(tensor):
    # DTensor.to_local() does not gather other ranks' shards.
    return tensor.to_local() if hasattr(tensor, "placements") and hasattr(tensor, "to_local") else tensor


def _tensor_layout(tensor):
    local = _local_tensor(tensor)
    layout = (tuple(tensor.shape), tuple(local.shape), tuple(local.stride()))
    if hasattr(tensor, "placements"):
        layout += (_mesh_signature(tensor.device_mesh), tuple(str(p) for p in tensor.placements))
    return layout


def _flat_layout(flat):
    # Includes intra-flat offsets and padding (not only the flat shard length).
    return tuple((key, repr(getattr(flat, key, None))) for key in (
        "_fqns", "_shapes", "_numels", "_numels_with_padding", "_is_padding_mask",
        "_shard_param_infos", "_sharded_size", "_unpadded_unsharded_size",
    )) + (("shared", tuple(tuple(getattr(info, field) for field in (
        "param_name", "module_name", "prim_param_name", "prim_module_name"))
        for info in getattr(flat, "_shared_param_infos", ()))),)


def _fsdp1_handles(model):
    return {name: module._handle for name, module in model.named_modules()
            if getattr(module, "_handle", None) is not None
            and hasattr(module._handle, "flat_param")}


def _assert_sharded(handle, name):
    flat = handle.flat_param
    local = getattr(flat, "_local_shard", flat)
    expected = getattr(flat, "_sharded_size", flat.shape)
    if tuple(flat.shape) != tuple(expected) or tuple(local.shape) != tuple(expected):
        raise RuntimeError(f"Reference refresh requires a resharded model at a step boundary: {name}")
    if flat.numel() and flat.data_ptr() != local.data_ptr():
        raise RuntimeError(f"Reference refresh found detached local shard storage: {name}")


def _persistent_buffers(model):
    result = {}
    for module_name, module in model.named_modules():
        for name, tensor in module._buffers.items():
            if tensor is not None and name not in module._non_persistent_buffers_set:
                result[f"{module_name}.{name}" if module_name else name] = tensor
    return result


def release_reference_full_params(model):
    """Discard all FSDP all-gather caches before replacing their local shards."""
    for handle in _fsdp1_handles(model).values():
        if hasattr(handle.flat_param, "_local_shard"):
            handle.reshard(True)
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:  # torch 2.4/2.5 expose FSDP2 under the composable API
        from torch.distributed._composable.fsdp import FSDPModule
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


def prepare_local_copy(actor_model, reference_model):
    """Validate every source/destination before returning a local-only copy plan."""
    source_handles, target_handles = _fsdp1_handles(actor_model), _fsdp1_handles(reference_model)
    if source_handles.keys() != target_handles.keys():
        raise ValueError("Reference refresh requires identical FSDP wrapping")
    for name, source_handle in source_handles.items():
        target_handle = target_handles[name]
        _assert_sharded(source_handle, name)
        _assert_sharded(target_handle, name)
        if _flat_layout(source_handle.flat_param) != _flat_layout(target_handle.flat_param):
            raise ValueError(f"Reference refresh FSDP flat parameter layout mismatch: {name}")
        if bool(source_handle._use_orig_params) != bool(target_handle._use_orig_params):
            raise ValueError(f"Reference refresh use_orig_params mismatch: {name}")

    # Flat parameters cover all original weights in FSDP1, including parameters
    # whose local original view is empty under use_orig_params=True.
    if source_handles:
        source_params = {name: handle.flat_param for name, handle in source_handles.items()}
        target_params = {name: handle.flat_param for name, handle in target_handles.items()}
    else:
        source_params, target_params = dict(actor_model.named_parameters()), dict(reference_model.named_parameters())
    sources = {"parameter:" + name: value for name, value in source_params.items()}
    targets = {"parameter:" + name: value for name, value in target_params.items()}
    sources.update({"buffer:" + name: value for name, value in _persistent_buffers(actor_model).items()})
    targets.update({"buffer:" + name: value for name, value in _persistent_buffers(reference_model).items()})
    if sources.keys() != targets.keys():
        raise ValueError("Reference refresh requires identical model parameters and persistent buffers")
    plan, aliases = [], {}
    for name, source in sources.items():
        target = targets[name]
        if _tensor_layout(source) != _tensor_layout(target):
            raise ValueError(f"Reference refresh parameter/shard layout mismatch: {name}")
        source, target = _local_tensor(source).detach(), _local_tensor(target).detach()
        if source.is_meta or target.is_meta:
            raise ValueError(f"Reference refresh cannot copy unmaterialized parameters: {name}")
        if source.layout != target.layout:
            raise ValueError(f"Reference refresh tensor layout mismatch: {name}")
        # Deduplicate tied destinations, and ensure tied source aliases agree.
        key = (str(target.device), target.data_ptr(), tuple(target.shape), tuple(target.stride()))
        source_key = (str(source.device), source.data_ptr(), tuple(source.shape), tuple(source.stride()))
        if target.numel() and key in aliases:
            if aliases[key] != source_key:
                raise ValueError(f"Reference refresh tied parameter mismatch: {name}")
            continue
        aliases[key] = source_key
        plan.append((source, target))
    return plan


def invalidate_reference_cache(worker, completed_step):
    for name in ("_trace_prefix_verified", "_trace_prefix_disabled", "_trace_prefix_output_dtype"):
        if hasattr(worker, name):
            delattr(worker, name)
    worker._trace_reference_step = int(completed_step)
    worker.ref_module_fsdp.eval()


def copy_local_reference(actor, reference, completed_step):
    """Copy weights with dtype conversion; keep actor gradients/storage intact."""
    import torch

    release_reference_full_params(reference.ref_module_fsdp)
    plan = prepare_local_copy(actor.actor_module_fsdp, reference.ref_module_fsdp)
    with torch.no_grad():
        for source, target in plan:
            target.copy_(source, non_blocking=False)
    invalidate_reference_cache(reference, completed_step)
    return {"reference_step": int(completed_step), "tensors": len(plan),
            "numel": sum(target.numel() for _, target in plan)}
