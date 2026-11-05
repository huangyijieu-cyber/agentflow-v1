"""CPU contracts for reference-only tree scoring and collective schedules."""

import ast
import importlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from weakref import WeakValueDictionary
from types import ModuleType, SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "trace_prefix_worker_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "agentflow/agentflow/trace")]
sys.modules[PACKAGE] = package
backend = importlib.import_module(PACKAGE + ".prefix_scoring")
scoring = importlib.import_module(PACKAGE + ".prefix_scoring")
credit = importlib.import_module(PACKAGE + ".credit")


class Proto:
    def __init__(self, batch, non_tensors=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensors or {}
        self.meta_info = dict(meta_info or {})

    @classmethod
    def from_dict(cls, tensors, **kwargs):
        return cls(tensors, **kwargs)


class ToyCausalModel(torch.nn.Module):
    """A real masked-attention module, with deterministic reference logits."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(77)
            self.embedding = torch.nn.Embedding(100, 16)
            self.position = torch.nn.Embedding(128, 16)
            self.projection = torch.nn.Linear(16, 100, bias=False)
        self.config = SimpleNamespace(model_type="qwen3", _attn_implementation="flash_attention_2")
        self.tree_calls = 0

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False, logits_to_keep=None):
        hidden = self.embedding(input_ids) + self.position(position_ids)
        weights = hidden @ hidden.transpose(-1, -2) / 4
        if attention_mask.ndim == 4:
            self.tree_calls += 1
            assert self.config._attn_implementation == "eager"
            additive = attention_mask[:, 0]
        else:
            length = input_ids.shape[1]
            allowed = torch.ones(length, length, dtype=torch.bool).tril()[None] & attention_mask[:, None, :].bool()
            additive = torch.zeros_like(weights).masked_fill(~allowed, -torch.inf)
        hidden = (weights + additive.to(weights.dtype)).softmax(-1) @ hidden
        if logits_to_keep is not None:
            hidden = hidden[:, logits_to_keep]
        return SimpleNamespace(logits=self.projection(hidden))


class Worker:
    _is_ref = True
    _is_lora = False
    ulysses_sequence_parallel_size = 1

    def __init__(self, model=None):
        self.model = ToyCausalModel() if model is None else model
        self.model.eval()
        self.ref_policy = SimpleNamespace(actor_module=self.model, use_fused_kernels=False)
        self.config = SimpleNamespace(model={})
        self.native_calls = 0
        self.inject_native_difference = False

    def compute_ref_log_prob(self, data):
        self.native_calls += 1
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=data.batch["input_ids"],
                             attention_mask=data.batch["attention_mask"],
                             position_ids=data.batch["position_ids"], use_cache=False).logits
            width = data.batch["responses"].shape[1]
            logits = out[:, -width - 1:-1].float().log_softmax(-1)
            scores = logits.gather(-1, data.batch["responses"].unsqueeze(-1)).squeeze(-1)
        if self.inject_native_difference:
            scores = scores + .25
        return Proto({"ref_log_prob": scores.to(getattr(self, "output_dtype", scores.dtype))})


def batch(*, verify=True):
    requests = [
        {"prompt_ids": [3, 4, 5, 6, 7, 8], "target_ids": [31, 32]},
        {"prompt_ids": [3, 4, 5, 6, 7, 9], "target_ids": [33, 34, 35]},
        {"prompt_ids": [3, 4, 5, 6, 7, 10], "target_ids": [36]},
    ]
    return Proto(credit.build_score_tensors(requests, 0), meta_info={
        "trace_gt_scoring": True, "temperature": 1., "trace_prefix_verify": verify,
        "trace_prefix_max_tokens": 64, "trace_prefix_max_mask_bytes": 64 ** 2 * 4,
    })


def compute(worker, data, **kwargs):
    return backend.compute_reference_prefix(worker, data, Proto, device_name="cpu", device_id="cpu", **kwargs)


class Consensus:
    """Model a peer rank's agreed reductions without starting an NPU runtime."""
    def __init__(self, *, rounds=None, remote_unsupported=False, fail_validation=False):
        self.max_calls = 0
        self.rounds = rounds
        self.remote_unsupported = remote_unsupported
        self.fail_validation = fail_validation

    def integer(self, value, operation):
        if operation == "MAX":
            self.max_calls += 1
            if self.max_calls == 2 and self.remote_unsupported:
                return 1
            if self.max_calls == 7 and self.rounds is not None:
                return max(int(value), self.rounds)
        if operation == "MIN" and self.fail_validation:
            return 0
        return int(value)

    def maximum(self, value):
        return float(value)


class PrefixWorkerTests(unittest.TestCase):
    def test_tree_scores_match_native_tokens_and_only_first_call_runs_verification(self):
        worker, data = Worker(), batch()
        expected = worker.compute_ref_log_prob(data).batch["ref_log_prob"]
        worker.native_calls = 0
        output = compute(worker, data)
        mask = data.batch["response_mask"].bool()
        torch.testing.assert_close(output.batch["ref_log_prob"][mask], expected[mask], rtol=1e-5, atol=1e-6)
        metrics = output.meta_info["trace_prefix_metrics"]
        self.assertEqual(metrics["trace/prefix_used"], 1)
        self.assertEqual(metrics["trace/prefix_validation_passed"], 1)
        self.assertGreater(metrics["trace/prefix_saved_tokens"], 0)
        self.assertEqual(worker.native_calls, 1)
        self.assertEqual(worker.model.config._attn_implementation, "flash_attention_2")
        again = compute(worker, data)
        self.assertEqual(again.meta_info["trace_prefix_metrics"]["trace/prefix_used"], 1)
        self.assertEqual(worker.native_calls, 1)
        self.assertEqual(again.batch["ref_log_prob"].dtype, expected.dtype)

    def test_peer_pack_count_adds_dummy_forwards_without_extra_scores(self):
        worker, data = Worker(), batch(verify=False)
        actual = compute(worker, data, consensus=Consensus(rounds=4))
        self.assertEqual(worker.model.tree_calls, 4)
        self.assertEqual(actual.batch["ref_log_prob"].shape, data.batch["responses"].shape)
        metrics = actual.meta_info["trace_prefix_metrics"]
        self.assertEqual(metrics["trace/prefix_forward_rounds"], 4)
        self.assertEqual(metrics["trace/prefix_dummy_forwards"], 4 - metrics["trace/prefix_packs"])

    def test_remote_unsupported_rank_falls_back_before_any_tree_forward(self):
        worker, data = Worker(), batch()
        output = compute(worker, data, consensus=Consensus(remote_unsupported=True))
        self.assertEqual(worker.model.tree_calls, 0)
        self.assertEqual(worker.native_calls, 1)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_unsupported_model"], 1)

    def test_memory_limit_falls_back_without_truncating_or_changing_inputs(self):
        worker, data = Worker(), batch()
        before = {key: tensor.clone() for key, tensor in data.batch.items()}
        data.meta_info["trace_prefix_max_mask_bytes"] = 4
        output = compute(worker, data)
        self.assertEqual(worker.model.tree_calls, 0)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_planning"], 1)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, data.batch[key]))

    def test_validation_failure_returns_native_and_disables_following_tree_calls(self):
        worker, data = Worker(), batch()
        worker.inject_native_difference = True
        expected = worker.compute_ref_log_prob(data).batch["ref_log_prob"]
        worker.native_calls = 0
        output = compute(worker, data)
        self.assertTrue(torch.equal(output.batch["ref_log_prob"], expected))
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_validation"], 1)
        self.assertTrue(worker._trace_prefix_disabled)
        tree_calls = worker.model.tree_calls
        again = compute(worker, data)
        self.assertEqual(worker.model.tree_calls, tree_calls)
        self.assertEqual(again.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_validation_disabled"], 1)

    def test_remote_validation_failure_also_disables_this_rank(self):
        worker, data = Worker(), batch()
        output = compute(worker, data, consensus=Consensus(fail_validation=True))
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_validation_passed"], 0)
        self.assertTrue(worker._trace_prefix_disabled)

    def test_native_output_precision_is_preserved_after_verification(self):
        worker, data = Worker(), batch()
        worker.output_dtype = torch.bfloat16
        output = compute(worker, data)
        self.assertEqual(output.batch["ref_log_prob"].dtype, torch.bfloat16)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_validation_passed"], 1)
        second = compute(worker, data)
        self.assertEqual(second.batch["ref_log_prob"].dtype, torch.bfloat16)
        self.assertEqual(worker.native_calls, 1)

    def test_no_shared_nodes_uses_original_backend(self):
        data = Proto(credit.build_score_tensors([
            {"prompt_ids": [1, 2, 3], "target_ids": [31]},
            {"prompt_ids": [4, 5, 6], "target_ids": [32]},
        ], 0), meta_info={"trace_gt_scoring": True})
        worker = Worker()
        output = compute(worker, data)
        self.assertEqual(worker.model.tree_calls, 0)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_no_sharing"], 1)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_used"], 0)

    def test_unsupported_lora_fused_window_and_rope_do_not_enter_tree(self):
        for setting in ("lora", "fused", "window", "dynamic_rope"):
            with self.subTest(setting=setting):
                worker, data = Worker(), batch()
                if setting == "lora":
                    worker._is_lora = True
                elif setting == "fused":
                    worker.ref_policy.use_fused_kernels = True
                elif setting == "window":
                    worker.model.config.use_sliding_window = True
                else:
                    worker.model.config.rope_scaling = {"rope_type": "dynamic"}
                output = compute(worker, data)
                self.assertEqual(worker.model.tree_calls, 0)
                self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_used"], 0)

    def test_eager_scope_restores_config_and_reshards_when_model_forward_raises(self):
        worker, data = Worker(), batch(verify=False)
        original = worker.model.forward
        def fail(*args, **kwargs):
            if kwargs["attention_mask"].ndim == 4:
                raise RuntimeError("model failure")
            return original(*args, **kwargs)
        worker.model.forward = fail
        resharded = []
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            compute(worker, data, reshard=lambda model: resharded.append(model))
        self.assertEqual(worker.model.config._attn_implementation, "flash_attention_2")
        self.assertEqual(resharded, [worker.model])
        self.assertEqual(worker.native_calls, 0)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "optional real HF model validation")
class RealQwenPrefixWorkerTests(unittest.TestCase):
    def test_qwen2_and_qwen3_match_independent_teacher_forcing(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
        for config_class, model_class in ((Qwen2Config, Qwen2ForCausalLM), (Qwen3Config, Qwen3ForCausalLM)):
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(81)
                config = config_class(vocab_size=100, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      max_position_embeddings=128, attention_dropout=0.)
                config._attn_implementation = "eager"
                model = model_class(config).eval()
                worker, data = Worker(model), batch()
                output = compute(worker, data)
                self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_used"], 1)
                self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_validation_passed"], 1)
                self.assertLessEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_validation_max_diff"], 1e-5)
                self.assertEqual(model.config._attn_implementation, "eager")


def load_reference_refresh_helpers():
    """Execute only worker shard helpers, without importing Ray/NPU dependencies."""
    path = ROOT / "agentflow/verl/diy/workers/fsdp_workers.py"
    tree = ast.parse(path.read_text())
    helpers = [node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name.startswith("_trace_")
               or isinstance(node, ast.Assign) and any(
                   isinstance(target, ast.Name) and target.id == "_TRACE_ACTORS"
                   for target in node.targets)]
    scope = {"Path": Path, "WeakValueDictionary": WeakValueDictionary}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec"), scope)
    return SimpleNamespace(**scope)


refresh = load_reference_refresh_helpers()


class ReferenceSnapshotModel(torch.nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.projection = torch.nn.Linear(width, 2)
        self.register_buffer("persistent", torch.tensor([2.]))
        self.register_buffer("rotary_cache", torch.tensor([4.]), persistent=False)


class FlatModel(torch.nn.Module):
    def __init__(self, names=("first", "second")):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(4.).clone())
        self.weight._fqns = list(names)
        self.weight._shapes = [torch.Size([2]), torch.Size([2])]
        self.weight._numels = (2, 2)
        self.weight._numels_with_padding = (2, 2)
        self.weight._is_padding_mask = (False, False)
        self.weight._shard_param_infos = ((True, 0, 2, 0, 1), (True, 2, 2, 0, 1))
        self.weight._sharded_size = torch.Size([4])
        self.weight._unpadded_unsharded_size = torch.Size([4])
        self.weight._shared_param_infos = ()
        self.weight._local_shard = self.weight.detach()
        self.reshards = 0
        self._handle = SimpleNamespace(flat_param=self.weight, _use_orig_params=False, reshard=self.reshard)

    def reshard(self, free):
        assert free
        self.reshards += 1


class ReferenceRefreshTests(unittest.TestCase):
    def setUp(self):
        refresh._TRACE_ACTORS.clear()

    def test_copies_latest_weights_and_persistent_buffers_without_touching_actor(self):
        actor_model, ref_model = ReferenceSnapshotModel(), ReferenceSnapshotModel().to(dtype=torch.bfloat16)
        actor_model.train()
        actor_model.projection.weight.grad = torch.ones_like(actor_model.projection.weight)
        actor_model.persistent.fill_(7.)
        actor_model.rotary_cache.fill_(8.)
        actor = SimpleNamespace(actor_module_fsdp=actor_model)
        reference = SimpleNamespace(ref_module_fsdp=ref_model, _trace_prefix_verified=True,
                                    _trace_prefix_disabled=True, _trace_prefix_output_dtype=torch.float32)
        before = actor_model.projection.weight.detach().clone()
        before_grad = actor_model.projection.weight.grad.clone()
        source_ptr = actor_model.projection.weight.data_ptr()
        stats = refresh._trace_copy_local_reference(actor, reference, 4)
        self.assertEqual(stats["reference_step"], 4)
        self.assertEqual(reference._trace_reference_step, 4)
        self.assertFalse(ref_model.training)
        self.assertTrue(actor_model.training)
        self.assertEqual(source_ptr, actor_model.projection.weight.data_ptr())
        torch.testing.assert_close(actor_model.projection.weight, before)
        torch.testing.assert_close(actor_model.projection.weight.grad, before_grad)
        torch.testing.assert_close(ref_model.projection.weight, before.to(torch.bfloat16))
        self.assertEqual(ref_model.persistent.item(), 7.)
        self.assertEqual(ref_model.rotary_cache.item(), 4.)
        self.assertFalse(hasattr(reference, "_trace_prefix_verified"))
        self.assertFalse(hasattr(reference, "_trace_prefix_disabled"))
        self.assertFalse(hasattr(reference, "_trace_prefix_output_dtype"))
        with torch.no_grad():
            actor_model.projection.weight.add_(1.)
        torch.testing.assert_close(ref_model.projection.weight, before.to(torch.bfloat16))

    def test_mismatch_is_found_before_first_destination_copy(self):
        actor, reference = ReferenceSnapshotModel(), ReferenceSnapshotModel()
        # Last tensor differs, so first parameter must still remain unchanged.
        reference.persistent = torch.tensor([2., 3.])
        before = reference.projection.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            refresh._trace_copy_local_reference(SimpleNamespace(actor_module_fsdp=actor),
                                         SimpleNamespace(ref_module_fsdp=reference), 3)
        torch.testing.assert_close(reference.projection.weight, before)

    def test_same_size_fsdp_flat_parameters_with_different_order_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "flat parameter layout"):
            refresh._trace_prepare_local_copy(FlatModel(), FlatModel(("second", "first")))

    def test_flat_shards_copy_and_reference_all_gather_cache_is_released(self):
        actor, reference = FlatModel(), FlatModel()
        with torch.no_grad():
            actor.weight.add_(2.)
        refresh._trace_copy_local_reference(SimpleNamespace(actor_module_fsdp=actor),
                                     SimpleNamespace(ref_module_fsdp=reference), 1)
        torch.testing.assert_close(reference.weight, actor.weight)
        self.assertEqual(reference.reshards, 1)
        self.assertEqual(actor.reshards, 0)

    def test_unsharded_or_detached_source_fails_explicitly(self):
        actor, reference = FlatModel(), FlatModel()
        actor.weight._local_shard = torch.zeros_like(actor.weight)
        with self.assertRaisesRegex(RuntimeError, "detached local shard"):
            refresh._trace_prepare_local_copy(actor, reference)

    def test_flat_padding_and_orig_parameter_layout_must_match(self):
        actor, reference = FlatModel(), FlatModel()
        reference.weight._numels_with_padding = (1, 3)
        with self.assertRaisesRegex(ValueError, "flat parameter layout"):
            refresh._trace_prepare_local_copy(actor, reference)
        reference = FlatModel()
        reference._handle._use_orig_params = True
        with self.assertRaisesRegex(ValueError, "use_orig_params"):
            refresh._trace_prepare_local_copy(actor, reference)

    def test_missing_colocated_actor_and_lora_fail_with_actionable_errors(self):
        reference = SimpleNamespace(_is_ref=True, _is_lora=False, rank=0,
                                    config=SimpleNamespace(model=SimpleNamespace(path="original"), ref={}))
        with self.assertRaisesRegex(RuntimeError, "colocated"):
            refresh._trace_colocated_actor(reference)
        reference._is_lora = True
        with self.assertRaisesRegex(ValueError, "non-LoRA"):
            refresh._trace_colocated_actor(reference)

    def test_distinct_initial_reference_model_fails_before_copying(self):
        reference = SimpleNamespace(_is_ref=True, _is_lora=False, rank=0,
                                    config=SimpleNamespace(model=SimpleNamespace(path="actor"),
                                                           ref={"model": {"path": "other-rope-model"}}))
        with self.assertRaisesRegex(ValueError, "same initial actor/reference model"):
            refresh._trace_colocated_actor(reference)

    def test_registry_finds_same_process_rank_with_matching_mesh(self):
        class Worker:
            pass

        mesh = SimpleNamespace(mesh=torch.tensor([0, 1]), mesh_dim_names=("fsdp",), device_type="cpu")
        actor = Worker()
        actor._is_actor, actor._is_lora = True, False
        actor.rank, actor.world_size, actor.device_mesh = 0, 2, mesh
        reference = SimpleNamespace(_is_ref=True, _is_lora=False, rank=0, world_size=2, device_mesh=mesh,
                                    config=SimpleNamespace(model=SimpleNamespace(path="original"), ref={}))
        refresh._trace_register_actor(actor)
        self.assertIs(refresh._trace_colocated_actor(reference), actor)
        reference.world_size = 4
        with self.assertRaisesRegex(ValueError, "world sizes"):
            refresh._trace_colocated_actor(reference)

    def test_empty_shards_supported(self):
        actor, reference = torch.nn.Linear(0, 2), torch.nn.Linear(0, 2)
        plan = refresh._trace_prepare_local_copy(actor, reference)
        self.assertEqual(len(plan), 2)

    def test_tied_parameters_are_not_untied(self):
        actor, reference = ReferenceSnapshotModel(), ReferenceSnapshotModel()
        actor.other = actor.projection
        reference.other = reference.projection
        plan = refresh._trace_prepare_local_copy(actor, reference)
        self.assertEqual(len(plan), 3)  # weight, bias, persistent buffer

    def test_worker_rpcs_delegate_without_distributed_weights_or_optimizer(self):
        tree = ast.parse((ROOT / "agentflow/verl/diy/workers/fsdp_workers.py").read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "ActorRolloutRefWorker")
        methods = {node.name: node for node in worker.body if isinstance(node, ast.FunctionDef)}
        for name in ("refresh_reference_from_actor", "validate_reference_refresh",
                     "save_reference_checkpoint", "load_reference_checkpoint"):
            self.assertIn("Dispatch.ONE_TO_ALL", ast.unparse(methods[name].decorator_list[0]))
        manager = ast.unparse(methods["_reference_checkpoint_manager"])
        self.assertIn("optimizer=None", manager)
        self.assertIn("'save_contents': ['model']", manager)
        self.assertIn("_trace_invalidate_reference_cache", ast.unparse(methods["load_reference_checkpoint"]))

    def test_refresh_rpc_waits_for_actor_offload_before_cpu_reads(self):
        tree = ast.parse((ROOT / "agentflow/verl/diy/workers/fsdp_workers.py").read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "ActorRolloutRefWorker")
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
                      and node.name == "refresh_reference_from_actor")
        method.decorator_list = []
        events = []
        scope = {"get_torch_device": lambda: SimpleNamespace(synchronize=lambda: events.append("sync")),
                 "_trace_colocated_actor": lambda worker: events.append("source"),
                 "_trace_copy_local_reference": lambda *args: events.append("copy")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     "<production-reference-refresh-rpc>", "exec"), scope)
        scope["refresh_reference_from_actor"](object(), 7)
        self.assertEqual(events, ["sync", "source", "copy"])

    def test_production_checkpoint_rpcs_roundtrip_frozen_snapshot_only(self):
        tree = ast.parse((ROOT / "agentflow/verl/diy/workers/fsdp_workers.py").read_text())
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                        and node.name == "ActorRolloutRefWorker")
        names = {"_reference_checkpoint_manager", "save_reference_checkpoint", "load_reference_checkpoint"}
        methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
        for method in methods:
            method.decorator_list = []
        klass = ast.ClassDef(name="Worker", bases=[], keywords=[], body=methods, decorator_list=[])
        module = ast.fix_missing_locations(ast.Module(body=[klass], type_ignores=[]))
        barriers, managers = [], []

        class Manager:
            def __init__(self, **kwargs):
                self.options = kwargs
                managers.append(self)

            def save_checkpoint(self, **kwargs):
                path = Path(kwargs["local_path"])
                path.mkdir(parents=True, exist_ok=True)
                torch.save(self.options["model"].state_dict(), path / "model.pt")

            def load_checkpoint(self, **kwargs):
                weights = torch.load(Path(kwargs["local_path"]) / "model.pt", weights_only=True)
                self.options["model"].load_state_dict(weights)

        scope = {"FSDPCheckpointManager": Manager, "OmegaConf": SimpleNamespace(create=lambda value: value),
                 "dist": SimpleNamespace(barrier=lambda: barriers.append(True)),
                 "_trace_release_reference_full_params": refresh._trace_release_reference_full_params,
                 "_trace_invalidate_reference_cache": refresh._trace_invalidate_reference_cache}
        exec(compile(module, "<production-reference-worker-rpcs>", "exec"), scope)
        worker = scope["Worker"]()
        worker._is_ref, worker._is_lora = True, False
        worker.ref_module_fsdp = ReferenceSnapshotModel()
        worker.processor, worker.tokenizer = None, object()
        before = worker.ref_module_fsdp.projection.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            worker.save_reference_checkpoint(directory, global_step=7)
            with torch.no_grad():
                worker.ref_module_fsdp.projection.weight.add_(10.)
            worker._trace_prefix_verified = True
            worker.load_reference_checkpoint(directory, reference_step=6)
        torch.testing.assert_close(worker.ref_module_fsdp.projection.weight, before)
        self.assertEqual(worker._trace_reference_step, 6)
        self.assertFalse(hasattr(worker, "_trace_prefix_verified"))
        self.assertEqual(len(managers), 1)
        self.assertIsNone(managers[0].options["optimizer"])
        self.assertIsNone(managers[0].options["lr_scheduler"])
        self.assertEqual(managers[0].options["checkpoint_config"],
                         {"load_contents": ["model"], "save_contents": ["model"]})
        self.assertEqual(barriers, [True])


if __name__ == "__main__":
    unittest.main()
