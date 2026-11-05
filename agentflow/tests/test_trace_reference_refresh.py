"""CPU contracts for local reference snapshots, independent of Ray/NPU imports."""

import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "reference_refresh_test_module", ROOT / "agentflow/agentflow/trace/reference_refresh.py")
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


class Model(torch.nn.Module):
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
        refresh._ACTORS.clear()

    def test_copies_latest_weights_and_persistent_buffers_without_touching_actor(self):
        actor_model, ref_model = Model(), Model().to(dtype=torch.bfloat16)
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
        stats = refresh.copy_local_reference(actor, reference, 4)
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
        actor, reference = Model(), Model()
        # Last tensor differs, so first parameter must still remain unchanged.
        reference.persistent = torch.tensor([2., 3.])
        before = reference.projection.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            refresh.copy_local_reference(SimpleNamespace(actor_module_fsdp=actor),
                                         SimpleNamespace(ref_module_fsdp=reference), 3)
        torch.testing.assert_close(reference.projection.weight, before)

    def test_same_size_fsdp_flat_parameters_with_different_order_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "flat parameter layout"):
            refresh.prepare_local_copy(FlatModel(), FlatModel(("second", "first")))

    def test_flat_shards_copy_and_reference_all_gather_cache_is_released(self):
        actor, reference = FlatModel(), FlatModel()
        with torch.no_grad():
            actor.weight.add_(2.)
        refresh.copy_local_reference(SimpleNamespace(actor_module_fsdp=actor),
                                     SimpleNamespace(ref_module_fsdp=reference), 1)
        torch.testing.assert_close(reference.weight, actor.weight)
        self.assertEqual(reference.reshards, 1)
        self.assertEqual(actor.reshards, 0)

    def test_unsharded_or_detached_source_fails_explicitly(self):
        actor, reference = FlatModel(), FlatModel()
        actor.weight._local_shard = torch.zeros_like(actor.weight)
        with self.assertRaisesRegex(RuntimeError, "detached local shard"):
            refresh.prepare_local_copy(actor, reference)

    def test_flat_padding_and_orig_parameter_layout_must_match(self):
        actor, reference = FlatModel(), FlatModel()
        reference.weight._numels_with_padding = (1, 3)
        with self.assertRaisesRegex(ValueError, "flat parameter layout"):
            refresh.prepare_local_copy(actor, reference)
        reference = FlatModel()
        reference._handle._use_orig_params = True
        with self.assertRaisesRegex(ValueError, "use_orig_params"):
            refresh.prepare_local_copy(actor, reference)

    def test_missing_colocated_actor_and_lora_fail_with_actionable_errors(self):
        reference = SimpleNamespace(_is_ref=True, _is_lora=False, rank=0,
                                    config=SimpleNamespace(model=SimpleNamespace(path="original"), ref={}))
        with self.assertRaisesRegex(RuntimeError, "colocated"):
            refresh.colocated_actor(reference)
        reference._is_lora = True
        with self.assertRaisesRegex(ValueError, "non-LoRA"):
            refresh.colocated_actor(reference)

    def test_distinct_initial_reference_model_fails_before_copying(self):
        reference = SimpleNamespace(_is_ref=True, _is_lora=False, rank=0,
                                    config=SimpleNamespace(model=SimpleNamespace(path="actor"),
                                                           ref={"model": {"path": "other-rope-model"}}))
        with self.assertRaisesRegex(ValueError, "same initial actor/reference model"):
            refresh.colocated_actor(reference)

    def test_registry_finds_same_process_rank_with_matching_mesh(self):
        class Worker:
            pass

        mesh = SimpleNamespace(mesh=torch.tensor([0, 1]), mesh_dim_names=("fsdp",), device_type="cpu")
        actor = Worker()
        actor._is_actor, actor._is_lora = True, False
        actor.rank, actor.world_size, actor.device_mesh = 0, 2, mesh
        reference = SimpleNamespace(_is_ref=True, _is_lora=False, rank=0, world_size=2, device_mesh=mesh,
                                    config=SimpleNamespace(model=SimpleNamespace(path="original"), ref={}))
        refresh.register_actor(actor)
        self.assertIs(refresh.colocated_actor(reference), actor)
        reference.world_size = 4
        with self.assertRaisesRegex(ValueError, "world sizes"):
            refresh.colocated_actor(reference)

    def test_empty_shards_supported(self):
        actor, reference = torch.nn.Linear(0, 2), torch.nn.Linear(0, 2)
        plan = refresh.prepare_local_copy(actor, reference)
        self.assertEqual(len(plan), 2)

    def test_tied_parameters_are_not_untied(self):
        actor, reference = Model(), Model()
        actor.other = actor.projection
        reference.other = reference.projection
        plan = refresh.prepare_local_copy(actor, reference)
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
        self.assertIn("invalidate_reference_cache", ast.unparse(methods["load_reference_checkpoint"]))

    def test_refresh_rpc_waits_for_actor_offload_before_cpu_reads(self):
        tree = ast.parse((ROOT / "agentflow/verl/diy/workers/fsdp_workers.py").read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "ActorRolloutRefWorker")
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
                      and node.name == "refresh_reference_from_actor")
        method.decorator_list = []
        events = []
        scope = {"get_torch_device": lambda: SimpleNamespace(synchronize=lambda: events.append("sync"))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                     "<production-reference-refresh-rpc>", "exec"), scope)
        with patch.dict(sys.modules, {"agentflow.trace.reference_refresh": refresh}), \
                patch.object(refresh, "colocated_actor", side_effect=lambda worker: events.append("source")), \
                patch.object(refresh, "copy_local_reference", side_effect=lambda *args: events.append("copy")):
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
                 "dist": SimpleNamespace(barrier=lambda: barriers.append(True))}
        exec(compile(module, "<production-reference-worker-rpcs>", "exec"), scope)
        worker = scope["Worker"]()
        worker._is_ref, worker._is_lora = True, False
        worker.ref_module_fsdp = Model()
        worker.processor, worker.tokenizer = None, object()
        before = worker.ref_module_fsdp.projection.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                sys.modules, {"agentflow.trace.reference_refresh": refresh}):
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
