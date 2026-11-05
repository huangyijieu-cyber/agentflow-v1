"""Opt-in CPU collective check for unequal tree plans and rank-wide fallback.

Run TRACE_TEST_PREFIX_DISTRIBUTED=1 with this test explicitly selected. It uses
two local Gloo processes; it does not substitute for Ascend/FSDP validation.
"""

import datetime
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist

import test_trace_prefix_worker as fixtures


class CollectiveModel(fixtures.ToyCausalModel):
    def forward(self, *args, **kwargs):
        # Model the collective that every FSDP forward must enter in order.
        mark = torch.ones((), dtype=torch.int64)
        dist.all_reduce(mark)
        assert mark.item() == 2
        return super().forward(*args, **kwargs)


def _rank_main(rank, rendezvous, directory):
    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=40))
    try:
        worker, data = fixtures.Worker(CollectiveModel()), fixtures.batch(verify=True)
        # The same row count produces different tree pack counts on each rank.
        data.meta_info["trace_prefix_max_tokens"] = 64 if rank == 0 else 9
        first = fixtures.compute(worker, data)
        metrics = first.meta_info["trace_prefix_metrics"]
        assert metrics["trace/prefix_used"] == 1
        assert metrics["trace/prefix_validation_passed"] == 1
        assert metrics["trace/prefix_dummy_forwards"] > 0
        assert worker.model.tree_calls == metrics["trace/prefix_forward_rounds"]
        second = fixtures.compute(worker, data)
        assert second.meta_info["trace_prefix_metrics"]["trace/prefix_used"] == 1
        assert worker.native_calls == 1
        if rank == 1:
            worker._is_lora = True
        before = worker.model.tree_calls
        fallback = fixtures.compute(worker, data)
        assert fallback.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_unsupported_model"] == 1
        assert worker.model.tree_calls == before
        assert worker.native_calls == 2
        peers = [None, None]
        dist.all_gather_object(peers, metrics)
        assert peers[0] == peers[1]
        Path(directory, f"rank{rank}.json").write_text(json.dumps(metrics))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("TRACE_TEST_PREFIX_DISTRIBUTED") == "1"
                     and dist.is_available() and dist.is_gloo_available(),
                     "opt-in local two-rank Gloo check")
class PrefixDistributedTests(unittest.TestCase):
    def test_unequal_local_plans_synchronize_and_peer_fallback_is_consistent(self):
        with tempfile.TemporaryDirectory(prefix="trace-prefix-gloo-") as directory:
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=_rank_main,
                         args=(rank, str(Path(directory, "rendezvous")), directory)) for rank in range(2)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(55)
                for process in processes:
                    self.assertFalse(process.is_alive(), "collective schedule did not complete")
                    self.assertEqual(process.exitcode, 0)
                outputs = [json.loads(Path(directory, f"rank{rank}.json").read_text()) for rank in range(2)]
                self.assertEqual(outputs[0], outputs[1])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join()


def _reference_rank_main(rank, rendezvous, output):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, fully_shard

    dist.init_process_group("gloo", rank=rank, world_size=2, init_method="file://" + rendezvous,
                            timeout=datetime.timedelta(seconds=40))
    try:
        torch.manual_seed(123)
        actor = FSDP(torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4)),
                     device_id=torch.device("cpu"))
        reference = FSDP(torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4)).to(torch.bfloat16),
                         device_id=torch.device("cpu"))
        fixtures.refresh._trace_prepare_local_copy(actor, reference)  # first forward has not run
        x = torch.ones(2, 4)
        actor(x).sum().backward()
        optimizer = torch.optim.SGD(actor.parameters(), lr=.03)
        optimizer.step()
        reference(x.to(torch.bfloat16))  # cached unsharded reference has existed
        # Same move/pointer update as VERL 0.6.0 offload_fsdp_model_to_cpu.
        for handle in actor._all_handles:
            handle.flat_param_to(torch.device("cpu"), non_blocking=True)
            handle.flat_param._local_shard = handle.flat_param.data
        reference_worker = SimpleNamespace(ref_module_fsdp=reference)
        actor_worker = SimpleNamespace(actor_module_fsdp=actor)
        fixtures.refresh._trace_copy_local_reference(actor_worker, reference_worker, 5)
        torch.testing.assert_close(reference(x.to(torch.bfloat16)).float(), actor(x).float(), rtol=.03, atol=.03)
        original = reference(x.to(torch.bfloat16)).detach().clone()
        optimizer.zero_grad()
        actor(x).sum().backward()
        optimizer.step()
        torch.testing.assert_close(reference(x.to(torch.bfloat16)), original)
        fixtures.refresh._trace_copy_local_reference(actor_worker, reference_worker, 6)
        torch.testing.assert_close(reference(x.to(torch.bfloat16)).float(), actor(x).float(), rtol=.03, atol=.3)

        # FSDP2 uses DTensor local shards and can keep full parameters after
        # forward. Explicit reshard in refresh must restore the sharded source.
        mesh = init_device_mesh("cpu", (2,))
        source2 = torch.nn.Linear(4, 4)
        target2 = torch.nn.Linear(4, 4).to(torch.bfloat16)
        fully_shard(source2, mesh=mesh, reshard_after_forward=False)
        fully_shard(target2, mesh=mesh, reshard_after_forward=False)
        target2(x.to(torch.bfloat16))
        source2(x).sum().backward()
        torch.optim.SGD(source2.parameters(), lr=.03).step()
        fixtures.refresh._trace_copy_local_reference(SimpleNamespace(actor_module_fsdp=source2),
                                     SimpleNamespace(ref_module_fsdp=target2), 7)
        for source, target in zip(source2.parameters(), target2.parameters()):
            torch.testing.assert_close(target.full_tensor(), source.full_tensor().to(torch.bfloat16))
        torch.testing.assert_close(target2(x.to(torch.bfloat16)).float(), source2(x).float(), rtol=.03, atol=.03)
        Path(output, f"rank{rank}.json").write_text(json.dumps({"fsdp1": True, "fsdp2_shards": True}))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("TRACE_TEST_REFERENCE_DISTRIBUTED") == "1"
                     and dist.is_available() and dist.is_gloo_available(), "opt-in two-rank CPU FSDP check")
class ReferenceDistributedTests(unittest.TestCase):
    def test_actor_update_and_frozen_reference_shard_refresh(self):
        with tempfile.TemporaryDirectory(prefix="trace-reference-gloo-") as directory:
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=_reference_rank_main,
                         args=(rank, str(Path(directory, "rendezvous")), directory)) for rank in range(2)]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(55)
                for process in processes:
                    self.assertFalse(process.is_alive(), "reference refresh did not complete")
                    self.assertEqual(process.exitcode, 0)
                outputs = [json.loads(Path(directory, f"rank{rank}.json").read_text()) for rank in range(2)]
                self.assertEqual(outputs[0], outputs[1])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join()


if __name__ == "__main__":
    unittest.main()
