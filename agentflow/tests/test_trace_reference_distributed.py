"""Opt-in two-rank CPU FSDP/DTensor refresh integration checks."""

import datetime
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, fully_shard


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "reference_refresh_distributed_module", ROOT / "agentflow/agentflow/trace/reference_refresh.py")
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


def _rank_main(rank, rendezvous, output):
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method="file://" + rendezvous,
                            timeout=datetime.timedelta(seconds=40))
    try:
        torch.manual_seed(123)
        actor = FSDP(torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4)),
                     device_id=torch.device("cpu"))
        reference = FSDP(torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4)).to(torch.bfloat16),
                         device_id=torch.device("cpu"))
        refresh.prepare_local_copy(actor, reference)  # first forward has not run
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
        refresh.copy_local_reference(actor_worker, reference_worker, 5)
        torch.testing.assert_close(reference(x.to(torch.bfloat16)).float(), actor(x).float(), rtol=.03, atol=.03)
        original = reference(x.to(torch.bfloat16)).detach().clone()
        optimizer.zero_grad()
        actor(x).sum().backward()
        optimizer.step()
        torch.testing.assert_close(reference(x.to(torch.bfloat16)), original)
        refresh.copy_local_reference(actor_worker, reference_worker, 6)
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
        refresh.copy_local_reference(SimpleNamespace(actor_module_fsdp=source2),
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
            processes = [context.Process(target=_rank_main,
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
