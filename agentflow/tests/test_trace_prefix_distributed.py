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


if __name__ == "__main__":
    unittest.main()
