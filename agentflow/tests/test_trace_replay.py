"""Replay diagnostics use pair identities, including padded duplicate entries."""
import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


class Proto:
    def __init__(self, identity, born_step):
        self.non_tensor_batch = {key: np.array([value]) for key, value in {
            "uid": "q", "rollout_id_list_a": identity, "turn_index_list_a": 0,
            "rollout_id_list_b": "rejected", "turn_index_list_b": 2,
            "born_step": born_step,
        }.items()}

    @classmethod
    def concat(cls, selected):
        return selected


ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "agentflow/verl/trainer.py"
trainer_class = next(n for n in ast.parse(path.read_text()).body
                     if isinstance(n, ast.ClassDef) and n.name == "AgentFlowTrainer")
method = next(n for n in trainer_class.body if isinstance(n, ast.FunctionDef) and n.name == "_sample_dpo_data")
scope = {"np": np, "deque": deque, "DataProto": Proto}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
sample = scope[method.name]


class ReplayMetricTests(unittest.TestCase):
    def trainer(self, enabled=True):
        return SimpleNamespace(global_steps=10, algorithm="slic",
                               config=SimpleNamespace(data={"trace": {"enabled": enabled}}))

    def test_duplicate_queue_entries_do_not_count_as_unique_pairs(self):
        pair = Proto("same", 7)
        queue = deque([pair, pair, Proto("different", 9)])
        trainer = self.trainer()
        selected = sample(trainer, queue, bs=3)
        self.assertEqual(len(selected), 3)
        metrics = trainer._trace_replay_metrics
        self.assertEqual(metrics["trace/replay/buffer_unique_pairs"], 2)
        self.assertEqual(metrics["trace/replay/unique_sampled_pairs"], 2)
        self.assertAlmostEqual(metrics["trace/replay/sample_unique_pair_rate"], 2 / 3)
        self.assertEqual(metrics["trace/replay/sample_age_p50"], 3)
        self.assertEqual(metrics["trace/replay/sample_age_max"], 3)

    def test_replacement_sampling_and_metric_collection_do_not_change_draws(self):
        queue = deque([Proto("one", 7), Proto("two", 9)])
        enabled, disabled = self.trainer(), self.trainer(False)
        np.random.seed(53)
        first = sample(enabled, queue, bs=5)
        np.random.seed(53)
        second = sample(disabled, queue, bs=5)
        self.assertEqual([id(p) for p in first], [id(p) for p in second])
        self.assertEqual(enabled._trace_replay_metrics["trace/replay/unique_sampled_pairs"], 2)
        self.assertEqual(enabled._trace_replay_metrics["trace/replay/sample_unique_pair_rate"], 0.4)
        self.assertEqual(len(queue), 2)

    def test_empty_queue_clears_previous_sampling_counts(self):
        trainer = self.trainer()
        sample(trainer, deque([Proto("one", 9)]), bs=1)
        self.assertIsNone(sample(trainer, deque(), bs=1))
        self.assertEqual(trainer._trace_replay_metrics["trace/replay/sampled_pairs"], 0)
        self.assertNotIn("trace/replay/sample_age_p50", trainer._trace_replay_metrics)


if __name__ == "__main__":
    unittest.main()
