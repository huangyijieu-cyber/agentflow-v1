"""Exercise real SLiC pairing without importing distributed worker runtimes."""
import ast
import contextlib
import io
import random
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


class Proto:
    def __init__(self, batch, non_tensor_batch, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info or {}

    @classmethod
    def from_single_dict(cls, values):
        return cls(
            {key: value for key, value in values.items() if isinstance(value, torch.Tensor)},
            {key: value for key, value in values.items() if not isinstance(value, torch.Tensor)},
        )


def load_pair_function():
    path = ROOT / 'agentflow/verl/trainer.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AgentFlowTrainer')
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_trans_to_dpo_batch')
    scope = dict(torch=torch, np=np, random=random, defaultdict=defaultdict, DataProto=Proto)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    return scope[fn.name]


PAIR = load_pair_function()


def run(rows, with_drop_mask=True):
    # Each row is rollout_id, turn, reward, mean log-prob, is_dropped.
    size = len(rows)
    ids = torch.arange(size).reshape(size, 1).repeat(1, 2)
    batch = {key: ids.clone() for key in ['input_ids', 'position_ids', 'prompts', 'responses']}
    batch['attention_mask'] = torch.ones(size, 2)
    batch['response_mask'] = torch.ones(size, 2)
    batch['token_level_scores'] = torch.tensor([[row[2], 0.] for row in rows])
    batch['old_log_probs'] = torch.tensor([[row[3], row[3]] for row in rows])
    if with_drop_mask:
        batch['is_drop_mask'] = torch.tensor([row[4] for row in rows], dtype=torch.bool)
    non = {key: np.array(['q'] * size) for key in ['uid', 'data_id_list']}
    non.update({key: np.array([row[0] for row in rows]) for key in ['rollout_id_list', 'traj_id_list', 'traj_uid']})
    non['turn_index_list'] = np.array([row[1] for row in rows])
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
            data={'task': 'qa', 'replay_collection': 'uid-turn'},
        ),
        use_reference_policy=False,
        global_steps=1,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return PAIR(trainer, [Proto(batch, non, {'temperature': 0.7})])


class SlicPromptFilterTests(unittest.TestCase):
    def test_dropped_rejected_cannot_create_pair(self):
        self.assertIsNone(run([('chosen', 0, 1., -1., False), ('rejected', 0, 0., -2., True)]))

    def test_dropped_chosen_cannot_create_pair(self):
        self.assertIsNone(run([('chosen', 0, 1., -1., True), ('rejected', 0, 0., -2., False)]))

    def test_dropped_sample_does_not_pollute_baseline(self):
        pairs = run([('chosen', 0, 1., -1., False), ('rejected', 0, 0., -3., False),
                     ('dropped', 0, 1., -100., True)])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(float(pairs[0].non_tensor_batch['group_baseline'][0]), -2.)
        self.assertEqual(int(pairs[0].non_tensor_batch['group_size'][0]), 2)
        self.assertEqual(pairs[0].non_tensor_batch['rollout_id_list_a'][0], 'chosen')
        self.assertEqual(pairs[0].non_tensor_batch['rollout_id_list_b'][0], 'rejected')

    def test_dropped_last_turn_does_not_reclassify_intermediate_turn_as_answer(self):
        rows = [('short', 0, 1., -1., False), ('long', 0, 0., -3., False),
                ('long', 1, 0., -5., True)]
        self.assertIsNone(run(rows))

    def test_no_drop_field_remains_supported(self):
        rows = [('chosen', 0, 1., -1., False), ('rejected', 0, 0., -3., False)]
        pairs = run(rows, with_drop_mask=False)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(float(pairs[0].non_tensor_batch['group_baseline'][0]), -2.)


if __name__ == '__main__':
    unittest.main()
