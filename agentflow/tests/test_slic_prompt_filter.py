"""Exercise real SLiC pairing without importing distributed worker runtimes."""
import ast
import contextlib
import io
import os
import random
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml


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


def run(rows, with_drop_mask=True, *, data_overrides=None, algorithm='grpo+slic',
        return_details=False, n_gpus=1):
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
    data_config = {'task': 'qa', 'replay_collection': 'uid-turn'}
    data_config.update(data_overrides or {})
    proto = Proto(batch, non, {'temperature': 0.7})
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            trainer=SimpleNamespace(n_gpus_per_node=n_gpus, nnodes=1),
            data=data_config,
        ),
        algorithm=algorithm,
        use_reference_policy=False,
        global_steps=1,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        pairs = PAIR(trainer, [proto])
    return (pairs, trainer, proto) if return_details else pairs


def qa_rows(last_turns=(4, 4)):
    return [(rollout, turn, reward, logp, False)
            for rollout, last, reward, logp in zip(
                ('chosen', 'rejected'), last_turns, (1., 0.), (-1., -3.))
            for turn in range(last + 1)]


def pair_turns(pairs, side='a'):
    return [int(pair.non_tensor_batch[f'turn_index_list_{side}'][0])
            for pair in pairs or []]


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


class SlicTurnScopeTests(unittest.TestCase):
    scope = {'slic_turn_scope': 'analysis_first_tool_final'}

    def test_only_analysis_first_tool_and_final_are_paired(self):
        pairs, trainer, proto = run(qa_rows(), data_overrides=self.scope, return_details=True)
        self.assertEqual(pair_turns(pairs), [0, 1, 4])
        self.assertEqual(pair_turns(pairs, 'b'), [0, 1, 4])
        self.assertEqual(trainer._slic_turn_metrics, {
            'replay/slic_candidate_turns': 10,
            'replay/slic_filtered_turns': 4,
            'replay/slic_pairs_analysis': 1,
            'replay/slic_pairs_first_tool': 1,
            'replay/slic_pairs_middle': 0,
            'replay/slic_pairs_final': 1,
        })
        # Pair selection must leave every original turn available to GRPO.
        self.assertEqual(proto.non_tensor_batch['turn_index_list'].tolist(),
                         [row[1] for row in qa_rows()])
        torch.testing.assert_close(proto.batch['token_level_scores'][:, 0],
                                   torch.tensor([row[2] for row in qa_rows()]))
        self.assertEqual(len(proto.batch['responses']), 10)
        self.assertTrue(all(p.meta_info['temperature'] == 0.7 for p in pairs))

    def test_default_and_explicit_all_keep_intermediate_turns(self):
        for config in ({}, {'slic_turn_scope': 'all'}):
            with self.subTest(config=config):
                pairs, trainer, _ = run(qa_rows(), data_overrides=config, return_details=True)
                self.assertEqual(pair_turns(pairs), [0, 1, 2, 3, 4])
                self.assertEqual(trainer._slic_turn_metrics['replay/slic_filtered_turns'], 0)
                self.assertEqual(trainer._slic_turn_metrics['replay/slic_pairs_middle'], 2)

    def test_single_tool_step_matches_all_scope(self):
        rows = qa_rows((2, 2))
        all_pairs = run(rows)
        scoped_pairs = run(rows, data_overrides=self.scope)
        self.assertEqual(pair_turns(scoped_pairs), [0, 1, 2])
        for expected, actual in zip(all_pairs, scoped_pairs):
            for key in expected.batch:
                torch.testing.assert_close(expected.batch[key], actual.batch[key])
            for key in expected.non_tensor_batch:
                np.testing.assert_array_equal(expected.non_tensor_batch[key], actual.non_tensor_batch[key])

    def test_final_turn_uses_each_rollout_length_and_existing_pairing_rules(self):
        # Turn 2 is final only for the short rollout; the long rollout's turn 2
        # must not be paired with it. Different final indices stay unpaired.
        pairs = run(qa_rows((2, 4)), data_overrides=self.scope)
        self.assertEqual(pair_turns(pairs), [0, 1])
        self.assertEqual(pair_turns(pairs, 'b'), [0, 1])

    def test_dropped_finals_do_not_promote_middle_turns(self):
        rows = [(rid, turn, reward, logp, turn == 4)
                for rid, turn, reward, logp, _ in qa_rows()]
        pairs, trainer, _ = run(rows, data_overrides=self.scope, return_details=True)
        self.assertEqual(pair_turns(pairs), [0, 1])
        self.assertEqual(trainer._slic_turn_metrics['replay/slic_pairs_final'], 0)
        self.assertEqual(trainer._slic_turn_metrics['replay/slic_filtered_turns'], 4)

    def test_excluded_turns_do_not_enter_baseline_or_group_size(self):
        rows = [(rid, turn, reward, -100. if turn == 2 else logp, dropped)
                for rid, turn, reward, logp, dropped in qa_rows((3, 3))]
        config = dict(self.scope, replay_collection='cross',
                      replay_filter={'k_pairs_per_group': 20})
        pairs = run(rows, data_overrides=config)
        self.assertTrue(pairs)
        self.assertNotIn(2, pair_turns(pairs))
        self.assertNotIn(2, pair_turns(pairs, 'b'))
        for pair in pairs:
            self.assertEqual(float(pair.non_tensor_batch['group_baseline'][0]), -2.)
            self.assertEqual(int(pair.non_tensor_batch['group_size'][0]), 6)

    def test_scope_without_drop_mask_and_with_multiple_partitions(self):
        pairs = run(qa_rows(), with_drop_mask=False, data_overrides=self.scope, n_gpus=8)
        self.assertEqual(len(pairs), 8)
        self.assertEqual(set(pair_turns(pairs)), {0, 1, 4})
        self.assertEqual(set(pair_turns(pairs, 'b')), {0, 1, 4})

    def test_other_preference_losses_are_not_filtered(self):
        for algorithm in ('grpo+abpo', 'grpo+simpo', 'grpo+hspo'):
            with self.subTest(algorithm=algorithm):
                pairs, trainer, _ = run(qa_rows(), algorithm=algorithm,
                                        data_overrides=self.scope, return_details=True)
                self.assertEqual(pair_turns(pairs), [0, 1, 2, 3, 4])
                self.assertEqual(trainer._slic_turn_metrics, {})

    def test_unknown_scope_and_non_qa_scope_raise(self):
        with self.assertRaisesRegex(ValueError, 'Invalid data.slic_turn_scope'):
            run(qa_rows(), data_overrides={'slic_turn_scope': 'typo'})
        with self.assertRaisesRegex(ValueError, 'requires data.task=qa'):
            run(qa_rows(), data_overrides=dict(self.scope, task='webshop'))
        # The all-turn mode remains available for other tasks.
        self.assertEqual(pair_turns(run(qa_rows(), data_overrides={'task': 'webshop'})),
                         [0, 1, 2, 3, 4])

    def test_empty_history_resets_scope_metrics(self):
        _, trainer, _ = run(qa_rows(), data_overrides=self.scope, return_details=True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(PAIR(trainer, []))
        self.assertEqual(trainer._slic_turn_metrics, {})


class SlicTurnConfigTests(unittest.TestCase):
    def test_training_configs_enable_scope_and_allow_environment_override(self):
        path = ROOT / 'train-roma/utils.py'
        fn = next(node for node in ast.parse(path.read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == 'load_and_set_env_from_yaml')
        namespace = dict(os=os, sys=sys, yaml=yaml)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
        load = namespace[fn.name]
        for filename in ('config.yaml', 'config_grpo+slic.yaml'):
            for override in (None, 'all'):
                with self.subTest(filename=filename, override=override), patch.dict(os.environ):
                    os.environ.pop('SLIC_TURN_SCOPE', None)
                    if override:
                        os.environ['SLIC_TURN_SCOPE'] = override
                    with contextlib.redirect_stdout(io.StringIO()):
                        config = load(ROOT / 'train-roma' / filename)
                    value = config['python_args']['+data.slic_turn_scope']
                    self.assertEqual(os.path.expandvars(value), override or 'analysis_first_tool_final')
                    self.assertEqual(config['env']['TOOL_STEPS'], '5')
                    self.assertEqual(config['env']['OUTPUT_TYPE'], 'final')


if __name__ == '__main__':
    unittest.main()
