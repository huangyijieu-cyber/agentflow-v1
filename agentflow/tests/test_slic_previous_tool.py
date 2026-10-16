"""CPU regressions for the previous-tool filter and the real QA metadata path."""
import ast
import contextlib
import io
import json
import logging
import os
import re
import sys
import time
import unittest
import uuid
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch

import numpy as np
from pydantic import BaseModel, Field
import torch
import yaml

from test_slic_prompt_filter import PAIR, Proto, run


ROOT = Path(__file__).resolve().parents[2]
NO_TOOL = '__no_previous_tool__'
WEB = 'Web_Search_Tool'
WIKI = 'Wikipedia_Search_Tool'
PYTHON = 'Python_Coder_Tool'
ENABLED = {'slic_match_previous_tool': True}


def extract(path, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def rows(last_turn=4):
    return [(rollout, turn, reward, logp, False)
            for rollout, reward, logp in [('chosen', 1., -1.), ('rejected', 0., -3.)]
            for turn in range(last_turn + 1)]


def turns(pairs, side='a'):
    return [int(p.non_tensor_batch[f'turn_index_list_{side}'][0]) for p in pairs or []]


class PreviousToolPairTests(unittest.TestCase):
    def test_filters_by_previous_tool_and_retains_later_matching_turns(self):
        tools = [NO_TOOL, NO_TOOL, WEB, PYTHON, WEB,
                 NO_TOOL, NO_TOOL, WIKI, PYTHON, PYTHON]
        pairs, trainer, proto = run(rows(), previous_tools=tools,
                                   data_overrides=ENABLED, return_details=True)
        self.assertEqual(turns(pairs), [0, 1, 3])
        self.assertEqual(turns(pairs, 'b'), [0, 1, 3])
        self.assertEqual(trainer._slic_previous_tool_metrics, {
            'replay/slic_prev_tool_candidate_pairs': 5,
            'replay/slic_prev_tool_turn_mismatch_pairs': 0,
            'replay/slic_prev_tool_unknown_pairs': 0,
            'replay/slic_prev_tool_mismatch_pairs': 2,
            'replay/slic_prev_tool_matched_pairs': 3,
            'replay/slic_prev_tool_selected_pairs': 3,
        })
        for pair in pairs:
            self.assertEqual(pair.non_tensor_batch['previous_tool_name_list_a'][0],
                             pair.non_tensor_batch['previous_tool_name_list_b'][0])
        # The complete rollout remains available to GRPO.
        self.assertEqual(len(proto.batch['responses']), 10)
        self.assertEqual(proto.non_tensor_batch['turn_index_list'].tolist(),
                         [row[1] for row in rows()])
        torch.testing.assert_close(proto.batch['token_level_scores'][:, 0],
                                   torch.tensor([row[2] for row in rows()]))

    def test_default_and_disabled_modes_keep_original_pairs(self):
        tools = [WEB] * 5 + [PYTHON] * 5
        for config in ({}, {'slic_match_previous_tool': False}):
            pairs, trainer, _ = run(rows(), previous_tools=tools,
                                   data_overrides=config, return_details=True)
            self.assertEqual(turns(pairs), [0, 1, 2, 3, 4])
            self.assertEqual(trainer._slic_previous_tool_metrics, {})
            self.assertNotIn('previous_tool_name_list_a', pairs[0].non_tensor_batch)

    def test_matching_tools_still_require_same_turn_in_cross_mode(self):
        sample_rows = [('chosen', 2, 1., -1., False), ('rejected', 3, 0., -3., False)]
        config = dict(ENABLED, replay_collection='cross')
        self.assertIsNone(run(sample_rows, previous_tools=[WEB, WEB], data_overrides=config))
        self.assertEqual(len(run(sample_rows, data_overrides={'replay_collection': 'cross'})), 1)

    def test_cross_mode_does_not_cross_pair_the_two_no_tool_boundaries(self):
        pairs = run(rows(2), previous_tools=[NO_TOOL, NO_TOOL, WEB] * 2,
                    data_overrides=dict(ENABLED, replay_collection='cross',
                                        replay_filter={'k_pairs_per_group': 20}))
        self.assertEqual(turns(pairs), [0, 1, 2])
        self.assertEqual(turns(pairs, 'b'), [0, 1, 2])

    def test_unknown_values_are_not_equal_tools(self):
        sample_rows = [('chosen', 2, 1., -1., False), ('rejected', 2, 0., -3., False)]
        for unknown in (None, '', '  ', float('nan')):
            with self.subTest(unknown=unknown):
                self.assertIsNone(run(sample_rows, previous_tools=[unknown, unknown], data_overrides=ENABLED))
                self.assertIsNone(run(sample_rows, previous_tools=[WEB, unknown], data_overrides=ENABLED))

    def test_missing_metadata_column_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'restart rollout workers'):
            run(rows(), data_overrides=ENABLED)

    def test_filter_precedes_top_k_and_does_not_change_baseline(self):
        sample_rows = [('chosen', 2, 1., -1., False),
                       ('wrong_tool', 2, 0., -1.001, False),
                       ('right_tool', 2, 0., -3., False)]
        pairs = run(sample_rows, previous_tools=[WEB, PYTHON, WEB],
                    data_overrides=dict(ENABLED, replay_filter={'k_pairs_per_group': 1}))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].non_tensor_batch['rollout_id_list_b'][0], 'right_tool')
        self.assertAlmostEqual(float(pairs[0].non_tensor_batch['group_baseline'][0]),
                               (-1. - 1.001 - 3.) / 3., places=6)
        self.assertEqual(int(pairs[0].non_tensor_batch['group_size'][0]), 3)

    def test_original_final_stage_and_drop_rules_still_apply(self):
        sample_rows = [('chosen', 2, 1., -1., False), ('rejected', 2, 0., -3., False),
                       ('rejected', 3, 0., -3., True)]
        self.assertIsNone(run(sample_rows, previous_tools=[WEB] * 3, data_overrides=ENABLED))
        dropped = [(rid, turn, score, logp, turn == 4)
                   for rid, turn, score, logp, _ in rows()]
        self.assertEqual(turns(run(dropped, previous_tools=[WEB] * 10, data_overrides=ENABLED)),
                         [0, 1, 2, 3])

    def test_original_length_filter_is_still_applied(self):
        _, trainer, proto = run(rows(0), previous_tools=[WEB, WEB],
                                data_overrides=ENABLED, return_details=True)
        proto.batch['response_mask'][1, 1] = 0
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(PAIR(trainer, [proto]))

    def test_padding_and_missing_drop_mask_do_not_add_forbidden_pairs(self):
        pairs = run(rows(), with_drop_mask=False, previous_tools=[WEB] * 5 + [WEB, WEB, PYTHON, WEB, WEB],
                    data_overrides=ENABLED, n_gpus=8)
        self.assertEqual(len(pairs), 8)
        self.assertEqual(set(turns(pairs)), {0, 1, 3, 4})
        for pair in pairs:
            self.assertEqual(pair.non_tensor_batch['previous_tool_name_list_a'][0],
                             pair.non_tensor_batch['previous_tool_name_list_b'][0])

    def test_other_losses_remain_unchanged_and_non_qa_is_reported(self):
        for algorithm in ('grpo+simpo', 'grpo+abpo', 'grpo+hspo'):
            self.assertEqual(turns(run(rows(), data_overrides=ENABLED, algorithm=algorithm)),
                             [0, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, 'requires data.task=qa'):
            run(rows(), data_overrides=dict(ENABLED, task='webshop'))


# Exercise actual solver/logging/encoding/daemon functions, with model calls and
# distributed containers replaced by CPU stand-ins. No token re-encoding occurs.
APPEND_LOG = extract('agentflow/agentflow/models/planner.py', '_append_log', {'Any': Any})
NextStep = extract('agentflow/agentflow/models/formatters.py', 'NextStep', {'BaseModel': BaseModel})
EXTRACT_TOOL = extract('agentflow/agentflow/models/planner.py', 'extract_context_subgoal_and_tool',
                       dict(Any=Any, Tuple=Tuple, re=re, NextStep=NextStep, robust_json_loads=json.loads))
SOLVE = extract('agentflow/agentflow/solver.py', 'solve', dict(
    Optional=Optional, time=time, nullcontext=contextlib.nullcontext, deepcopy=deepcopy,
    json=json, profiling_enabled=lambda: False, is_search_tool=lambda name: 'Search' in str(name),
    make_json_serializable_truncated=lambda value: value, log_info=lambda *args: None))
Triplet = extract('agentflow/agent_types.py', 'Triplet',
                  dict(Any=Any, Dict=Dict, Optional=Optional, BaseModel=BaseModel, Field=Field))
ENCODE = extract('train-roma/rollout.py', 'encode_logs', {'Triplet': Triplet})


class CpuProto(Proto):
    def __init__(self, batch, non_tensor_batch=None, meta_info=None):
        super().__init__(batch, non_tensor_batch or {}, meta_info)


daemon_scope = dict(torch=torch, np=np, uuid=uuid, List=list,
                    logger=logging.getLogger(__name__), DataProto=CpuProto,
                    TensorDict=lambda values, **kwargs: values)
for function in ('get_left_padded_ids_and_attention_mask', 'get_right_padded_ids_and_attention_mask',
                 'get_train_data_batch'):
    GET_BATCH = extract('agentflow/verl/daemon.py', function, daemon_scope)


class PlannerHarness:
    _append_log = APPEND_LOG
    extract_context_subgoal_and_tool = EXTRACT_TOOL

    def __init__(self, choices):
        self.choices = choices
        self.logs = []
        self.available_tools = [WEB, PYTHON, WIKI]
        self.toolbox_metadata = {tool: {} for tool in self.available_tools}
        self.llm_engine = NS(last_generation_metadata={
            'prompt_token_ids': [101, 102], 'response_token_ids': [201, 202], 'finish_reason': 'stop'})

    def log(self, response):
        self._append_log('prompt', response)
        return response

    def analyze_query(self, *args):
        return self.log('analysis')

    def generate_next_step(self, question, image, analysis, memory, step_count, *args):
        return self.log(NextStep(justification='why', context='context', sub_goal='goal',
                                 tool_name=self.choices[step_count - 1]))

    def generate_final_output(self, *args):
        return self.log('final')


class MemoryHarness:
    def clear(self):
        self.actions = []

    def add_action(self, *args):
        self.actions.append(args)

    def get_all_actions(self):
        return self.actions


def solve(planner, stop_after=None):
    executor = NS(set_query_cache_dir=lambda *args: None, generate_tool_command=lambda *args: 'command',
                  extract_explanation_and_command=lambda *args: ('analysis', 'why', 'command'),
                  execute_tool_command=lambda *args: {'result': 'ok'})
    verifier = NS(verificate_context=lambda *args: args[4],
                  extract_conclusion=lambda step: ('ok', 'STOP' if step == stop_after else 'CONTINUE'))
    solver = NS(task='qa', planner=planner, memory=MemoryHarness(), executor=executor, verifier=verifier,
                output_types=['final'], max_steps=len(planner.choices), max_time=60, max_tokens=16,
                root_cache_dir='unused', verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        result, _ = SOLVE(solver, 'question')
    return result['planner_logs']


def make_batch(rollouts, max_prompt_length=4):
    daemon = NS(is_train=True, _completed_rollouts=rollouts, _total_tasks_queued=len(rollouts),
                _task_id_to_original_sample={rid: {'data_id': 'question'} for rid in rollouts},
                _fillna_reward=lambda rollout: rollout.final_reward, pad_token_id=0)
    batch, _ = GET_BATCH(daemon, max_prompt_length, 2, 'cpu')
    return batch


def rollout(logs, reward):
    # Round-trip the metadata through the actual triplet schema, as over HTTP.
    triplets = [Triplet.model_validate_json(t.model_dump_json()) for t in ENCODE(logs)]
    return NS(triplets=triplets, final_reward=reward, metadata={})


class PreviousToolPipelineTests(unittest.TestCase):
    def test_real_solver_records_preceding_canonical_choice_and_final_tool(self):
        logs = solve(PlannerHarness(['web search tool', 'Python_Coder_Tool']))
        self.assertEqual([log['previous_tool_name'] for log in logs], [NO_TOOL, NO_TOOL, WEB, PYTHON])
        triplets = ENCODE(logs)
        self.assertEqual([t.metadata['previous_tool_name'] for t in triplets], [NO_TOOL, NO_TOOL, WEB, PYTHON])
        for triplet in triplets:
            self.assertEqual(triplet.prompt['token_ids'], [101, 102])
            self.assertEqual(triplet.response['token_ids'], [201, 202])
            self.assertEqual(triplet.metadata['finish_reason'], 'stop')

    def test_early_stop_and_reused_planner_reset_previous_tool(self):
        planner = PlannerHarness([WEB, PYTHON])
        first = solve(planner)
        second = solve(planner, stop_after=1)
        self.assertEqual([log['previous_tool_name'] for log in first], [NO_TOOL, NO_TOOL, WEB, PYTHON])
        self.assertEqual([log['previous_tool_name'] for log in second], [NO_TOOL, NO_TOOL, WEB])

    def test_invalid_selection_is_unknown_instead_of_no_previous_tool(self):
        logs = solve(PlannerHarness(['not a valid tool', PYTHON]))
        self.assertEqual([log['previous_tool_name'] for log in logs], [NO_TOOL, NO_TOOL, None, PYTHON])
        logs = solve(PlannerHarness(['not a valid tool']))
        self.assertIsNone(logs[-1]['previous_tool_name'])

    def test_solver_to_triplet_to_daemon_to_pair_matches_previous_not_current_tool(self):
        chosen = solve(PlannerHarness([WEB, PYTHON]))
        rejected = solve(PlannerHarness([WEB, WIKI]))
        batch = make_batch({'chosen': rollout(chosen, 1.), 'rejected': rollout(rejected, 0.)})
        self.assertEqual(batch.non_tensor_batch['previous_tool_name_list'].tolist(),
                         [NO_TOOL, NO_TOOL, WEB, PYTHON, NO_TOOL, NO_TOOL, WEB, WIKI])
        batch.batch['response_mask'] = torch.ones(8, 2)
        batch.batch['old_log_probs'] = torch.tensor([[-1., -1.]] * 4 + [[-3., -3.]] * 4)
        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['data_id_list']
        batch.non_tensor_batch['traj_uid'] = batch.non_tensor_batch['traj_id_list']
        _, trainer, _ = run(rows(), data_overrides={'slic_match_previous_tool': False}, return_details=True)
        trainer.config.data.update(ENABLED)
        with contextlib.redirect_stdout(io.StringIO()):
            pairs = PAIR(trainer, [batch])
        # Different choices at turn 2 are allowed: their previous choice is WEB.
        # Final turn 3 is rejected because its preceding choices differ.
        self.assertEqual(turns(pairs), [0, 1, 2])
        self.assertEqual(turns(pairs, 'b'), [0, 1, 2])
        self.assertEqual(len(batch.batch['responses']), 8)

    def test_daemon_metadata_alignment_survives_empty_trace_and_prompt_drop(self):
        logs = solve(PlannerHarness([WEB, PYTHON]))
        triplets = ENCODE(logs)
        triplets[1].prompt['token_ids'] = []
        triplets[1].response['token_ids'] = []
        triplets[2].prompt['token_ids'] = list(range(10))
        batch = make_batch({'chosen': NS(triplets=triplets, final_reward=1., metadata={})})
        self.assertEqual(batch.non_tensor_batch['turn_index_list'].tolist(), [0, 2, 3])
        self.assertEqual(batch.non_tensor_batch['previous_tool_name_list'].tolist(), [NO_TOOL, WEB, PYTHON])
        self.assertEqual(batch.batch['is_drop_mask'].tolist(), [False, True, False])

    def test_legacy_triplets_preserve_unknown_metadata(self):
        legacy = Triplet(prompt={'token_ids': [101]}, response={'token_ids': [201]})
        batch = make_batch({'old': NS(triplets=[legacy], final_reward=0., metadata={})})
        self.assertEqual(batch.non_tensor_batch['previous_tool_name_list'].tolist(), [None])


class PreviousToolConfigTests(unittest.TestCase):
    def test_configs_enable_filter_and_environment_can_disable_it(self):
        load = extract('train-roma/utils.py', 'load_and_set_env_from_yaml', dict(os=os, sys=sys, yaml=yaml))
        for filename in ('config.yaml', 'config_grpo+slic.yaml'):
            for override in (None, 'false'):
                with self.subTest(filename=filename, override=override), patch.dict(os.environ):
                    os.environ.pop('SLIC_MATCH_PREVIOUS_TOOL', None)
                    if override:
                        os.environ['SLIC_MATCH_PREVIOUS_TOOL'] = override
                    with contextlib.redirect_stdout(io.StringIO()):
                        config = load(ROOT / 'train-roma' / filename)
                    value = config['python_args']['+data.slic_match_previous_tool']
                    self.assertEqual(yaml.safe_load(os.path.expandvars(value)), override is None)
                    self.assertEqual(config['env']['TOOL_STEPS'], '5')
                    self.assertNotIn('+data.slic_turn_scope', config['python_args'])


if __name__ == '__main__':
    unittest.main()
