"""CPU tests of actual QA functions; no vLLM/VERL workers or RL training."""
import ast
import asyncio
import contextlib
import importlib.util
import io
import json
import os
import re
import string
import tempfile
import unicodedata
import unittest
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, Optional
from unittest.mock import patch

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[2]


def extract(path, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), scope)
    return scope[name]


def payload():
    # vLLM 0.11 native protocol. Intentionally no top-level response_token_ids.
    return {'id': 'test', 'object': 'chat.completion', 'created': 1, 'model': 'test',
            'prompt_token_ids': [151644, 872, 198],
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '2'},
                         'finish_reason': 'stop', 'token_ids': [17, 151645]}]}


class NativeTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generate = staticmethod(extract('agentflow/agentflow/engine/vllm.py', '_generate_text', {'os': os}))
        cls.append = staticmethod(extract('agentflow/agentflow/models/planner.py', '_append_log', {'Any': Any}))
        triplet = extract('agentflow/agent_types.py', 'Triplet', dict(Any=Any, Dict=Dict, Optional=Optional, BaseModel=BaseModel, Field=Field))
        cls.encode = staticmethod(extract('train-roma/rollout.py', 'encode_logs', {'Triplet': triplet}))

    def run_chain(self, data):
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=data)
        with OpenAI(api_key='test', base_url='http://serving.test/v1', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            engine = NS(client=client, model_string='test', system_prompt='system', use_cache=False)
            with contextlib.redirect_stdout(io.StringIO()):
                text = self.generate(engine, 'question', max_tokens=16)
            planner = NS(llm_engine=engine, logs=[])
            self.append(planner, 'question', text)
        return requests, engine, planner, text

    def test_native_http_to_planner_to_triplet_keeps_eos(self):
        requests, engine, planner, text = self.run_chain(payload())
        self.assertIs(requests[0]['return_token_ids'], True)
        self.assertIs(requests[0]['stream'], False)
        self.assertEqual(requests[0]['messages'][0]['content'], 'system')
        self.assertEqual(requests[0]['max_tokens'], 16)
        self.assertIs(type(text), str)
        self.assertEqual(text, '2')
        triplet = self.encode(deepcopy(planner.logs))[0]
        self.assertEqual(triplet.prompt['token_ids'], payload()['prompt_token_ids'])
        self.assertEqual(triplet.response['token_ids'], payload()['choices'][0]['token_ids'])
        self.assertEqual(triplet.metadata['finish_reason'], 'stop')

    def test_sdk_dump_fallback(self):
        data = payload()
        response = NS(choices=[NS(message=NS(content='2'), finish_reason='stop')], model_dump=lambda: data)
        completions = NS(create=lambda **kw: response)
        engine = NS(client=NS(chat=NS(completions=completions)))
        engine.model_string, engine.system_prompt, engine.use_cache = 'test', 'system', False
        with contextlib.redirect_stdout(io.StringIO()):
            self.generate(engine, 'question')
        self.assertEqual(engine.last_generation_metadata['response_token_ids'], [17, 151645])
        self.assertEqual(engine.last_generation_metadata['prompt_token_ids'], data['prompt_token_ids'])

    def test_missing_native_fields_rejected_without_text_fallback(self):
        for missing in ('prompt', 'response'):
            data = payload()
            if missing == 'prompt':
                data['prompt_token_ids'] = None
            else:
                del data['choices'][0]['token_ids']
                data['response_token_ids'] = [[999]]  # Wrong protocol must not be used.
            _, _, planner, _ = self.run_chain(data)
            with self.assertRaisesRegex(RuntimeError, 'missing exact vLLM token ids'):
                self.encode(planner.logs)

    def test_length_finish_does_not_fabricate_eos(self):
        data = payload()
        data['choices'][0].update(finish_reason='length', token_ids=[17])
        _, _, planner, _ = self.run_chain(data)
        triplet = self.encode(planner.logs)[0]
        self.assertEqual(triplet.response['token_ids'], [17])
        self.assertEqual(triplet.metadata['finish_reason'], 'length')

    def test_previous_log_is_not_overwritten(self):
        _, engine, planner, _ = self.run_chain(payload())
        engine.last_generation_metadata = None
        self.assertEqual(planner.logs[0]['response_token_ids'], [17, 151645])

    def test_probe_requests_native_fields_and_checks_all_choices(self):
        path = ROOT / 'agentflow/scripts/check_serving_token_ids.py'
        spec = importlib.util.spec_from_file_location('native_probe', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from openai.types.chat import ChatCompletion
        for valid in (True, False):
            data = payload()
            data['choices'].append({'index': 1, 'message': {'role': 'assistant', 'content': 'two'},
                                    'finish_reason': 'stop', 'token_ids': [2, 151645] if valid else None})
            completions = NS(create=lambda **kw: self.probe_response(kw, data, ChatCompletion))
            client = NS(chat=NS(completions=completions))
            with patch.object(module, 'OpenAI', return_value=client), patch('sys.argv', ['probe', '--base-url', 'http://serving.test/v1', '--model', 'test']), contextlib.redirect_stdout(io.StringIO()):
                if valid:
                    module.main()
                else:
                    with self.assertRaises(SystemExit):
                        module.main()

    def probe_response(self, kw, data, model):
        self.assertEqual(kw['extra_body'], {'return_token_ids': True})
        return model.model_validate(data)


class IdeaRewardCompatibilityTests(unittest.TestCase):
    def test_train_and_val_keep_exact_ids_and_infoseek_rewards(self):
        scope = dict(Any=Any, Dict=Dict, List=list, Optional=Optional,
                     BaseModel=BaseModel, Field=Field, ReadableSpan=Any,
                     AgentFlowRollout=Any, asyncio=asyncio, re=re, json=json,
                     string=string, unicodedata=unicodedata, datetime=datetime,
                     os=os, FileLock=FileLock)
        scope['Triplet'] = extract('agentflow/agent_types.py', 'Triplet', scope)
        scope['Rollout'] = extract('agentflow/agent_types.py', 'Rollout', scope)
        for name in ('_as_dict', '_normalize_entity', '_entity_mentioned',
                     '_training_reward', 'compute_search_subreward',
                     'build_gigpo_anchors', 'build_turn_process_rewards'):
            extract('train-roma/rollout.py', name, scope)

        async def judge(question, groundtruth, answer, val=False):
            self.assertEqual((question, groundtruth, answer), ('question', '2', '2'))
            return 1.0

        scope['evaluate'] = judge
        solve_and_evaluate = extract('train-roma/rollout.py', '_solve_and_evaluate', scope)
        logs = [
            {'prompt': f'prompt {i}',
             'response': '<answer>2</answer>' if i == 2 else '{}',
             'prompt_token_ids': [151644, 100 + i, 151645, 151644, 77091, 198],
             'response_token_ids': [200 + i, 151645],
             'finish_reason': 'stop'}
            for i in range(3)
        ]
        result = {
            'final_output': '<answer>2</answer>', 'planner_logs': logs,
            'memory': {'Action Step 1': {
                'tool_name': 'Wikipedia_Search_Tool',
                'result': 'John Smith supplies the relevant evidence.',
            }},
        }
        task = {
            'question': 'question', 'result': '2', 'extra_info': {'idx': 0},
            'reward_spec': {'subgoals': [{'id': 'person', 'answer': 'John Smith'}]},
        }

        def forbid_reencoding(*args, **kwargs):
            self.fail('The rollout must not reconstruct tokens from response text')

        for val in (False, True):
            with self.subTest(val=val), tempfile.TemporaryDirectory() as directory:
                agent = NS(task='qa', tools=['Wikipedia_Search_Tool'],
                           val_rollout_dir=directory,
                           tokenizer=NS(encode=forbid_reencoding))
                rollout = NS(llm_engine='test', solve=lambda **kw: (deepcopy(result), None))
                with contextlib.redirect_stdout(io.StringIO()):
                    package = asyncio.run(solve_and_evaluate(
                        agent, 'rollout-test', rollout, task, 1, val=val))

                self.assertEqual(len(package.triplets), len(logs))
                for triplet, log in zip(package.triplets, logs):
                    self.assertEqual(triplet.prompt['token_ids'], log['prompt_token_ids'])
                    self.assertEqual(triplet.response['token_ids'], log['response_token_ids'])
                    self.assertEqual(triplet.metadata['finish_reason'], 'stop')
                breakdown = package.metadata['reward_breakdown']
                self.assertEqual(breakdown['final_reward'], 1.0)
                self.assertEqual(breakdown['subreward'], 1.0)
                self.assertEqual(breakdown['turn_process_rewards'], {'1': 1.0})
                self.assertEqual(package.metadata['gigpo_pair_mask'], [True, True, False])
                self.assertEqual(json.loads(package.metadata['anchor'][1]),
                                 {'hit_subgoals': [], 'anchor_visit': 1})
                if val:
                    saved = json.loads((Path(directory) / 'rollouts.jsonl').read_text())
                    self.assertEqual(saved['final_reward'], 1.0)
                    self.assertEqual(saved['subreward'], 1.0)
                    self.assertEqual(saved['total_result']['planner_logs'], logs)


if __name__ == '__main__':
    unittest.main()
