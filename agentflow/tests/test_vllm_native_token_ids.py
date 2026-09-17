"""CPU tests of actual QA functions; no vLLM/VERL workers or RL training."""
import ast
import contextlib
import importlib.util
import io
import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, Optional
from unittest.mock import patch

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


def extract(path, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
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


if __name__ == '__main__':
    unittest.main()
