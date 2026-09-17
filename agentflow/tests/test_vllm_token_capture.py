"""CPU contract tests: python -m unittest discover -s agentflow/tests -v."""
import asyncio
import ast
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, model_serializer


class Choice(BaseModel):
    index: int
    finish_reason: str = 'stop'
    message: dict = {'role': 'assistant', 'content': 'answer'}


class Completion(BaseModel):
    # Reproduce a pre-imported response class which drops undeclared extra fields.
    model_config = ConfigDict(extra='ignore')
    id: str = 'test'
    object: str = 'chat.completion'
    created: int = 0
    model: str = 'test'
    choices: list[Choice]
    ascend_extension: str = 'preserved'


class ErrorResponse(BaseModel):
    message: str
    code: int = 400


class Serving:
    async def chat_completion_full_generator(self, request, result_generator, *, extra=None):
        final = None
        async for result in result_generator:
            final = result
        return Completion(choices=[Choice(index=output.index) for output in final.outputs])


def load_instrumentation():
    modules = {name: types.ModuleType(name) for name in (
        'vllm', 'vllm.entrypoints', 'vllm.entrypoints.openai',
        'vllm.entrypoints.openai.protocol', 'vllm.entrypoints.openai.serving_chat')}
    modules['vllm.entrypoints.openai.protocol'].ChatCompletionResponse = Completion
    modules['vllm.entrypoints.openai.protocol'].ErrorResponse = ErrorResponse
    modules['vllm.entrypoints.openai.serving_chat'].OpenAIServingChat = Serving
    path = Path(__file__).resolve().parents[1] / 'instrumentation/vllm.py'
    spec = importlib.util.spec_from_file_location('token_capture_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


capture = load_instrumentation()


def result(prompt=(10, 11), outputs=((0, (20, 151645)),)):
    return NS(prompt_token_ids=list(prompt) if prompt is not None else None,
              outputs=[NS(index=index, token_ids=list(ids) if ids is not None else None)
                       for index, ids in outputs])


async def sequence(*results):
    for item in results:
        await asyncio.sleep(0)
        yield item


class TokenCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_bound_instance_json_and_openai_client(self):
        from openai.types.chat import ChatCompletion
        serving = Serving()
        capture.instrument_vllm(serving)
        response = await serving.chat_completion_full_generator(None, sequence(result()), extra='new parameter')
        payload = capture.serialize_chat_completion(response)
        parsed = ChatCompletion.model_validate_json(json.dumps(payload))
        self.assertEqual(parsed.prompt_token_ids, [10, 11])
        self.assertEqual(parsed.response_token_ids, [[20, 151645]])
        self.assertEqual(payload['ascend_extension'], 'preserved')
        self.assertIn('response_token_ids', response.model_dump())
        self.assertNotIn('response_token_ids', Completion.model_fields)

    async def test_subclass_override_and_already_bound_reference(self):
        class AscendServing(Serving):
            async def chat_completion_full_generator(self, request, result_generator, *, extra=None):
                return await super().chat_completion_full_generator(request, result_generator, extra=extra)
        serving = AscendServing()
        serving.chat_completion_full_generator = serving.chat_completion_full_generator
        original = serving.chat_completion_full_generator
        capture.instrument_vllm(serving)
        wrapper = serving.chat_completion_full_generator
        capture.instrument_vllm(serving)
        self.assertIs(serving.chat_completion_full_generator, wrapper)
        response = await wrapper(request=None, result_generator=sequence(result()))
        self.assertEqual(response.response_token_ids, [[20, 151645]])
        capture.uninstrument_vllm(serving)
        self.assertIs(serving.chat_completion_full_generator, original)

    async def test_early_consumer_exit(self):
        class Early(Serving):
            async def chat_completion_full_generator(self, request, result_generator):
                async for item in result_generator:
                    return Completion(choices=[Choice(index=0)])
        serving = Early()
        capture.instrument_vllm(serving)
        response = await serving.chat_completion_full_generator(None, sequence(result()))
        self.assertEqual(response.response_token_ids, [[20, 151645]])

    async def test_cumulative_and_choice_order(self):
        serving = Serving()
        capture.instrument_vllm(serving)
        response = await serving.chat_completion_full_generator(None, sequence(
            result(outputs=((1, (30,)), (0, (20,)))),
            result(outputs=((1, (30, 31, 151645)), (0, (20, 21, 151645)))),
        ))
        self.assertEqual(response.response_token_ids, [[30, 31, 151645], [20, 21, 151645]])

    async def test_non_cumulative_output_fails(self):
        serving = Serving()
        capture.instrument_vllm(serving)
        with self.assertRaisesRegex(capture.TokenCaptureError, 'non-cumulative'):
            await serving.chat_completion_full_generator(None, sequence(
                result(outputs=((0, (20,)),)), result(outputs=((0, (21,)),))))

    async def test_openai_http_client_transport(self):
        import httpx
        from openai import OpenAI
        serving = Serving()
        capture.instrument_vllm(serving)
        response = await serving.chat_completion_full_generator(None, sequence(result()))
        payload = capture.serialize_chat_completion(response)
        with OpenAI(api_key='test', base_url='http://serving.test/v1',
                    http_client=httpx.Client(transport=httpx.MockTransport(
                        lambda request: httpx.Response(200, json=payload)))) as client:
            parsed = client.chat.completions.create(model='test', messages=[{'role': 'user', 'content': 'test'}])
        self.assertEqual(parsed.model_dump()['response_token_ids'], [[20, 151645]])

    async def test_concurrent_requests_do_not_mix(self):
        serving = Serving()
        capture.instrument_vllm(serving)
        async def run(token):
            return await serving.chat_completion_full_generator(None, sequence(result(prompt=(token,), outputs=((0, (token, 151645)),))))
        first, second = await asyncio.gather(run(101), run(202))
        self.assertEqual(first.prompt_token_ids, [101])
        self.assertEqual(second.response_token_ids, [[202, 151645]])

    async def test_missing_ids_fail_without_retokenizing(self):
        serving = Serving()
        capture.instrument_vllm(serving)
        for item in (result(prompt=None), result(outputs=((0, None),)), result(outputs=((0, ()),))):
            with self.assertRaisesRegex(capture.TokenCaptureError, 'TOKEN B'):
                await serving.chat_completion_full_generator(None, sequence(item))

    async def test_error_response_passes_through(self):
        class Failed(Serving):
            async def chat_completion_full_generator(self, request, result_generator):
                return ErrorResponse(message='bad request')
        serving = Failed()
        capture.instrument_vllm(serving)
        response = await serving.chat_completion_full_generator(None, sequence())
        self.assertIsInstance(response, ErrorResponse)

    async def test_unsupported_signature_is_explicit(self):
        class Changed:
            async def chat_completion_full_generator(self, different_iterator):
                pass
        with self.assertRaisesRegex(capture.TokenCaptureError, 'TOKEN A'):
            capture.instrument_vllm(Changed())

    async def test_custom_serializer_cannot_drop_http_ids(self):
        class Custom(Completion):
            @model_serializer(mode='wrap')
            def omit_ids(self, handler):
                payload = handler(self)
                payload.pop('prompt_token_ids', None)
                payload.pop('response_token_ids', None)
                return payload
        class CustomServing(Serving):
            async def chat_completion_full_generator(self, request, result_generator, *, extra=None):
                async for item in result_generator:
                    pass
                return Custom(choices=[Choice(index=0)])
        serving = CustomServing()
        capture.instrument_vllm(serving)
        response = await serving.chat_completion_full_generator(None, sequence(result()))
        self.assertEqual(capture.serialize_chat_completion(response)['response_token_ids'], [[20, 151645]])

    async def test_global_install_and_restore(self):
        original = Serving.chat_completion_full_generator
        try:
            capture.instrument_vllm()
            capture.instrument_vllm()
            response = await Serving().chat_completion_full_generator(None, sequence(result()))
            self.assertEqual(response.response_token_ids, [[20, 151645]])
        finally:
            capture.uninstrument_vllm()
        self.assertIs(Serving.chat_completion_full_generator, original)

    async def test_http_rejects_bypassed_wrapper(self):
        with self.assertRaisesRegex(capture.TokenCaptureError, 'TOKEN A/C'):
            capture.serialize_chat_completion(Completion(choices=[Choice(index=0)]))

    async def test_real_endpoint_body_with_stubbed_inference(self):
        path = Path(__file__).resolve().parents[1] / 'verl/async_server.py'
        tree = ast.parse(path.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == 'chat_completion')
        class Request:
            async def json(self):
                return {'stream': False}
        class JSONResponse:
            def __init__(self, content, status_code=200):
                self.body = json.dumps(content)
                self.status_code = status_code
        scope = dict(asyncio=asyncio, Request=Request, ChatCompletionRequest=lambda **kw: NS(**kw),
                     ErrorResponse=ErrorResponse, JSONResponse=JSONResponse,
                     instrument_vllm=capture.instrument_vllm, TokenCaptureError=capture.TokenCaptureError,
                     serialize_chat_completion=capture.serialize_chat_completion)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
        class EndpointServing(Serving):
            async def create_chat_completion(self, request, raw_request):
                return await self.chat_completion_full_generator(request, sequence(result()))
        response = await scope['chat_completion'](NS(openai_serving_chat=EndpointServing()), Request())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)['response_token_ids'], [[20, 151645]])


if __name__ == '__main__':
    unittest.main()
