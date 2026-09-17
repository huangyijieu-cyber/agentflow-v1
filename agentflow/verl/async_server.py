# [修改目的] 原代码下面已经显式调用 asyncio.wait_for / asyncio.TimeoutError，
# 但没有导入 asyncio，会在执行 chat_completion 时触发 NameError。
import asyncio
import ray
from copy import deepcopy

# [原代码保留] from agentflow.instrumentation.vllm import instrument_vllm, ChatCompletionResponsePatched
# [修改目的] 在实际 serving 对象安装捕获器，在 HTTP 边界显式序列化 token 字段。
from agentflow.instrumentation.vllm import (
    instrument_vllm, serialize_chat_completion, TokenCaptureError,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


@ray.remote(num_cpus=1)
class PatchedvLLMServer(_unwrap_ray_remote(AsyncvLLMServer)):

    def __init__(self, *args, **kwargs):
        # [原代码保留] instrument_vllm()
        # [修改目的] VERL 可延迟初始化或使用子类；请求时绑定实际 serving 实例。
        super().__init__(*args, **kwargs)

        self.config = deepcopy(self.config)
        self.config.rollout.multi_turn.tool_config_path = "/dev/null"

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)

        try:
            # [修改目的] 此时实际 serving 已初始化；流式接口保持原行为。
            if not request.stream:
                instrument_vllm(self.openai_serving_chat)
            # 仅对生成部分设置超时，例如 100 秒
            generator = await asyncio.wait_for(
                self.openai_serving_chat.create_chat_completion(request, raw_request),
                timeout=1200 * 10
            )
        except TokenCaptureError as exc:
            # [修改目的] 返回可定位的 serving 错误，不让缺少 IDs 的成功响应进入 rollout。
            return JSONResponse(content={"error": {"message": str(exc), "type": "token_capture_error"}}, status_code=500)
        except asyncio.TimeoutError:
            return JSONResponse(
                content={"error": "Model inference timeout"},
                status_code=504
            )


        # generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            # [原代码保留] return JSONResponse(content=generator.model_dump())
            # [修改目的] 验证真实 token 字段并显式加入 HTTP JSON，禁止重新 tokenize。
            try:
                payload = serialize_chat_completion(generator)
            except TokenCaptureError as exc:
                return JSONResponse(content={"error": {"message": str(exc), "type": "token_capture_error"}}, status_code=500)
            return JSONResponse(content=payload)

