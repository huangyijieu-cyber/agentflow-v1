from __future__ import annotations

# [修改目的] 在真正处理请求的 serving 实例上安装捕获器；不依赖 protocol 类别名替换。
import inspect
import logging
import os
from functools import lru_cache, wraps
from importlib.metadata import PackageNotFoundError, version
from typing import List

from pydantic import create_model
from vllm.entrypoints.openai.protocol import ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

logger = logging.getLogger(__name__)


class TokenCaptureError(RuntimeError):
    """An exact-token contract failure, never repaired by text re-tokenization."""


class ChatCompletionResponsePatched(ChatCompletionResponse):
    prompt_token_ids: List[int] | None = None
    # [原代码保留] response_token_ids: List[int] | None = None
    # [修改目的] HTTP 协议保留每个 choice 的真实 token 序列。
    response_token_ids: List[List[int]] | None = None


@lru_cache(maxsize=16)
def _response_model(base):
    # [修改目的] 继承实际响应类型，保留当前 vLLM/Ascend 扩展字段，显式声明 IDs。
    return create_model(
        f"AgentFlowTokens{base.__name__}",
        __base__=base,
        prompt_token_ids=(List[int] | None, None),
        response_token_ids=(List[List[int]] | None, None),
    )


def _ids(value, field):
    if value is None:
        raise TokenCaptureError(f"[TOKEN B] missing {field} in generation output")
    ids = list(value)
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        raise TokenCaptureError(f"[TOKEN B] invalid/empty {field} in generation output")
    return ids


def _debug(message, *args):
    # [修改目的] 仅按需打印类型、字段名、长度，不打印题目、回答或完整 token 序列。
    if os.environ.get("AGENTFLOW_TOKEN_DEBUG") == "1":
        logger.warning("[TOKEN DEBUG] " + message, *args)


def _wrap_full_generator(original):
    signature = inspect.signature(original)
    if "result_generator" not in signature.parameters:
        raise TokenCaptureError(
            f"[TOKEN A] unsupported serving signature {signature}; "
            "inspect the installed vLLM/VERL implementation"
        )

    @wraps(original)
    async def wrapped(*args, **kwargs):
        # [修改目的] 绑定实际方法签名，兼容新增参数与 positional/keyword 调用。
        bound = signature.bind(*args, **kwargs)
        source = bound.arguments["result_generator"]
        prompt_ids = None
        output_ids = {}
        observed = False
        _debug("wrapper entered: %s.%s", original.__module__, original.__qualname__)

        async def capture():
            nonlocal prompt_ids, observed
            async for result in source:
                observed = True
                outputs = getattr(result, "outputs", None)
                _debug("result=%s fields=%s prompt_ids_present=%s outputs=%s",
                       type(result).__qualname__, sorted(getattr(result, "__dict__", {})),
                       getattr(result, "prompt_token_ids", None) is not None,
                       type(outputs).__qualname__)
                # [修改目的] 复制原始 IDs；在 yield 前捕获，兼容最后一项后停止消费。
                raw_prompt = getattr(result, "prompt_token_ids", None)
                if raw_prompt is not None:
                    prompt_ids = list(raw_prompt)
                if outputs is not None:
                    for output in outputs:
                        index = getattr(output, "index", None)
                        raw_ids = getattr(output, "token_ids", None)
                        _debug("output=%s fields=%s index=%s token_count=%s",
                               type(output).__qualname__, sorted(getattr(output, "__dict__", {})),
                               index, len(raw_ids) if raw_ids is not None else None)
                        if type(index) is not int:
                            raise TokenCaptureError("[TOKEN B] output.index missing; cannot align choices")
                        # Full (non-streaming) serving consumes cumulative/final RequestOutput.
                        current_ids = list(raw_ids) if raw_ids is not None else None
                        previous_ids = output_ids.get(index)
                        if previous_ids and current_ids is not None and current_ids[:len(previous_ids)] != previous_ids:
                            raise TokenCaptureError("[TOKEN B] non-cumulative token output; inspect installed output_kind")
                        output_ids[index] = current_ids
                yield result

        bound.arguments["result_generator"] = capture()
        response = await original(*bound.args, **bound.kwargs)
        if isinstance(response, ErrorResponse):
            return response
        if not observed:
            raise TokenCaptureError("[TOKEN B] wrapper ran but no RequestOutput was consumed")
        prompt_ids = _ids(prompt_ids, "prompt_token_ids")
        choices = getattr(response, "choices", None)
        if not choices:
            raise TokenCaptureError("[TOKEN C] unexpected successful response without choices")
        response_ids = [_ids(output_ids.get(choice.index),
                             f"response_token_ids[choice={choice.index}]") for choice in choices]
        payload = response.model_dump()
        _debug("response=%s fields=%s dump_keys=%s prompt_count=%d response_counts=%s",
               type(response).__qualname__, sorted(type(response).model_fields),
               sorted(payload), len(prompt_ids), [len(ids) for ids in response_ids])
        payload.update(prompt_token_ids=prompt_ids, response_token_ids=response_ids)
        # [修改目的] 不使用 model_copy 向旧模型塞未声明字段，避免 extra=ignore 丢字段。
        return _response_model(type(response)).model_validate(payload)

    wrapped._agentflow_token_capture = True
    wrapped._agentflow_original = original
    return wrapped


def instrument_vllm(serving=None):
    # [修改目的] 请求时传入真实实例，覆盖子类 override/提前绑定的方法；重复调用幂等。
    target = OpenAIServingChat if serving is None else serving
    original = getattr(target, "chat_completion_full_generator", None)
    if original is None:
        raise TokenCaptureError(f"[TOKEN A] {type(target).__qualname__} has no full generator")
    if getattr(original, "_agentflow_token_capture", False):
        return
    wrapped = _wrap_full_generator(original)
    setattr(target, "chat_completion_full_generator", wrapped)
    packages = {}
    for package in ("verl", "vllm", "vllm-ascend"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not registered"
    logger.warning(
        "[TOKEN PATCH] serving=%s.%s method=%s.%s signature=%s source=%s versions=%s",
        getattr(target, "__module__", type(target).__module__),
        getattr(target, "__qualname__", type(target).__qualname__),
        original.__module__, original.__qualname__, inspect.signature(original),
        inspect.getsourcefile(original), packages,
    )


def serialize_chat_completion(response):
    # [修改目的] HTTP 边界显式写入 IDs，避免响应模型的自定义 serializer 再次丢字段。
    payload = response.model_dump()
    prompt = getattr(response, "prompt_token_ids", None)
    responses = getattr(response, "response_token_ids", None)
    if prompt is None or responses is None:
        raise TokenCaptureError(
            "[TOKEN A/C] completion bypassed token wrapper or response fields were lost; "
            "enable AGENTFLOW_TOKEN_DEBUG=1 and check wrapper entry"
        )
    payload["prompt_token_ids"] = _ids(prompt, "prompt_token_ids")
    payload["response_token_ids"] = [_ids(ids, "response_token_ids") for ids in responses]
    if not responses or len(responses) != len(response.choices):
        raise TokenCaptureError("[TOKEN C] response token sequences do not match choices")
    _debug("HTTP keys=%s prompt_count=%d response_counts=%s", sorted(payload),
           len(prompt), [len(ids) for ids in responses])
    return payload


def uninstrument_vllm(serving=None):
    target = OpenAIServingChat if serving is None else serving
    current = getattr(target, "chat_completion_full_generator", None)
    original = getattr(current, "_agentflow_original", None)
    if original is not None:
        setattr(target, "chat_completion_full_generator", original)


# [原代码保留] 旧版全局 patch 完整保留供对照。
# [修改目的] 上面改为实际实例绑定和显式响应字段；不再替换全局 protocol 类。
# from __future__ import annotations
#
# import warnings
# from typing import List
#
# from vllm.entrypoints.openai.protocol import ChatCompletionResponse
# import vllm.entrypoints.openai.protocol
# from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
#
#
# class ChatCompletionResponsePatched(ChatCompletionResponse):
#     prompt_token_ids: List[int] | None = None
#     response_token_ids: List[int] | None = None
#
#
# original_chat_completion_full_generator = OpenAIServingChat.chat_completion_full_generator
#
#
# async def chat_completion_full_generator(
#     self,
#     request,
#     result_generator,
#     request_id: str,
#     model_name: str,
#     conversation,
#     tokenizer,
#     request_metadata,
# ):
#     prompt_token_ids: List[int] | None = None
#     response_token_ids: List[List[int]] | None = None
#
#     async def _generate_inceptor():
#         nonlocal prompt_token_ids, response_token_ids
#         async for res in result_generator:
#             yield res
#             prompt_token_ids = res.prompt_token_ids
#             response_token_ids = [output.token_ids for output in res.outputs]
#
#     response = await original_chat_completion_full_generator(
#         self,
#         request,
#         _generate_inceptor(),
#         request_id,
#         model_name,
#         conversation,
#         tokenizer,
#         request_metadata,
#     )
#     response = response.model_copy(
#         update={
#             "prompt_token_ids": prompt_token_ids,
#             "response_token_ids": response_token_ids,
#         }
#     )
#
#     return response
#
#
# def instrument_vllm():
#     if vllm.entrypoints.openai.protocol.ChatCompletionResponse is ChatCompletionResponsePatched:
#         warnings.warn("vllm is already instrumented. Skip the instrumentation.")
#         return
#
#     vllm.entrypoints.openai.protocol.ChatCompletionResponse = ChatCompletionResponsePatched
#     OpenAIServingChat.chat_completion_full_generator = chat_completion_full_generator
#
#
# def uninstrument_vllm():
#     OpenAIServingChat.chat_completion_full_generator = original_chat_completion_full_generator
#
