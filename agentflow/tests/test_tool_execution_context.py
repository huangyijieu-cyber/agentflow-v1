"""Wall deadlines, cancellation and actual SDK errors; no external services."""
import ast
import contextlib
import io
import os
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from typing import List, Union
from unittest.mock import Mock

import httpx
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.tools.execution_context import (
    ToolExecutionContext, ToolExecutionError, bind_context, check_api_result,
    current_context, ensure_tool_context, owned_api_client, run_bounded,
    safe_failure, tool_sdk_client,
)


def api_connection_error():
    return openai.APIConnectionError(message="secret-key/prompt must not be exposed",
                                     request=httpx.Request("POST", "http://local.test"))


class ExecutionContextTests(unittest.TestCase):
    def test_real_sdk_connection_and_http_errors_preserve_phase_without_payload(self):
        errors = [api_connection_error(), openai.APITimeoutError(request=httpx.Request("POST", "http://local.test")),
                  openai.InternalServerError("secret-body", response=httpx.Response(
                      503, request=httpx.Request("POST", "http://local.test")), body={"secret": "prompt"})]
        for error, code in zip(errors, ("api_connection_error", "api_timeout", "api_http_error")):
            with self.subTest(code=code), self.assertRaises(ToolExecutionError) as caught:
                run_bounded(Mock(side_effect=error), phase="web.embedding")
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.phase, "web.embedding")
            self.assertNotIn("secret", str(caught.exception))
        self.assertEqual(caught.exception.status_code, 503)
        self.assertIsNone(current_context())

    def test_timeout_closes_only_owned_transport_and_stops_followup_work(self):
        release, returned = threading.Event(), threading.Event()
        clients, followup = [], Mock()
        def factory(timeout):
            client = NS(timeout=timeout, close=Mock(side_effect=release.set))
            clients.append(client)
            return client
        def slow():
            with owned_api_client(factory):
                release.wait(2)
                returned.set()
            followup()
        with ensure_tool_context(timeout=0.04) as context:
            started = time.monotonic()
            with self.assertRaises(ToolExecutionError) as caught:
                run_bounded(slow, phase="web.summary")
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(caught.exception.code, "tool_operation_timeout")
            self.assertTrue(context.cancel_event.is_set())
            self.assertTrue(returned.wait(1))
            self.assertGreater(clients[0].timeout, 0)
            self.assertLessEqual(clients[0].timeout, 0.04)
            clients[0].close.assert_called()
            with self.assertRaises(ToolExecutionError):
                run_bounded(followup, phase="next.page")
        followup.assert_not_called()

    def test_sdk_failure_during_slow_cleanup_is_kept_instead_of_generic_timeout(self):
        release = threading.Event()
        client = NS(close=Mock(side_effect=lambda: release.wait(1)))
        def fail():
            with owned_api_client(lambda timeout: client):
                raise api_connection_error()
        try:
            with ensure_tool_context(timeout=0.04), self.assertRaises(ToolExecutionError) as caught:
                run_bounded(fail, phase="web.embedding")
            self.assertEqual(caught.exception.code, "api_connection_error")
        finally:
            release.set()

    def test_context_is_isolated_between_simultaneous_rollouts(self):
        barrier = threading.Barrier(6)
        def run(index):
            with ensure_tool_context(timeout=1) as context:
                barrier.wait()
                value = run_bounded(lambda: current_context(), phase=f"rollout.{index}")
                self.assertIs(value, context)
                self.assertEqual(context.phase, f"rollout.{index}")
                if index == 0:
                    context.cancel()
                else:
                    context.check()
                return context
        with ThreadPoolExecutor(max_workers=6) as pool:
            contexts = list(pool.map(run, range(6)))
        self.assertEqual(len({id(context) for context in contexts}), 6)
        self.assertIsNone(current_context())

    def test_returned_api_error_envelope_is_not_treated_as_empty_model_output(self):
        with self.assertRaises(ToolExecutionError) as caught:
            check_api_result({"error": "RateLimitError", "message": "secret-prompt"}, "wiki.select_pages")
        self.assertEqual(caught.exception.code, "api_error_response")
        self.assertNotIn("secret", str(caught.exception))

    def test_reusing_parent_context_does_not_reset_total_deadline(self):
        with ensure_tool_context(timeout=0.15) as parent:
            deadline = parent.deadline
            with ensure_tool_context(timeout=300) as nested:
                self.assertIs(nested, parent)
                self.assertEqual(nested.deadline, deadline)

    def test_earlier_failure_cannot_become_the_next_operation_timeout_cause(self):
        release = threading.Event()
        with ensure_tool_context(timeout=1):
            with self.assertRaises(ToolExecutionError):
                run_bounded(Mock(side_effect=api_connection_error()), phase="web.embedding")
            try:
                with self.assertRaises(ToolExecutionError) as caught:
                    run_bounded(lambda: release.wait(1), phase="web.embedding", timeout=0.03,
                                abort=release.set)
                self.assertEqual(caught.exception.code, "tool_operation_timeout")
            finally:
                release.set()


class ToolVLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "agentflow/engine/vllm.py"
        node = next(item for item in ast.parse(path.read_text()).body
                    if isinstance(item, ast.ClassDef) and item.name == "ChatVLLM")
        scope = dict(EngineLM=object, CachedEngine=type("Cache", (), {}), os=os, List=List, Union=Union,
                     retry=retry, stop_after_attempt=stop_after_attempt,
                     wait_random_exponential=wait_random_exponential, retry_if_exception_type=retry_if_exception_type,
                     APIConnectionError=openai.APIConnectionError, APITimeoutError=openai.APITimeoutError,
                     ToolExecutionError=ToolExecutionError, current_context=current_context,
                     safe_failure=safe_failure, tool_sdk_client=tool_sdk_client)
        # Use two independent base classes rather than object + its subclass.
        scope["EngineLM"] = type("Engine", (), {})
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        cls.Engine, cls.scope = scope["ChatVLLM"], scope

    def test_tool_sdk_failure_is_raised_once_and_does_not_close_shared_client(self):
        shared = NS(close=Mock(), chat=NS(completions=NS(create=Mock())))
        owned = NS(close=Mock(), chat=NS(completions=NS(create=Mock(side_effect=api_connection_error()))))
        constructor = Mock(return_value=owned)
        self.scope["OpenAI"] = constructor
        engine = self.Engine.__new__(self.Engine)
        engine.client, engine.base_url, engine.api_key = shared, "http://local.test", "test-key"
        engine.model_string, engine.system_prompt, engine.temperature, engine.use_cache = "model", "system", 0, False
        with ensure_tool_context(timeout=1), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ToolExecutionError) as caught:
            run_bounded(lambda: engine.generate("prompt"), phase="wiki.select_pages")
        self.assertEqual(caught.exception.code, "api_connection_error")
        owned.chat.completions.create.assert_called_once()
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)
        self.assertLessEqual(constructor.call_args.kwargs["timeout"], 1)
        owned.close.assert_called_once()
        shared.close.assert_not_called()
        shared.chat.completions.create.assert_not_called()
        self.assertIsNone(engine.last_generation_metadata)


if __name__ == "__main__":
    unittest.main()
