"""Offline lifecycle regressions using actual classes/methods, without GPU imports."""
import ast
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]


def load_nodes(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)


NS = {"BaseModel": BaseModel, "Field": Field, "Any": object, "TaskInput": object,
      "Optional": __import__("typing").Optional, "List": list, "Dict": dict,
      "Literal": __import__("typing").Literal, "asyncio": asyncio, "time": time,
      "os": os, "uuid": uuid, "logger": logging.getLogger("lifecycle_test"),
      "ParallelWorkerBase": object, "ReadableSpan": type("ReadableSpan", (), {}),
      "json": json, "asynccontextmanager": asynccontextmanager}
load_nodes("agentflow/agent_types.py", {"Triplet", "Rollout", "Task"}, NS)
Triplet, Rollout, Task = (NS[name] for name in ("Triplet", "Rollout", "Task"))
for model in (Triplet, Rollout, Task):
    model.model_rebuild(_types_namespace=NS)
load_nodes("agentflow/server.py", {"ServerDataStore", "AgentFlowServer"}, NS)
load_nodes("agentflow/runner.py", {"AgentRunner"}, NS)
Store, Server, Runner = (NS[name] for name in ("ServerDataStore", "AgentFlowServer", "AgentRunner"))


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def task(self, store, timeout=1200, mode="val"):
        task_id = await store.add_task({"question": "q"}, mode=mode, resources_id="r1",
                                       metadata={"validation_timeout_s": timeout})
        return await store.get_next_task()

    def runner(self, store, task, method):
        async def post(result):
            await store.store_rollout(result)
            return {"status": "ok"}
        client = types.SimpleNamespace(
            poll_next_task_async=AsyncMock(return_value=task),
            get_resources_by_id_async=AsyncMock(return_value=types.SimpleNamespace(
                resources_id="r1", resources={"llm": "test"})),
            get_latest_resources_async=AsyncMock(),
            post_rollout_async=AsyncMock(side_effect=post))
        agent = types.SimpleNamespace(set_runner=lambda _: None,
                                      validation_rollout_async=method, training_rollout_async=method)
        return Runner(agent, client, None), client

    async def daemon(self, store, ids, completed=None, clock=None):
        tree = ast.parse((ROOT / "agentflow/verl/daemon.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "AgentModeDaemon")
        method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef)
                      and node.name == "_async_run_until_finished")
        ns = dict(NS, profiling_enabled=lambda: False)
        if clock is not None:
            async def sleep(seconds):
                clock[0] += seconds
                await asyncio.sleep(0)
            ns["time"] = types.SimpleNamespace(time=lambda: clock[0])
            ns["asyncio"] = types.SimpleNamespace(sleep=sleep)
        exec(compile(ast.Module(body=[method], type_ignores=[]), "daemon_loop", "exec"), ns)
        daemon = types.SimpleNamespace(
            _total_tasks_queued=len(ids), _completed_rollouts=completed or {}, _failed_rollouts={},
            _task_id_to_original_sample={task_id: {} for task_id in ids}, is_train=False,
            server=store, _validate_data=lambda _: None)
        await asyncio.wait_for(ns[method.name](daemon, verbose=False), timeout=1)
        return daemon

    async def test_exception_reports_failure_and_99_of_100_finishes(self):
        store = Store()
        task = await self.task(store)
        runner, client = self.runner(store, task, AsyncMock(side_effect=ValueError("context overflow")))
        self.assertTrue(await runner.run_async())
        report = client.post_rollout_async.call_args.args[0]
        self.assertEqual(report.metadata["error_type"], "ValueError")
        self.assertIsNone(report.final_reward)
        self.assertIsNone(report.triplets)
        completed = {f"done-{i}": Rollout(rollout_id=f"done-{i}") for i in range(99)}
        daemon = await self.daemon(store, [*completed, task.rollout_id], completed)
        self.assertEqual(len(daemon._completed_rollouts), 99)
        self.assertEqual(len(daemon._failed_rollouts), 1)
        self.assertFalse(store.get_processing_tasks())

    async def test_hung_coroutine_times_out_and_reports(self):
        store = Store()
        task = await self.task(store, timeout=0.01)
        cancelled = asyncio.Event()
        async def hang(*args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        runner, client = self.runner(store, task, hang)
        self.assertTrue(await asyncio.wait_for(runner.run_async(), timeout=1))
        self.assertTrue(cancelled.is_set())
        report = client.post_rollout_async.call_args.args[0]
        self.assertEqual(report.metadata["execution_status"], "failed")
        self.assertIn("deadline", report.metadata["error_message"])

    async def test_missing_resources_and_rejected_report_are_not_success(self):
        store = Store()
        task = await self.task(store)
        method = AsyncMock()
        runner, client = self.runner(store, task, method)
        client.get_resources_by_id_async.return_value = None
        self.assertTrue(await runner.run_async())
        self.assertEqual(client.post_rollout_async.call_args.args[0].metadata["error_type"], "RuntimeError")
        method.assert_not_awaited()
        client.post_rollout_async.side_effect = None
        client.post_rollout_async.return_value = None
        self.assertFalse(await runner.run_async())

    async def test_success_keeps_model_reward_and_triplets(self):
        store = Store()
        task = await self.task(store)
        result = Rollout(rollout_id=task.rollout_id, final_reward=0.7,
                         triplets=[Triplet(prompt={}, response={}, reward=0.7)])
        runner, client = self.runner(store, task, AsyncMock(return_value=result))
        self.assertTrue(await runner.run_async())
        daemon = await self.daemon(store, [task.rollout_id])
        self.assertEqual(daemon._completed_rollouts[task.rollout_id].final_reward, 0.7)
        self.assertFalse(daemon._failed_rollouts)

    async def test_server_deadline_handles_dead_worker_and_ignores_late_result(self):
        store = Store()
        task = await self.task(store)
        store._processing_tasks[task.rollout_id].last_claim_time = time.time() - 1300
        server = types.SimpleNamespace(_store=store, _task_timeout_seconds=12000)
        await Server._check_and_requeue_stale_tasks(server)
        failures = await store.retrieve_completed_rollouts()
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].metadata["error_type"], "TaskTimeout")
        await store.store_rollout(Rollout(rollout_id=task.rollout_id, final_reward=1.0))
        self.assertEqual(await store.retrieve_completed_rollouts(), [])
        self.assertIsNone(await store.get_next_task())

    async def test_deadline_does_not_count_time_waiting_in_queue(self):
        store = Store()
        task_id = await store.add_task({}, mode="val", metadata={"validation_timeout_s": 1200})
        task = await store.get_next_task()
        task.create_time = time.time() - 5000
        server = types.SimpleNamespace(_store=store, _task_timeout_seconds=12000)
        await Server._check_and_requeue_stale_tasks(server)
        self.assertIn(task_id, store.get_processing_tasks())
        self.assertEqual(await store.retrieve_completed_rollouts(), [])

    async def test_batch_deadline_settles_unclaimed_and_missing_tasks(self):
        store = Store()
        task_id = await store.add_task({}, mode="val")
        training_id = await store.add_task({}, mode="train")
        with patch.dict(os.environ, {"AGENTFLOW_VAL_BATCH_TIMEOUT_S": "1"}):
            daemon = await self.daemon(store, [task_id, "lost-id"], clock=[0.0])
        self.assertEqual(set(daemon._failed_rollouts), {task_id, "lost-id"})
        self.assertFalse(daemon._completed_rollouts)
        self.assertEqual((await store.get_next_task()).rollout_id, training_id)
        self.assertIsNone(await store.get_next_task())

class ConfigurationAndReportTests(unittest.TestCase):
    def test_model_clients_bound_requests_without_nested_sdk_retries(self):
        for engine_file in ("vllm.py", "openai.py"):
            tree = ast.parse((ROOT / "agentflow/agentflow/engine" / engine_file).read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name in {"ChatVLLM", "ChatOpenAI"})
            init = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                        and node.name == "__init__")
            calls = []
            ns = {"os": os, "DEFAULT_SYSTEM_PROMPT": "system",
                  "OpenAI": lambda **kwargs: calls.append(kwargs),
                  "validate_structured_output_model": lambda _: False,
                  "validate_chat_model": lambda _: False,
                  "validate_reasoning_model": lambda _: False,
                  "validate_pro_reasoning_model": lambda _: False,
                  "validate_local_model": lambda _: True,
                  "validate_local_url": lambda _: True}
            exec(compile(ast.Module(body=[init], type_ignores=[]), engine_file, "exec"), ns)
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "AGENTFLOW_LLM_REQUEST_TIMEOUT_S": "7"}):
                ns["__init__"](types.SimpleNamespace(), model_string="qwen-test", use_cache=False)
            self.assertEqual(calls[0]["timeout"], 7)
            self.assertEqual(calls[0]["max_retries"], 0)

    def test_client_honors_configured_timeout(self):
        ns = dict(NS)
        load_nodes("agentflow/client.py", {"AgentFlowClient"}, ns)
        client = ns["AgentFlowClient"]("http://unused", poll_interval=2, timeout=7)
        self.assertEqual(client.timeout, 7)

    def test_report_keeps_failures_out_of_completed_denominator(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("timing_report_test", ROOT / "agentflow/agentflow/rollout_timing.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        failure = Rollout(rollout_id="failed", metadata={"error_type": "TaskTimeout", "error_message": "deadline"})
        with tempfile.TemporaryDirectory() as directory:
            _, path, summary = module.write_validation_report(
                [Rollout(rollout_id="done")], directory, 2, "test", failed_rollouts=[failure])
            self.assertEqual(summary["completed_count"], 1)
            self.assertEqual(summary["failed_count"], 1)
            self.assertEqual(summary["settled_count"], 2)
            self.assertFalse(summary["complete_expected"])
            self.assertEqual(json.loads(path.read_text())["failures"][0]["rollout_id"], "failed")


if __name__ == "__main__":
    unittest.main()
