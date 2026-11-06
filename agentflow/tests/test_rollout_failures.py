"""Run the production worker with real payload models and fake async transports."""

import ast
import asyncio
import json
import logging
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    # Avoid distributed/model/OpenTelemetry dependencies while executing the
    # original class and the actual Pydantic task/result serialization schema.
    scope = dict(vars(typing), __name__=__name__, asyncio=asyncio, json=json,
                 logging=logging, math=math, os=os, time=time, BaseModel=BaseModel,
                 Field=Field, TaskInput=typing.Any, RolloutRawResult=typing.Any,
                 AgentFlowClient=typing.Any, LitAgent=typing.Any, TripletExporter=typing.Any,
                 ReadableSpan=type("ReadableSpan", (), {}), logger=logging.getLogger(__name__))
    path = ROOT / "agentflow/agent_types.py"
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name in {"Task", "Triplet", "Rollout", "ParallelWorkerBase"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    path = ROOT / "agentflow/runner.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AgentRunner")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["AgentRunner"], scope["Task"], scope["Rollout"], scope["Triplet"]


AgentRunner, Task, Rollout, Triplet = load_runner()


class RolloutFailureReportingTests(unittest.IsolatedAsyncioTestCase):
    def runner(self, *, mode="train", resources_id="policy-1"):
        task = Task(rollout_id="attempt-1", input={"question": "q"},
                    mode=mode, resources_id=resources_id)
        update = SimpleNamespace(resources_id="policy-1", resources={"policy": "model"})
        result = Rollout(rollout_id="unused", final_reward=0.0,
                         triplets=[Triplet(prompt="p", response="a", reward=0.0)],
                         logs=["original log"], metadata={"task": "qa"})
        agent = SimpleNamespace(set_runner=Mock(), training_rollout_async=AsyncMock(return_value=result),
                                validation_rollout_async=AsyncMock(return_value=result))
        client = SimpleNamespace(poll_next_task_async=AsyncMock(return_value=task),
                                 get_resources_by_id_async=AsyncMock(return_value=update),
                                 get_latest_resources_async=AsyncMock(return_value=update),
                                 post_rollout_async=AsyncMock(return_value={"status": "ok"}))
        with patch.dict(os.environ, {"AGENTFLOW_ROLLOUT_RPC_TIMEOUT_S": "1"}):
            return AgentRunner(agent, client, Mock(), worker_id=7)

    def reported(self, runner):
        runner.client.post_rollout_async.assert_awaited_once()
        return runner.client.post_rollout_async.await_args.args[0]

    async def test_success_preserves_reward_triplets_and_original_metadata(self):
        for mode in ("train", "val"):
            with self.subTest(mode=mode):
                runner = self.runner(mode=mode)
                self.assertTrue(await runner.run_async())
                payload = self.reported(runner)
                self.assertEqual(payload.rollout_id, "attempt-1")
                self.assertEqual(payload.final_reward, 0.0)
                self.assertEqual(payload.triplets[0].response, "a")
                self.assertEqual(payload.logs, ["original log"])
                self.assertEqual(payload.metadata, {"task": "qa", "resources_id": "policy-1"})
                chosen = (runner.agent.training_rollout_async if mode == "train"
                          else runner.agent.validation_rollout_async)
                chosen.assert_awaited_once_with({"question": "q"}, "attempt-1", {"policy": "model"})

    async def test_execution_failure_is_reported_without_a_reward_or_training_data(self):
        runner = self.runner()
        runner.agent.training_rollout_async.side_effect = RuntimeError("Judge API unavailable: api_key=private")
        with self.assertLogs(__name__, level="ERROR"):
            self.assertTrue(await runner.run_async())
        payload = self.reported(runner)
        self.assertEqual(payload.rollout_id, "attempt-1")
        self.assertIsNone(payload.final_reward)
        self.assertIsNone(payload.triplets)
        self.assertEqual(payload.metadata["resources_id"], "policy-1")
        self.assertEqual(payload.metadata["rollout_error"], {
            "type": "RuntimeError", "message": "Rollout execution or evaluation failed", "phase": "rollout"})
        self.assertNotIn("private", json.dumps(payload.model_dump(mode="json")))
        self.assertIsNone(payload.model_dump(mode="json")["final_reward"])

    async def test_missing_or_failed_resource_fetch_reports_the_claimed_task(self):
        for response, error in ((None, None), (None, ConnectionError("resource server down"))):
            with self.subTest(error=error):
                runner = self.runner()
                runner.client.get_resources_by_id_async.return_value = response
                runner.client.get_resources_by_id_async.side_effect = error
                with self.assertLogs(__name__, level="ERROR"):
                    self.assertTrue(await runner.run_async())
                payload = self.reported(runner)
                self.assertEqual(payload.metadata["rollout_error"]["phase"], "resources")
                self.assertEqual(payload.metadata["resources_id"], "policy-1")
                self.assertIsNone(payload.final_reward)
                runner.agent.training_rollout_async.assert_not_awaited()

    async def test_latest_resource_id_is_reported_for_tasks_without_a_version(self):
        runner = self.runner(resources_id=None)
        self.assertTrue(await runner.run_async())
        self.assertEqual(self.reported(runner).metadata["resources_id"], "policy-1")
        runner.client.get_resources_by_id_async.assert_not_awaited()
        runner.client.get_latest_resources_async.assert_awaited_once()

    async def test_conversion_failure_also_reports_terminal_attempt(self):
        runner = self.runner()
        runner._to_rollout_object = Mock(side_effect=ValueError("bad rollout result"))
        with self.assertLogs(__name__, level="ERROR"):
            self.assertTrue(await runner.run_async())
        self.assertEqual(self.reported(runner).metadata["rollout_error"]["type"], "ValueError")

    async def test_missing_or_rejected_acknowledgement_is_not_reported_as_success(self):
        for response in (None, {}, {"status": "rejected"}, {"status": "error"}, "ok"):
            with self.subTest(response=response):
                runner = self.runner()
                runner.client.post_rollout_async.return_value = response
                with self.assertLogs(__name__, level="ERROR"):
                    self.assertFalse(await runner.run_async())
                payload = self.reported(runner)
                self.assertNotIn("rollout_error", payload.metadata)
                self.assertEqual(payload.final_reward, 0.0)

    async def test_transport_exception_does_not_overwrite_a_success_with_failure(self):
        runner = self.runner()
        runner.client.post_rollout_async.side_effect = ConnectionError("ack lost")
        with self.assertLogs(__name__, level="ERROR"):
            self.assertFalse(await runner.run_async())
        self.assertNotIn("rollout_error", self.reported(runner).metadata)

    async def test_resource_rpc_timeout_is_reported_and_does_not_run_the_agent(self):
        runner = self.runner()
        runner.rollout_rpc_timeout = 0.01
        cancelled = asyncio.Event()

        async def stalled(*_args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        runner.client.get_resources_by_id_async.side_effect = stalled
        with self.assertLogs(__name__, level="ERROR"):
            self.assertTrue(await asyncio.wait_for(runner.run_async(), timeout=1))
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.reported(runner).metadata["rollout_error"]["phase"], "resources")
        runner.agent.training_rollout_async.assert_not_awaited()

    async def test_report_rpc_timeout_returns_without_a_second_payload(self):
        runner = self.runner()
        runner.rollout_rpc_timeout = 0.01
        cancelled = asyncio.Event()

        async def stalled(_payload):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        runner.client.post_rollout_async.side_effect = stalled
        with self.assertLogs(__name__, level="ERROR"):
            self.assertFalse(await asyncio.wait_for(runner.run_async(), timeout=1))
        self.assertTrue(cancelled.is_set())
        self.assertNotIn("rollout_error", self.reported(runner).metadata)

    async def test_cancellation_propagates_without_fabricating_an_outcome(self):
        runner = self.runner()
        runner.agent.training_rollout_async.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await runner.run_async()
        runner.client.post_rollout_async.assert_not_awaited()

    async def test_legacy_development_client_ack_is_supported(self):
        runner = self.runner()
        runner.client.post_rollout_async.return_value = {"status": "received"}
        self.assertTrue(await runner.run_async())

    async def test_no_task_does_not_report_a_fake_completion(self):
        runner = self.runner()
        runner.client.poll_next_task_async.return_value = None
        self.assertFalse(await runner.run_async())
        runner.client.post_rollout_async.assert_not_awaited()

    def test_rpc_timeout_must_be_finite_positive(self):
        for value in ("0", "-1", "nan", "inf", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {"AGENTFLOW_ROLLOUT_RPC_TIMEOUT_S": value}):
                with self.assertRaises(ValueError):
                    AgentRunner(Mock(), Mock(), Mock())


if __name__ == "__main__":
    unittest.main()
