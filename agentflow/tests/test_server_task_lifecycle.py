"""CPU async regressions for rollout completion, stale retries and batch fencing."""
import ast
import asyncio
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_protocol_and_server():
    # Run the production models/server without importing model/tracer backends.
    protocol = ModuleType("lifecycle_test_protocol")
    protocol.ReadableSpan = object
    sys.modules[protocol.__name__] = protocol
    path = ROOT / "agent_types.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "opentelemetry.sdk.trace")]
    exec(compile(tree, str(path), "exec"), protocol.__dict__)

    server = ModuleType("lifecycle_test_server")
    sys.modules[server.__name__] = server
    for name in ("Rollout", "Task", "TaskIfAny", "NamedResources", "GenericResponse", "ResourcesUpdate"):
        setattr(server, name, getattr(protocol, name))
    path = ROOT / "server.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "agent_types")]
    exec(compile(tree, str(path), "exec"), server.__dict__)
    return protocol, server


PROTOCOL, SERVER = load_protocol_and_server()


def async_api(**changes):
    return SimpleNamespace(**{**vars(asyncio), **changes})


class ServerTaskLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = SERVER.ServerDataStore()

    async def claimed(self, store=None):
        store = store or self.store
        rollout_id = await store.add_task({"question": "sample"}, resources_id="res-1")
        task = await store.get_next_task()
        self.assertEqual(task.rollout_id, rollout_id)
        return task

    def result(self, task, reward=1.0):
        return PROTOCOL.Rollout(rollout_id=task.rollout_id, final_reward=reward,
                                metadata={"resources_id": task.resources_id})

    async def test_success_and_zero_reward_preserved_and_report_is_idempotent_after_retrieval(self):
        for reward in (1.0, 0.0):
            task = await self.claimed()
            result = self.result(task, reward)
            self.assertTrue(await self.store.store_rollout(result))
            self.assertTrue(await self.store.store_rollout(result))
            self.assertEqual(await self.store.retrieve_completed_rollouts(), [result])
            self.assertTrue(await self.store.store_rollout(self.result(task, 99.0)))
            self.assertEqual(await self.store.retrieve_completed_rollouts(), [])
            self.assertNotIn(task.rollout_id, self.store.get_processing_tasks())

    async def test_unknown_and_never_claimed_result_rejected(self):
        self.assertFalse(await self.store.store_rollout(PROTOCOL.Rollout(rollout_id="unknown")))
        rollout_id = await self.store.add_task({"question": "pending"})
        self.assertFalse(await self.store.store_rollout(PROTOCOL.Rollout(rollout_id=rollout_id)))
        self.assertEqual(await self.store.retrieve_completed_rollouts(), [])
        self.assertEqual((await self.store.get_next_task()).rollout_id, rollout_id)

    async def test_clear_discards_pending_processing_completed_and_rejects_late_result(self):
        done_task = await self.claimed()
        await self.store.store_rollout(self.result(done_task))
        running_task = await self.claimed()
        await self.store.add_task({"question": "pending"})
        update = PROTOCOL.ResourcesUpdate(resources_id="res-1", resources={
            "main_llm": PROTOCOL.LLM(endpoint="http://localhost/v1", model="test")})
        await self.store.update_resources(update)
        await self.store.clear_queues()
        self.assertIsNone(await self.store.get_next_task())
        self.assertEqual(self.store.get_processing_tasks(), {})
        self.assertEqual(await self.store.retrieve_completed_rollouts(), [])
        self.assertFalse(await self.store.store_rollout(self.result(running_task)))
        self.assertFalse(await self.store.store_rollout(self.result(done_task)))
        self.assertEqual(await self.store.get_latest_resources(), update)
        self.assertEqual(self.store._failed_tasks, {})
        await self.claimed()

    async def test_timeout_exhaustion_becomes_visible_failure_without_reward(self):
        store = SERVER.ServerDataStore(max_retries=1)
        task = await self.claimed(store)
        await store.requeue_task(task)
        results = await store.retrieve_completed_rollouts()
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.rollout_id, task.rollout_id)
        self.assertIsNone(result.final_reward)
        self.assertIsNone(result.triplets)
        self.assertEqual(result.metadata["resources_id"], "res-1")
        self.assertEqual(result.metadata["rollout_error"]["type"], "TaskTimeout")
        self.assertEqual(result.metadata["rollout_error"]["phase"], "task_timeout")
        self.assertIs(result.metadata["rollout_error"]["retry_exhausted"], True)
        await store.requeue_task(task)
        self.assertTrue(await store.store_rollout(self.result(task)))
        self.assertEqual(await store.retrieve_completed_rollouts(), [])
        self.assertIsNone(await store.get_next_task())

    async def test_clear_during_backoff_does_not_resurrect_old_batch(self):
        task = await self.claimed()
        entered, release = asyncio.Event(), asyncio.Event()

        async def pause(delay):
            entered.set()
            await release.wait()

        with mock.patch.object(SERVER, "asyncio", async_api(sleep=pause)):
            pending = asyncio.create_task(self.store.requeue_task(task))
            await entered.wait()
            await self.store.clear_queues()
            new_id = await self.store.add_task({"question": "next batch"})
            release.set()
            await pending
        self.assertEqual((await self.store.get_next_task()).rollout_id, new_id)
        self.assertIsNone(await self.store.get_next_task())
        self.assertFalse(await self.store.store_rollout(self.result(task)))

    async def test_late_result_consumed_during_backoff_prevents_requeue(self):
        task = await self.claimed()
        entered, release = asyncio.Event(), asyncio.Event()

        async def pause(delay):
            entered.set()
            await release.wait()

        with mock.patch.object(SERVER, "asyncio", async_api(sleep=pause)):
            pending = asyncio.create_task(self.store.requeue_task(task))
            await entered.wait()
            result = self.result(task)
            self.assertTrue(await self.store.store_rollout(result))
            self.assertEqual(await self.store.retrieve_completed_rollouts(), [result])
            release.set()
            await pending
        self.assertIsNone(await self.store.get_next_task())

    async def test_late_result_after_requeue_prevents_claiming_queued_copy(self):
        task = await self.claimed()
        with mock.patch.object(SERVER, "asyncio", async_api(sleep=mock.AsyncMock())):
            await self.store.requeue_task(task)
        self.assertTrue(await self.store.store_rollout(self.result(task)))
        await self.store.retrieve_completed_rollouts()
        self.assertIsNone(await self.store.get_next_task())

    async def test_old_stale_snapshot_does_not_remove_newer_claim(self):
        first = await self.claimed()
        with mock.patch.object(SERVER, "asyncio", async_api(sleep=mock.AsyncMock())):
            await self.store.requeue_task(first)
            second = await self.store.get_next_task()
            self.assertEqual(second.num_claims, 2)
            await self.store.requeue_task(first)
        self.assertIs(self.store.get_processing_tasks()[first.rollout_id], second)
        self.assertIsNone(await self.store.get_next_task())
        self.assertEqual(await self.store.retrieve_completed_rollouts(), [])

    async def test_concurrent_stale_checks_queue_only_one_retry(self):
        task = await self.claimed()
        with mock.patch.object(SERVER, "asyncio", async_api(sleep=mock.AsyncMock())):
            await asyncio.gather(self.store.requeue_task(task), self.store.requeue_task(task))
        retried = await self.store.get_next_task()
        self.assertEqual(retried.rollout_id, task.rollout_id)
        self.assertEqual(retried.num_claims, 2)
        self.assertIsNone(await self.store.get_next_task())


class AgentFlowServerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_starts_without_background_tasks_and_closes(self):
        server = SERVER.AgentFlowServer()
        tasks_before = asyncio.all_tasks()
        async with server._lifespan(server._app):
            await asyncio.sleep(0)
            self.assertTrue(server.startup_event.is_set())
            self.assertIs(server.loop, asyncio.get_running_loop())
            self.assertIsInstance(server._store, SERVER.ServerDataStore)
            self.assertEqual(asyncio.all_tasks(), tasks_before)
        self.assertIsNone(server.loop)
        self.assertIsNone(server._store)
        self.assertFalse(server.startup_event.is_set())

    async def test_lifespan_cleans_up_on_exception(self):
        server = SERVER.AgentFlowServer()
        with self.assertRaisesRegex(RuntimeError, "test shutdown"):
            async with server._lifespan(server._app):
                self.assertTrue(server.startup_event.is_set())
                self.assertIs(server.loop, asyncio.get_running_loop())
                self.assertIsInstance(server._store, SERVER.ServerDataStore)
                raise RuntimeError("test shutdown")
        self.assertIsNone(server.loop)
        self.assertIsNone(server._store)
        self.assertFalse(server.startup_event.is_set())

    async def test_task_poll_requeues_only_after_legacy_timeout(self):
        server = SERVER.AgentFlowServer(task_timeout_seconds=12000)
        clock = mock.Mock(return_value=1000.0)
        with mock.patch.object(SERVER, "time", SimpleNamespace(time=clock)), \
             mock.patch.object(SERVER, "asyncio", async_api(sleep=mock.AsyncMock())):
            async with server._lifespan(server._app):
                poll = next(route.endpoint for route in server._app.routes if route.path == "/task")
                rollout_id = await server.queue_task({"question": "sample"})
                first = await poll()
                self.assertTrue(first.is_available)
                self.assertEqual(first.task.rollout_id, rollout_id)
                self.assertEqual(first.task.num_claims, 1)

                for elapsed in (601, 12000):
                    clock.return_value = first.task.last_claim_time + elapsed
                    self.assertFalse((await poll()).is_available)
                    self.assertIs(server._store.get_processing_tasks()[rollout_id], first.task)

                clock.return_value = first.task.last_claim_time + 12001
                retried = await poll()
                self.assertTrue(retried.is_available)
                self.assertEqual(retried.task.rollout_id, rollout_id)
                self.assertEqual(retried.task.num_claims, 2)
                self.assertIsNone(await server._store.get_next_task())

    async def test_http_acknowledges_duplicates_and_rejects_cleared_result(self):
        server = SERVER.AgentFlowServer()
        async with server._lifespan(server._app):
            resource_id = await server.update_resources({
                "main_llm": PROTOCOL.LLM(endpoint="http://localhost/v1", model="test")})
            rollout_id = await server.queue_task({"question": "sample"}, resources_id=resource_id)
            await server._store.get_next_task()
            post = next(route.endpoint for route in server._app.routes if route.path == "/rollout")
            result = PROTOCOL.Rollout(rollout_id=rollout_id, final_reward=0.0,
                                     metadata={"resources_id": resource_id})
            self.assertEqual((await post(result)).status, "ok")
            self.assertEqual(await server.retrieve_completed_rollouts(), [result])
            self.assertEqual((await post(result)).status, "ok")
            self.assertEqual(await server.retrieve_completed_rollouts(), [])
            await server.clear_queues()
            self.assertEqual((await post(result)).status, "rejected")
            self.assertEqual(await server.retrieve_completed_rollouts(), [])


if __name__ == "__main__":
    unittest.main()
