"""CPU lifecycle regressions executing production daemon, worker and server code."""
import ast
import asyncio
import concurrent.futures
import contextlib
import copy
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
import typing
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def production_class(path, name, scope):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), scope)
    return scope[name]


def load_server():
    path = Path(__file__).with_name("test_server_task_lifecycle.py")
    spec = importlib.util.spec_from_file_location("daemon_server_test_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROTOCOL, module.SERVER


PROTOCOL, SERVER = load_server()


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    async def sleep(self, delay):
        self.now += delay
        await asyncio.sleep(0)


class ScriptServer:
    def __init__(self, batches=()):
        self.batches = list(batches)
        self.queued = []
        self.clear_queues = mock.AsyncMock()

    async def retrieve_completed_rollouts(self):
        if not self.batches:
            return []
        batch = self.batches.pop(0)
        if isinstance(batch, BaseException):
            raise batch
        return batch

    async def queue_task(self, **kwargs):
        self.queued.append(kwargs)
        return f"retry-{len(self.queued)}"


def make_daemon(server=None, roots=("root",), **kwargs):
    clock = Clock()
    scope = dict(vars(typing), asyncio=SimpleNamespace(**{**vars(asyncio), "sleep": clock.sleep}),
                 concurrent=concurrent, threading=threading, math=math, os=os, uuid=uuid, json=json,
                 time=clock, logger=LOGGER, Rollout=PROTOCOL.Rollout, LLM=PROTOCOL.LLM,
                 NamedResources=PROTOCOL.NamedResources, profiling_enabled=lambda: False,
                 AgentFlowServer=lambda **kw: SimpleNamespace(**kw), _find_available_port=lambda: 0)
    cls = production_class("agentflow/verl/daemon.py", "AgentModeDaemon", scope)
    daemon = cls(9999, len(roots), {}, None, 1, 0, **kwargs)
    daemon.server = server or ScriptServer()
    daemon._total_tasks_queued = len(roots)
    # Repeated rollouts for one question must have independent retry budgets.
    daemon._task_id_to_original_sample = {root: {"data_id": "same-question", "question": "q"} for root in roots}
    daemon._current_resources_id = "policy-1"
    return daemon, clock, scope


def result(rollout_id="root", *, error=False, reward=0.0, invalid=False, exhausted=False):
    return PROTOCOL.Rollout(
        rollout_id=rollout_id, final_reward=None if error else reward,
        triplets=None if error or invalid else [PROTOCOL.Triplet(
            prompt={"token_ids": [1]}, response={"token_ids": [2]}, reward=reward)],
        metadata={"resources_id": "policy-1", **({"rollout_error": {
            "type": "TaskTimeout" if exhausted else "RuntimeError", "phase": "rollout",
            **({"retry_exhausted": True} if exhausted else {})}} if error else {})})


class DaemonLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_group_waits_before_cleanup_and_preserves_zero_reward(self):
        for is_train in (True, False):
            with self.subTest(is_train=is_train):
                server = ScriptServer([[result("a"), result("b", reward=1)]])
                daemon, clock, scope = make_daemon(
                    server, roots=("a", "b"), rollout_timeout_seconds=1,
                    rollout_no_progress_timeout_seconds=1)
                daemon.is_train = is_train
                entered, release = asyncio.Event(), asyncio.Event()

                async def completion_delay(delay):
                    entered.set()
                    await release.wait()
                    await clock.sleep(delay)

                sleep = mock.AsyncMock(side_effect=completion_delay)
                scope["asyncio"].sleep = sleep
                task = asyncio.create_task(daemon._async_run_until_finished(False))
                try:
                    await asyncio.wait_for(entered.wait(), timeout=1)
                    self.assertEqual([r.final_reward for r in daemon._completed_rollouts.values()], [0.0, 1.0])
                    sleep.assert_awaited_once_with(2)
                    self.assertFalse(task.done())
                    server.clear_queues.assert_not_awaited()
                finally:
                    release.set()
                    await task
                self.assertEqual(clock.now, 2)
                self.assertEqual(server.queued, [])
                server.clear_queues.assert_awaited_once()

    async def test_same_question_rollouts_each_receive_full_retry_budget(self):
        server = ScriptServer([[result("a", error=True), result("b", error=True)],
                               [result("retry-1", error=True), result("retry-2", error=True)],
                               [result("retry-3"), result("retry-4")]])
        daemon, _, _ = make_daemon(server, roots=("a", "b"))
        await daemon._async_run_until_finished(False)
        self.assertEqual(daemon._empty_rollout_counts, {"a": 2, "b": 2})
        self.assertEqual(daemon._total_tasks_queued, 2)
        self.assertEqual(set(daemon._completed_rollouts), {"retry-3", "retry-4"})
        self.assertEqual(set(daemon._task_id_to_original_sample), {"retry-3", "retry-4"})
        self.assertEqual([q["metadata"]["original_rollout_id"] for q in server.queued], ["a", "b", "a", "b"])

    async def test_reported_errors_retry_in_validation_and_with_validation_disabled(self):
        for train, enabled in ((False, True), (True, False)):
            with self.subTest(train=train, enabled=enabled):
                server = ScriptServer([[result(error=True)], [result("retry-1")]])
                daemon, _, _ = make_daemon(server, enable_rollout_validation=enabled)
                daemon.is_train = train
                await daemon._async_run_until_finished(False)
                self.assertEqual(list(daemon._completed_rollouts), ["retry-1"])
                self.assertEqual(server.queued[0]["mode"], "train" if train else "val")

    async def test_empty_training_data_retries_without_changing_original_slot_count(self):
        server = ScriptServer([[result(invalid=True)], [result("retry-1")]])
        daemon, _, _ = make_daemon(server)
        await daemon._async_run_until_finished(False)
        self.assertEqual(daemon._total_tasks_queued, 1)
        self.assertEqual(list(daemon._completed_rollouts), ["retry-1"])

    async def test_retry_exhaustion_aborts_without_counting_failure_as_completion(self):
        server = ScriptServer([[result(error=True)], [result("retry-1", error=True)],
                               [result("retry-2", error=True)]])
        daemon, _, _ = make_daemon(server)
        with self.assertRaisesRegex(RuntimeError, "exhausted after 2 resubmissions"):
            await daemon._async_run_until_finished(False)
        self.assertEqual(daemon._completed_rollouts, {})
        self.assertEqual(len(server.queued), 2)
        server.clear_queues.assert_awaited_once()

    async def test_server_stale_exhaustion_is_terminal(self):
        server = ScriptServer([[result(error=True, exhausted=True)]])
        daemon, _, scope = make_daemon(server)
        scope["asyncio"].sleep = mock.AsyncMock()
        with self.assertRaisesRegex(RuntimeError, "TaskTimeout"):
            await daemon._async_run_until_finished(False)
        scope["asyncio"].sleep.assert_not_awaited()
        self.assertEqual(server.queued, [])
        self.assertEqual(daemon._completed_rollouts, {})

    async def test_duplicate_orphan_and_replaced_attempt_cannot_fill_slots(self):
        server = ScriptServer([[result("a"), result("a"), result("b", error=True)],
                               [result("b"), result("orphan"), result("retry-1")]])
        daemon, _, _ = make_daemon(server, roots=("a", "b"))
        await daemon._async_run_until_finished(False)
        self.assertEqual(set(daemon._completed_rollouts), {"a", "retry-1"})

    async def test_retry_queue_failure_propagates_and_cleans_up(self):
        server = ScriptServer([[result(error=True)]])
        server.queue_task = mock.AsyncMock(side_effect=ConnectionError("queue unavailable"))
        daemon, _, _ = make_daemon(server)
        with self.assertRaisesRegex(ConnectionError, "queue unavailable"):
            await daemon._async_run_until_finished(False)
        self.assertEqual(daemon._completed_rollouts, {})
        server.clear_queues.assert_awaited_once()

    async def test_no_progress_aborts_even_after_partial_success(self):
        server = ScriptServer([[result("a")]])
        daemon, clock, _ = make_daemon(server, roots=("a", "b"), rollout_no_progress_timeout_seconds=3)
        with self.assertRaisesRegex(TimeoutError, "No rollout progress.*1/2 completed"):
            await daemon._async_run_until_finished(False)
        self.assertEqual(list(daemon._completed_rollouts), ["a"])
        self.assertEqual(clock.now, 3)
        server.clear_queues.assert_awaited_once()

    async def test_total_deadline_aborts_despite_retry_progress(self):
        server = ScriptServer([[result(error=True)], [result("retry-1", error=True)]])
        daemon, clock, _ = make_daemon(server, rollout_timeout_seconds=3, rollout_no_progress_timeout_seconds=10)
        with self.assertRaisesRegex(TimeoutError, "batch exceeded 3s"):
            await daemon._async_run_until_finished(False)
        self.assertEqual(clock.now, 3)
        self.assertEqual(daemon._completed_rollouts, {})

    async def test_only_opt_in_validation_profile_can_return_partial(self):
        server = ScriptServer([[result("a")]])
        daemon, _, scope = make_daemon(server, roots=("a", "b"))
        daemon.is_train = False
        scope["profiling_enabled"] = lambda: True
        with mock.patch.dict(os.environ, {"AGENTFLOW_PROFILE_VAL_TIMEOUT_S": "1"}):
            await daemon._async_run_until_finished(False)
        self.assertEqual(list(daemon._completed_rollouts), ["a"])
        server.clear_queues.assert_awaited_once()

    async def test_invalid_profile_config_also_cleans_up(self):
        daemon, _, scope = make_daemon()
        daemon.is_train = False
        scope["profiling_enabled"] = lambda: True
        with mock.patch.dict(os.environ, {"AGENTFLOW_PROFILE_VAL_TIMEOUT_S": "bad"}), self.assertRaises(ValueError):
            await daemon._async_run_until_finished(False)
        daemon.server.clear_queues.assert_awaited_once()

    async def test_cleanup_error_preserves_retry_exhaustion_root_cause(self):
        server = ScriptServer([[result(error=True)]])
        server.clear_queues.side_effect = ConnectionError("cleanup unavailable")
        daemon, _, _ = make_daemon(server, max_empty_retries=0)
        with self.assertLogs(LOGGER, logging.ERROR), self.assertRaisesRegex(RuntimeError, "retry budget exhausted"):
            await daemon._async_run_until_finished(False)

    async def test_cleanup_error_after_success_is_not_silenced(self):
        server = ScriptServer([[result()]])
        server.clear_queues.side_effect = ConnectionError("cleanup unavailable")
        daemon, _, _ = make_daemon(server)
        with self.assertRaisesRegex(ConnectionError, "cleanup unavailable"):
            await daemon._async_run_until_finished(False)

    async def test_cancellation_cleans_up_and_survives_cleanup_error(self):
        server = ScriptServer()
        entered = asyncio.Event()

        async def stalled():
            entered.set()
            await asyncio.Event().wait()

        server.retrieve_completed_rollouts = stalled
        server.clear_queues.side_effect = ConnectionError("cleanup unavailable")
        daemon, _, _ = make_daemon(server)
        task = asyncio.create_task(daemon._async_run_until_finished(False))
        await entered.wait()
        task.cancel()
        with self.assertLogs(LOGGER, logging.ERROR), self.assertRaises(asyncio.CancelledError):
            await task
        server.clear_queues.assert_awaited_once()
        self.assertEqual(daemon._completed_rollouts, {})


class NetworkHealthTests(unittest.TestCase):
    def turn(self, index, tool="Yibu_Brave_Search_Tool", *, external_error=False,
             turn_type="tool_plan", empty_tokens=False, **metadata):
        return PROTOCOL.Triplet(
            prompt={"token_ids": [] if empty_tokens else [1]},
            response={"token_ids": [] if empty_tokens else [2]},
            metadata={"trace_task": None, "trace_turn_type": turn_type,
                      "trace_original_turn": index,
                      "trace_action_key": json.dumps([tool, "execution = tool.execute(query='q')"]),
                      "trace_external_error": external_error, **metadata})

    def daemon(self, *rollouts):
        daemon, _, _ = make_daemon(roots=tuple(rollout.rollout_id for rollout in rollouts))
        daemon._completed_rollouts = {rollout.rollout_id: rollout for rollout in rollouts}
        return daemon

    def rollout(self, triplets, rollout_id="root"):
        return PROTOCOL.Rollout(rollout_id=rollout_id, final_reward=0.0, triplets=triplets)

    def test_three_network_tools_use_raw_rollout_denominator(self):
        daemon = self.daemon(
            self.rollout([self.turn(1, "Web_RAG_Search_Tool", external_error=True),
                          self.turn(2, "Yibu_Brave_Search_Tool", external_error=True)], "a"),
            self.rollout([self.turn(1, "Wikipedia_RAG_Search_Tool", external_error=True,
                                   trace_task="qa")], "b"),
            self.rollout([self.turn(0, turn_type="base", trace_action_key=None)], "c"))
        self.assertEqual(daemon.get_network_health(), {
            "network_turns": 3, "external_error_turns": 3,
            "affected_rollouts": 2, "total_rollouts": 3,
            "metadata_complete": True, "missing_metadata_turns": 0,
            "uninspectable_rollouts": 0, "metadata_issues": {}})

    def test_excludes_local_environment_and_non_tool_turns(self):
        turns = [self.turn(0, "Python_Coder_Tool", external_error=True),
                 self.turn(1, "Web_RAG_Search_Tool", external_error=True, trace_task="alfworld"),
                 self.turn(2, "Yibu_Brave_Search_Tool", external_error=True, trace_task="webshop"),
                 self.turn(3, None, external_error=False)]
        turns.extend(self.turn(index + 4, external_error=True, turn_type=kind,
                               trace_action_key=None, trace_original_turn=None)
                     for index, kind in enumerate(("analysis", "final_answer", "base", "direct")))
        health = self.daemon(self.rollout(turns)).get_network_health()
        self.assertEqual(health["network_turns"], 0)
        self.assertEqual(health["external_error_turns"], 0)
        self.assertEqual(health["affected_rollouts"], 0)
        self.assertTrue(health["metadata_complete"])

    def test_original_failure_without_tokens_survives_batch_filtering(self):
        daemon = self.daemon(self.rollout([self.turn(7, external_error=True, empty_tokens=True)]))
        before = daemon.get_network_health()
        batch, metrics = daemon.get_train_data_batch(4, 4, "cpu")
        self.assertIsNone(batch)
        self.assertEqual(metrics["agent_mode/n_sample_to_train"], 0)
        self.assertEqual(before["network_turns"], 1)
        self.assertEqual(before["external_error_turns"], 1)
        self.assertTrue(before["metadata_complete"])
        self.assertEqual(daemon.get_network_health(), before)

    def test_deduplicates_original_turns_but_not_repeated_commands(self):
        duplicate = self.turn(7, external_error=True, trace_dup=True)
        daemon = self.daemon(
            self.rollout([duplicate, duplicate, self.turn(8, trace_dup=True)], "a"),
            self.rollout([self.turn(7, external_error=True)], "b"))
        health = daemon.get_network_health()
        self.assertEqual(health["network_turns"], 3)
        self.assertEqual(health["external_error_turns"], 2)
        self.assertEqual(health["affected_rollouts"], 2)
        self.assertTrue(health["metadata_complete"])
        daemon.clear_data_and_server()
        self.assertEqual(daemon.get_network_health()["total_rollouts"], 0)
        self.assertEqual(daemon.get_network_health()["network_turns"], 0)

    def test_final_success_and_normal_empty_results_follow_existing_classifier(self):
        spec = importlib.util.spec_from_file_location(
            "network_health_execution", ROOT / "agentflow/agentflow/trace/execution.py")
        execution = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(execution)
        results = [None, "", [], {"results": []}, "No search results",
                   {"error": "No results found for query: q"},
                   {"success": True, "error": "ConnectionError",
                    "attempts": [{"error": "ConnectionError"}], "results": ["evidence"]},
                   {"status": "ok", "error": "HTTP 503", "results": ["evidence"]},
                   {"error": "ConnectionError: peer reset connection"}]
        turns = []
        for index, tool_result in enumerate(results):
            status = execution.classify_tool_result(tool_result)
            turns.append(self.turn(index, external_error=status["external_error"],
                                   trace_tool_failed=status["failed"],
                                   trace_error_type=status["error_type"],
                                   trace_error_source=status["error_source"],
                                   trace_outcome_success=True))
        health = self.daemon(self.rollout(turns)).get_network_health()
        self.assertEqual(health["network_turns"], len(results))
        self.assertEqual(health["external_error_turns"], 1)
        self.assertTrue(health["metadata_complete"])

    def test_actual_qa_model_errors_survive_both_production_encoders(self):
        modules = {}
        for name, path in {
            "formatters": "agentflow/agentflow/models/formatters.py",
            "utils": "agentflow/agentflow/models/utils.py",
            "execution": "agentflow/agentflow/trace/execution.py",
        }.items():
            spec = importlib.util.spec_from_file_location("network_health_" + name, ROOT / path)
            modules[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modules[name])
        formatters, utils, execution = (modules[name] for name in ("formatters", "utils", "execution"))
        planner_cls = production_class("agentflow/agentflow/models/planner.py", "Planner", {
            **vars(typing), "os": os, "re": re, "Memory": SimpleNamespace,
            "NextStep": formatters.NextStep, "QueryAnalysis": formatters.QueryAnalysis,
            "robust_json_loads": utils.robust_json_loads})
        executor_cls = production_class("agentflow/agentflow/models/executor.py", "Executor", {
            **vars(typing), "re": re, "ToolCommand": formatters.ToolCommand,
            "robust_json_loads": utils.robust_json_loads,
            # The cases below generate no executable command block.
            "ToolExecutionContext": lambda timeout: None})
        solver_cls = production_class("agentflow/agentflow/solver.py", "Solver", {
            **vars(typing), "time": time, "json": json, "deepcopy": copy.deepcopy,
            "nullcontext": contextlib.nullcontext, "profiling_enabled": lambda: False,
            "log_info": lambda message: None, "is_search_tool": lambda tool: True,
            "classify_tool_result": execution.classify_tool_result,
            "make_json_serializable_truncated": utils.make_json_serializable_truncated})
        encoders = []
        for path in ("train-roma/rollout.py", "train-roma/rollout_main_profiled.py"):
            tree = ast.parse((ROOT / path).read_text())
            node = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "encode_logs")
            scope = {"Triplet": PROTOCOL.Triplet}
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
            encoders.append((path, scope["encode_logs"]))

        selected = formatters.NextStep(justification="reason", context="context",
                                       sub_goal="goal", tool_name="Yibu_Brave_Search_Tool")
        unknown = selected.model_copy(update={"tool_name": "Unknown_Tool"})
        cases = [("invalid-format", "unstructured planner response", "unused"),
                 ("missing-tool", '{"context":"c","sub_goal":"s"}', "unused"),
                 ("unknown-tool", unknown, "unused"),
                 ("missing-command", selected, "unstructured executor response"),
                 ("invalid-command-type", selected, None)]
        for name, next_step, command_response in cases:
            with self.subTest(case=name), mock.patch("builtins.print"):
                def engine(prompt, **kwargs):
                    return next_step if kwargs.get("usage_by") == "[planner] next step" else "answer"
                engine.last_generation_metadata = {
                    "prompt_token_ids": [1], "response_token_ids": [2], "finish_reason": "stop"}
                planner = object.__new__(planner_cls)
                planner.llm_engine, planner.logs = engine, []
                planner.available_tools = ["Yibu_Brave_Search_Tool"]
                planner.toolbox_metadata = {"Yibu_Brave_Search_Tool": {}}
                planner.is_multimodal, planner.temperature = False, 0.0
                tool = SimpleNamespace(set_custom_output_dir=mock.Mock(), execute=mock.Mock())
                executor = object.__new__(executor_cls)
                executor.tool_instances_cache = {"Yibu_Brave_Search_Tool": tool}
                executor.max_time, executor.temperature, executor.query_cache_dir = 1, 0.0, "unused"
                executor.set_query_cache_dir = mock.Mock()
                executor.llm_generate_tool_command = lambda *args, **kwargs: command_response
                memory = SimpleNamespace(clear=lambda: None, get_actions=lambda: {},
                                         get_all_actions=lambda: [], add_action=mock.Mock())
                verifier = SimpleNamespace(verificate_context=lambda *args: None,
                                           extract_conclusion=lambda response: ("", "STOP"))
                solver = solver_cls(planner, verifier, memory, executor,
                                    max_steps=1, verbose=False, task="qa")
                output, _ = solver.solve("question")
                logs = output["planner_logs"]
                self.assertEqual([log["trace_turn_type"] for log in logs],
                                 ["base", "analysis", "tool_plan", "final_answer", "direct"])
                self.assertIsNotNone(logs[2]["trace_action_key"])
                tool.execute.assert_not_called()
                for path, encode in encoders:
                    with self.subTest(encoder=path):
                        triplets = encode(None, logs) if "profiled" in path else encode(logs)
                        health = self.daemon(self.rollout(triplets)).get_network_health()
                        self.assertEqual(health["network_turns"], 0)
                        self.assertEqual(health["external_error_turns"], 0)
                        self.assertTrue(health["metadata_complete"], health)

    def test_unexecuted_command_placeholders_cannot_hide_external_or_missing_flags(self):
        for command in ("No command found.", "Invalid response type."):
            for error in (True, None):
                with self.subTest(command=command, error=error):
                    turn = self.turn(0, external_error=error, trace_action_key=json.dumps([
                        "Yibu_Brave_Search_Tool", command]))
                    health = self.daemon(self.rollout([turn])).get_network_health()
                    self.assertEqual(health["network_turns"], 1)
                    self.assertEqual(health["external_error_turns"], int(error is True))
                    self.assertEqual(health["metadata_complete"], error is True)

    def test_missing_and_invalid_metadata_never_becomes_a_healthy_batch(self):
        cases = [({}, "unknown_trace_turn_type"),
                 ({"trace_turn_type": []}, "unknown_trace_turn_type"),
                 ({"trace_task": "unknown"}, "unknown_trace_task"),
                 ({"trace_action_key": None, "trace_extraction_valid": False,
                   "trace_tool_failed": True}, "invalid_trace_action_key"),
                 ({"trace_action_key": "malformed"}, "invalid_trace_action_key"),
                 ({"trace_original_turn": None}, "invalid_trace_original_turn"),
                 ({"trace_original_turn": True}, "invalid_trace_original_turn"),
                 ({"trace_external_error": None}, "invalid_trace_external_error"),
                 ({"trace_external_error": "false"}, "invalid_trace_external_error")]
        for metadata, reason in cases:
            with self.subTest(metadata=metadata):
                turn = self.turn(0, **metadata)
                if not metadata:
                    turn.metadata = {}
                health = self.daemon(self.rollout([turn])).get_network_health()
                self.assertFalse(health["metadata_complete"])
                self.assertEqual(health["missing_metadata_turns"], 1)
                self.assertEqual(health["metadata_issues"], {reason: 1})

    def test_missing_rollout_payload_is_explicitly_uninspectable(self):
        for triplets in (None, []):
            with self.subTest(triplets=triplets):
                health = self.daemon(self.rollout(triplets)).get_network_health()
                self.assertFalse(health["metadata_complete"])
                self.assertEqual(health["total_rollouts"], 1)
                self.assertEqual(health["uninspectable_rollouts"], 1)

    def test_conflicting_copies_of_an_original_turn_are_explicitly_incomplete(self):
        for duplicate in (self.turn(7, external_error=True),
                          self.turn(7, "Python_Coder_Tool"),
                          self.turn(7, trace_action_key=json.dumps([
                              "Yibu_Brave_Search_Tool", "different command"]))):
            with self.subTest(duplicate=duplicate):
                health = self.daemon(self.rollout([self.turn(7), duplicate])).get_network_health()
                self.assertFalse(health["metadata_complete"])
                self.assertEqual(health["network_turns"], 1)
                self.assertEqual(health["missing_metadata_turns"], 1)
                self.assertEqual(health["metadata_issues"], {"conflicting_duplicate_turn": 1})


class SyncDeadlineTests(unittest.TestCase):
    def test_thread_side_timeout_cancels_stalled_future(self):
        daemon, _, scope = make_daemon(rollout_timeout_seconds=7)
        daemon.server.loop = mock.Mock(is_running=lambda: True)
        daemon.server.startup_event = mock.Mock(is_set=lambda: True)
        future = mock.Mock()
        future.result.side_effect = concurrent.futures.TimeoutError()
        future.done.return_value = False

        def dispatch(coro, loop):
            coro.close()
            return future

        scope["asyncio"].run_coroutine_threadsafe = dispatch
        with self.assertRaisesRegex(TimeoutError, "did not respond within 22.0s"):
            daemon.run_until_all_finished()
        future.result.assert_called_once_with(timeout=22.0)
        future.cancel.assert_called_once()

    def test_inner_timeout_keeps_specific_cause(self):
        daemon, _, scope = make_daemon()
        daemon.server.loop = mock.Mock(is_running=lambda: True)
        daemon.server.startup_event = mock.Mock(is_set=lambda: True)
        future = mock.Mock()
        future.result.side_effect = TimeoutError("No rollout progress root cause")
        future.done.return_value = True

        def dispatch(coro, loop):
            coro.close()
            return future

        scope["asyncio"].run_coroutine_threadsafe = dispatch
        with self.assertRaisesRegex(TimeoutError, "No rollout progress root cause"):
            daemon.run_until_all_finished()
        future.cancel.assert_not_called()

    def test_timeout_and_retry_settings_reject_invalid_values(self):
        for name in ("rollout_timeout_seconds", "rollout_no_progress_timeout_seconds"):
            for value in (0, -1, float("nan"), float("inf"), True, "1"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    make_daemon(**{name: value})
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_empty_retries"):
                make_daemon(max_empty_retries=value)

    def test_task_timeout_keeps_the_original_llm_timeout(self):
        daemon, _, _ = make_daemon()
        for kwargs, expected in (({}, 12000), ({"llm_timeout_seconds": 1800}, 1800)):
            with self.subTest(kwargs=kwargs):
                configured = type(daemon)(9999, 1, {}, None, 1, 0, **kwargs)
                self.assertEqual(configured.server.task_timeout_seconds, expected)


class EndToEndRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, failures):
        # Real Judge response handling, Runner, ServerDataStore and Daemon; only
        # model responses and transport are local so no accelerator/API is needed.
        store = SERVER.ServerDataStore()
        await store.update_resources(PROTOCOL.ResourcesUpdate(resources_id="policy-1", resources={
            "main_llm": PROTOCOL.LLM(endpoint="http://unused/v1", model="test")}))
        root = await store.add_task({"question": "q", "data_id": "same-question"}, resources_id="policy-1")
        attempts, reports = [], []
        score_scope = {"re": re, "AnswerVerification": object}

        def judge(*args, **kwargs):
            if len(attempts) <= failures:
                return {"error": "APIConnectionError", "message": "unreachable"}
            return SimpleNamespace(true_false=False)

        score_scope["llm_scorer_engine"] = judge
        tree = ast.parse((ROOT / "train-roma/utils.py").read_text())
        score_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_score")
        exec(compile(ast.Module(body=[score_node], type_ignores=[]), "utils.py", "exec"), score_scope)

        async def rollout(sample, rollout_id, resources):
            attempts.append(rollout_id)
            score = score_scope["compute_score"]("q", "gold", "answer")
            return result(rollout_id, reward=float(score))

        async def post(payload):
            # Serialize just as an actual RPC would, using the production schema.
            payload = PROTOCOL.Rollout.model_validate_json(payload.model_dump_json())
            reports.append(payload)
            return {"status": "ok" if await store.store_rollout(payload) else "rejected"}

        client = SimpleNamespace(poll_next_task_async=store.get_next_task,
                                 get_resources_by_id_async=store.get_resources_by_id,
                                 get_latest_resources_async=store.get_latest_resources,
                                 post_rollout_async=post)
        agent = SimpleNamespace(set_runner=lambda runner: None, training_rollout_async=rollout,
                                validation_rollout_async=rollout)
        runner_scope = dict(vars(typing), **{name: getattr(PROTOCOL, name) for name in (
            "Rollout", "Task", "Triplet", "ParallelWorkerBase", "RolloutRawResult")},
            asyncio=asyncio, json=json, logging=logging, math=math, os=os, time=time,
            AgentFlowClient=typing.Any, LitAgent=typing.Any, TripletExporter=typing.Any,
            ReadableSpan=object, logger=LOGGER)
        runner_cls = production_class("agentflow/runner.py", "AgentRunner", runner_scope)
        runner = runner_cls(agent, client, None)

        async def retrieve():
            await runner.run_async()
            return await store.retrieve_completed_rollouts()

        adapter = SimpleNamespace(retrieve_completed_rollouts=retrieve, queue_task=store.add_task,
                                  clear_queues=store.clear_queues)
        daemon, _, _ = make_daemon(adapter, roots=(root,))
        try:
            await daemon._async_run_until_finished(False)
        except RuntimeError:
            if failures <= 2:
                raise
        else:
            self.assertLessEqual(failures, 2, "persistent failure must abort")
        self.assertIsNone(await store.get_next_task())
        self.assertEqual(store.get_processing_tasks(), {})
        self.assertEqual(await store.retrieve_completed_rollouts(), [])
        for payload in reports[:failures]:
            self.assertIsNone(payload.final_reward)
            self.assertIsNone(payload.triplets)
            self.assertEqual(payload.metadata["rollout_error"]["type"], "RuntimeError")
            self.assertFalse(await store.store_rollout(payload), "late old-batch report must be fenced")
        return daemon, attempts, reports

    async def test_judge_connection_failure_recovers_and_only_success_enters_batch(self):
        daemon, attempts, reports = await self.exercise(1)
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(attempts[0], attempts[1])
        self.assertEqual(list(daemon._completed_rollouts), [attempts[-1]])
        self.assertEqual(reports[-1].final_reward, 0.0)
        self.assertEqual(daemon._total_tasks_queued, 1)

    async def test_persistent_judge_connection_failure_exhausts_without_training_payload(self):
        daemon, attempts, reports = await self.exercise(3)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(reports), 3)
        self.assertEqual(daemon._completed_rollouts, {})


if __name__ == "__main__":
    unittest.main()
