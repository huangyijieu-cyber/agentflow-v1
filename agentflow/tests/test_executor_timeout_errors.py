"""Execute the real Executor class with fake tools, without model dependencies."""
import ast
import re
import threading
import time
import unittest
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentflow.tools.execution_context import ToolExecutionContext, bind_context, run_bounded


path = Path(__file__).resolve().parents[1] / "agentflow/models/executor.py"
tree = ast.parse(path.read_text())
node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Executor")
scope = dict(Any=Any, Dict=Dict, List=List, Optional=Optional, re=re, threading=threading,
             ToolExecutionContext=ToolExecutionContext, bind_context=bind_context, run_bounded=run_bounded)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
Executor = scope["Executor"]


class Tool:
    def __init__(self, execute):
        self.execute = execute

    def set_custom_output_dir(self, directory):
        pass


class ExecutorErrorTests(unittest.TestCase):
    def executor(self, execute, timeout=0.02):
        executor = Executor.__new__(Executor)
        executor.max_time = timeout
        executor.query_cache_dir = "/unused-test-cache"
        executor.tool_instances_cache = {"test": Tool(execute)}
        return executor

    def test_immediate_proxy_error_returns_without_waiting_for_timeout(self):
        def fail():
            raise RuntimeError("upstream_proxy_error HTTP 407")
        executor = self.executor(fail, timeout=10)
        started = time.monotonic()
        result = executor.execute_tool_command("test", "execution = tool.execute()")
        self.assertIn("upstream_proxy_error HTTP 407", result)
        self.assertNotIn("Execution timed out", result)
        self.assertLess(time.monotonic() - started, 1)

    def test_error_during_cancellation_grace_is_preserved(self):
        def fail():
            time.sleep(0.05)
            raise RuntimeError("upstream_proxy_error HTTP 502")
        result = self.executor(fail).execute_tool_command("test", "execution = tool.execute()")
        self.assertIn("upstream_proxy_error HTTP 502", result)
        self.assertNotIn("Execution timed out", result)

    def test_completed_error_output_during_grace_is_not_replaced(self):
        def error_result():
            time.sleep(0.05)
            return "Error fetching URL: upstream_timeout"
        result = self.executor(error_result).execute_tool_command("test", "execution = tool.execute()")
        self.assertEqual(result, ["Error fetching URL: upstream_timeout"])

    def test_true_timeout_does_not_start_later_command_blocks(self):
        release, done = threading.Event(), threading.Event()
        calls = []
        def hold():
            calls.append(1)
            release.wait(2)
            done.set()
            return "late result"
        try:
            result = self.executor(hold).execute_tool_command(
                "test", "execution = tool.execute()\nexecution = tool.execute()"
            )
            self.assertEqual(len(result), 1)
            self.assertIn("Execution timed out", result[0])
        finally:
            release.set()
            self.assertTrue(done.wait(1))
        self.assertEqual(len(calls), 1)

    def test_command_blocks_share_one_total_budget(self):
        calls = []
        def slow():
            calls.append(1)
            time.sleep(0.04)
            return "result"
        executor = self.executor(slow, timeout=0.06)
        result = executor.execute_tool_command(
            "test", "execution = tool.execute()\nexecution = tool.execute()\nexecution = tool.execute()"
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
