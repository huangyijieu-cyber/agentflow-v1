"""Task-aware solver construction without importing model or search clients."""
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOLVER = ROOT / "agentflow/agentflow/solver.py"


def constructor(initializer):
    tree = ast.parse(SOLVER.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "construct_solver")
    scope = {"Initializer": initializer,
             **{name: SimpleNamespace for name in ("Planner", "Verifier", "Memory", "Executor", "Solver")}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOLVER), "exec"), scope)
    return scope[function.name]


class TaskToolRoutingTests(unittest.TestCase):
    def construct(self, initializer, task, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return constructor(initializer)(
                llm_engine_name="local-policy", task=task,
                enabled_tools=["Wikipedia_Search_Tool", "Web_Search_Tool", "Brave_Search_Tool"],
                tool_engine=["Default"] * 3,
                model_engine=["trainable"] * 4,
                base_url="http://127.0.0.1:19998/v1", temperature=.7, **kwargs)

    def test_environment_tasks_never_initialize_qa_search_clients(self):
        for task in ("alfworld", "webshop", "rule"):
            with self.subTest(task=task):
                initializer = mock.Mock(side_effect=AssertionError("Search tool initialization is forbidden"))
                solver = self.construct(initializer, task)
                initializer.assert_not_called()
                self.assertEqual(solver.task, task)
                self.assertEqual(solver.planner.toolbox_metadata, {})
                self.assertEqual(solver.planner.available_tools, [])
                self.assertEqual(solver.verifier.toolbox_metadata, {})
                self.assertEqual(solver.verifier.available_tools, [])
                self.assertEqual(solver.executor.tool_instances_cache, {})

    def test_environment_tasks_keep_model_rpc_and_policy_settings(self):
        for task in ("alfworld", "webshop", "rule"):
            with self.subTest(task=task):
                solver = self.construct(mock.Mock(side_effect=AssertionError("No QA tools")), task)
                for client in (solver.planner, solver.verifier, solver.executor):
                    self.assertEqual(client.llm_engine_name, "local-policy")
                    self.assertEqual(client.base_url, "http://127.0.0.1:19998/v1")
                    self.assertEqual(client.temperature, .7)
                self.assertEqual(solver.planner.llm_engine_fixed_name, "local-policy")
                self.assertEqual(solver.max_steps, 10)
                self.assertEqual(solver.max_tokens, 4000)

    def test_direct_solver_calls_normalize_the_task_before_loading_tools(self):
        initializer = mock.Mock(side_effect=AssertionError("No search clients for ALFWorld"))
        solver = self.construct(initializer, " AlFWorld ")
        self.assertEqual(solver.task, "alfworld")
        initializer.assert_not_called()

    def test_qa_and_unknown_tasks_keep_configured_tool_initialization(self):
        for task in ("qa", "custom-online-task"):
            with self.subTest(task=task):
                metadata, tools, cache = {"Brave": {"description": "search"}}, ["Brave"], {"Brave": object()}
                initializer = mock.Mock(return_value=SimpleNamespace(
                    toolbox_metadata=metadata, available_tools=tools, tool_instances_cache=cache))
                solver = self.construct(initializer, task, verbose=False)
                initializer.assert_called_once_with(
                    enabled_tools=["Wikipedia_Search_Tool", "Web_Search_Tool", "Brave_Search_Tool"],
                    tool_engine=["Default"] * 3, model_string="local-policy",
                    verbose=False, vllm_config_path=None)
                self.assertIs(solver.planner.toolbox_metadata, metadata)
                self.assertIs(solver.planner.available_tools, tools)
                self.assertIs(solver.verifier.toolbox_metadata, metadata)
                self.assertIs(solver.executor.tool_instances_cache, cache)

    def test_rollout_entrypoints_do_not_import_unused_public_ip_probe(self):
        for name in ("rollout.py", "rollout_main_profiled.py"):
            path = ROOT / "train-roma" / name
            with self.subTest(path=name):
                tree = ast.parse(path.read_text())
                imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
                self.assertNotIn("util.get_pub_ip", imports)
                addresses = [node.value.value for node in ast.walk(tree)
                             if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                             and any(isinstance(target, ast.Name) and target.id == "server_public_ip"
                                     for target in node.targets)]
                self.assertEqual(addresses, ["127.0.0.1"])


if __name__ == "__main__":
    unittest.main()
