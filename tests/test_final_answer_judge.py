"""Exercise real QA scoring code without loading GPU or model services."""
import ast
import asyncio
from pathlib import Path
import runpy
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
REWARD = runpy.run_path(str(ROOT / 'agentflow/reward.py'))['reward']


def scoring_namespace(filename, judge):
    tree = ast.parse((ROOT / 'train-roma' / filename).read_text())
    evaluate = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'evaluate')
    solve = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == '_solve_and_evaluate')
    qa = next(n for n in ast.walk(solve) if isinstance(n, ast.If) and ast.unparse(n.test) == "self.task == 'qa'")
    assignment = next(n for n in qa.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'final_reward' for t in n.targets))
    wrapper = ast.parse('async def score(task, answer, val):\n    pass\n').body[0]
    wrapper.body = [assignment, ast.Return(value=ast.Name(id='final_reward', ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[evaluate, wrapper], type_ignores=[]))
    namespace = {'asyncio': asyncio, 'reward': REWARD, 'compute_score': judge}
    exec(compile(module, filename, 'exec'), namespace)
    return namespace


class FinalAnswerJudgeTests(unittest.TestCase):
    def test_qa_uses_judge_for_train_and_validation(self):
        for filename in ('rollout.py', 'rollout_main_profiled.py'):
            for val in (False, True):
                for verdict in (True, False):
                    with self.subTest(filename=filename, val=val, verdict=verdict):
                        judge = Mock(return_value=verdict)
                        ns = scoring_namespace(filename, judge)
                        task = {'question': 'Where did it launch?', 'result': 'Apple Arcade'}
                        answer = 'It launched on Apple Arcade in 2021.'
                        score = asyncio.run(ns['score'](task, answer, val))
                        self.assertEqual(score, float(verdict))
                        judge.assert_called_once_with(task['question'], task['result'], answer)

    def test_judge_failure_is_not_silently_replaced_by_exact_match(self):
        for filename in ('rollout.py', 'rollout_main_profiled.py'):
            with self.subTest(filename=filename):
                judge = Mock(side_effect=RuntimeError('judge unavailable'))
                ns = scoring_namespace(filename, judge)
                with self.assertRaisesRegex(RuntimeError, 'judge unavailable'):
                    asyncio.run(ns['score']({'question': 'Q', 'result': 'A'}, 'A', True))


if __name__ == '__main__':
    unittest.main()
