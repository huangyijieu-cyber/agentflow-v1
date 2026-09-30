"""Test Wikipedia cooldowns without real requests or sleeping."""
import ast
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def load_get(responses):
    tree = ast.parse((ROOT / 'agentflow/agentflow/tools/wikipedia_search/tool.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ('_patched_get', 'WikipediaRateLimitError')]
    ns = runpy.run_path(str(ROOT / 'agentflow/agentflow/tools/network_retry.py'))
    ns.update(_original_get=Mock(side_effect=responses),
              time=SimpleNamespace(sleep=Mock()),
              requests=SimpleNamespace(exceptions=SimpleNamespace(
                  ConnectionError=ConnectionError, Timeout=TimeoutError)), print=Mock())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<wiki-retry>', 'exec'), ns)
    return ns


def response(status, retry_after=None):
    return SimpleNamespace(status_code=status,
                           headers={} if retry_after is None else {'Retry-After': retry_after},
                           close=Mock(), raise_for_status=Mock())


class WikipediaRetryTests(unittest.TestCase):
    def test_long_retry_after_is_waited(self):
        for delay in ('30', '60', '120'):
            with self.subTest(delay=delay):
                success = response(200)
                ns = load_get([response(429, delay), success])
                self.assertIs(ns['_patched_get']('https://en.wikipedia.org/w/api.php'), success)
                ns['time'].sleep.assert_called_once_with(float(delay) + 0.5)
                self.assertEqual(ns['_original_get'].call_count, 2)

    def test_still_stops_after_three_retries(self):
        ns = load_get([response(429, '60') for _ in range(4)])
        with self.assertRaisesRegex(ns['WikipediaRateLimitError'], 'after 3 retries'):
            ns['_patched_get']('https://en.wikipedia.org/w/api.php')
        self.assertEqual(ns['_original_get'].call_count, 4)
        self.assertEqual(ns['time'].sleep.call_count, 3)
        self.assertEqual([c.args[0] for c in ns['time'].sleep.call_args_list], [60.5] * 3)

    def test_missing_header_uses_backoff(self):
        ns = load_get([response(429) for _ in range(3)] + [response(200)])
        ns['_patched_get']('https://en.wikipedia.org/w/api.php')
        for call, base in zip(ns['time'].sleep.call_args_list, (1, 2, 4)):
            self.assertGreaterEqual(call.args[0], base)
            self.assertLessEqual(call.args[0], base + 0.5)
        self.assertEqual(ns['time'].sleep.call_count, 3)


if __name__ == '__main__':
    unittest.main()
