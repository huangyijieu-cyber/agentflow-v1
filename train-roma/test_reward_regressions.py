"""Focused regressions for InfoSeek reward parsing and GiGPO turn advantages."""

import ast
import importlib.util
import json
import re
import string
import sys
import types
import unicodedata
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path: Path, names: set[str], namespace: dict) -> dict:
    """Load small pure functions without importing the rollout/training services."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    assert {node.name for node in selected} == names
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class InfoSeekRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rollout = load_functions(
            ROOT / "train-roma" / "rollout.py",
            {"_as_dict", "_normalize_entity", "_entity_mentioned", "_training_reward",
             "compute_search_subreward"},
            {
                "Any": Any,
                "json": json,
                "re": re,
                "string": string,
                "unicodedata": unicodedata,
            },
        )
        cls.daemon = load_functions(
            ROOT / "agentflow" / "verl" / "daemon.py",
            {"_gigpo_turn_step_reward"},
            {},
        )

    def test_first_hit_has_turn_and_entity_boundaries(self):
        result = {"memory": {
            "Action Step 1": {
                "tool_name": "Search",
                "result": "John   Smith was born in London.",
            },
            "Action Step 2": {
                "tool_name": "Search",
                "result": "John Smith is also mentioned here.",
            },
        }}
        spec = {
            "count_each_subgoal_once": True,
            "subreward_weight": 0.3,
            "subgoals": [{"id": "person", "answer": "John Smith", "weight": 1.0}],
        }
        reward, hits = self.rollout["compute_search_subreward"](result, spec)
        self.assertEqual(reward, 1.0)
        self.assertEqual([hit["turn"] for hit in hits], [1])
        self.assertEqual(self.rollout["_training_reward"](1.0, reward), 2.0)
        self.assertTrue(self.rollout["_entity_mentioned"]("John", "John Smith"))
        self.assertFalse(self.rollout["_entity_mentioned"]("John", "Johnson"))

    def test_step_reward_is_subgoal_plus_final_with_no_point_three(self):
        compute = self.daemon["_gigpo_turn_step_reward"]
        hits_by_turn = {"1": 1.0}
        self.assertEqual(compute(hits_by_turn, 0, 1.0), 1.0)
        self.assertEqual(compute(hits_by_turn, 1, 1.0), 2.0)
        self.assertEqual(compute(hits_by_turn, 1, 0.0), 1.0)
        self.assertEqual(compute(hits_by_turn, 2, 0.0), 0.0)


class GiGPOAdvantageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # core_gigpo only needs the DataProto name at import time; the tested
        # pure advantage function does not construct it.
        verl = types.ModuleType("verl")
        verl.DataProto = object
        path = ROOT / "agentflow" / "verl" / "diy" / "trainer" / "ppo" / "core_gigpo.py"
        spec = importlib.util.spec_from_file_location("core_gigpo_reward_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"verl": verl}):
            spec.loader.exec_module(module)
        cls.core_gigpo = module

    def test_singleton_keeps_episode_advantage_and_answer_excludes_step(self):
        # Two rollouts of one question, each with analysis, tool and answer turns.
        # The tool anchors occur only once, while analysis anchors are paired.
        scores, _ = self.core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=torch.tensor([[1.0], [1.0], [1.0],
                                              [0.0], [0.0], [0.0]]),
            step_rewards=torch.tensor([1.0, 2.0, 1.0, 0.0, 0.0, 0.0]),
            response_mask=torch.ones(6, 1),
            anchor_obs=np.array(["analysis", "a_only", "a_answer",
                                 "analysis", "b_only", "b_answer"], dtype=object),
            index=np.array(["question"] * 6),
            traj_index=np.array(["a"] * 3 + ["b"] * 3),
            step_pair_mask=np.array([True, True, False, True, True, False]),
            compute_mean_std_cross_steps=False,
        )
        self.assertTrue(torch.allclose(
            scores[:, 0], torch.tensor([1.0, 0.5, 0.5, -1.0, -0.5, -0.5])
        ))


if __name__ == "__main__":
    unittest.main()
