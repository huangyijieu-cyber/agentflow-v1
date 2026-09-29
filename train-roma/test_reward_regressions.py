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
             "compute_search_subreward", "build_gigpo_anchors", "build_turn_process_rewards"},
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
            {"_gigpo_return_to_go", "_training_token_scores"},
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

    def test_subreward_only_uses_model_visible_result_prefix(self):
        spec = {"subgoals": [{"id": "person", "answer": "John Smith", "weight": 1.0}]}
        action = {"tool_name": "Wikipedia_Search_Tool", "result": "x" * 2000 + " John Smith"}
        result = {"memory": {"Action Step 1": action}}
        self.assertEqual(self.rollout["compute_search_subreward"](result, spec), (0.0, []))

        action["result"] = "x" * 1988 + " John Smith"
        reward, hits = self.rollout["compute_search_subreward"](result, spec)
        self.assertEqual(reward, 1.0)
        self.assertEqual(hits[0]["turn"], 1)

        # Query text remains eligible, as requested, even when no page is returned.
        action["result"] = {"query": "John Smith spouse", "relevant_pages": []}
        self.assertEqual(self.rollout["compute_search_subreward"](result, spec)[0], 1.0)

    def test_subgoal_and_terminal_rewards_propagate_by_half_per_turn(self):
        hits = [{"turn": 2, "subgoal_id": "a", "weight": 0.3}]
        turn_rewards = self.rollout["build_turn_process_rewards"](hits)
        self.assertEqual(turn_rewards, {"2": 1.0})
        compute = self.daemon["_gigpo_return_to_go"]
        self.assertEqual(compute(turn_rewards, 4, 0.0), [0.25, 0.5, 1.0, 0.0])
        self.assertEqual(compute({}, 4, 1.0), [0.125, 0.25, 0.5, 1.0])
        self.assertEqual(compute(turn_rewards, 4, 1.0), [0.375, 0.75, 1.5, 1.0])

    def test_active_subgoal_hit_is_one_point_even_with_fractional_source_weight(self):
        result = {"memory": {"Action Step 1": {
            "tool_name": "Wikipedia_Search_Tool", "result": "John Smith",
        }}}
        spec = {"subgoals": [{"id": "a", "answer": "John Smith", "weight": 0.3}]}
        score, hits = self.rollout["compute_search_subreward"](result, spec)
        self.assertEqual(score, 1.0)
        self.assertEqual(hits[0]["weight"], 1.0)

        spec["subgoals"][0]["weight"] = 0.0
        self.assertEqual(self.rollout["compute_search_subreward"](result, spec), (0.0, []))

    def test_anchor_visit_pairs_equal_progress_from_different_rollouts(self):
        make_anchors = self.rollout["build_gigpo_anchors"]
        a, a_mask = make_anchors([{"turn": 2, "subgoal_id": "f1"}], 6, "a")
        b, b_mask = make_anchors([{"turn": 3, "subgoal_id": "f1"}], 7, "b")
        self.assertNotEqual(a[1], a[2])  # same rollout, different visits to empty anchor
        self.assertEqual(a[3], b[4])  # first visit after finding f1
        self.assertEqual(a[4], b[5])  # second visit after finding f1
        self.assertFalse(a_mask[-1])
        self.assertFalse(b_mask[-1])

    def test_gigpo_critic_token_scores_use_only_final_reward(self):
        choose = self.daemon["_training_token_scores"]
        self.assertEqual(choose([1.0, 0.0], [2.0, 1.0], "gigpo"), [1.0, 0.0])
        self.assertEqual(choose([1.0, 0.0], [2.0, 1.0], "grpo"), [2.0, 1.0])


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


class RewardMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "agentflow" / "verl" / "reward_metrics.py"
        spec = importlib.util.spec_from_file_location("agentflow_reward_metrics_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.metrics = module

    def test_rollout_metrics_use_eligible_denominators(self):
        annotated = {
            "final_reward": 1.0, "subreward": 1.0, "n_subgoals": 2,
            "answer_tag_valid": True,
            "subgoal_hits": [{"subgoal_id": "a"}],
            "turn_process_rewards": {"1": 1.0},
            "search_turn_indices": [1, 2],
        }
        unannotated = {"final_reward": 0.0, "n_subgoals": 0, "answer_tag_valid": False}
        result = self.metrics.summarize_reward_metrics([annotated, unannotated], "val")
        self.assertEqual(result["val/final_correct_rate"], 0.5)
        self.assertEqual(result["val/reward_spec_coverage"], 0.5)
        self.assertEqual(result["val/subreward_activation_rate"], 1.0)
        self.assertEqual(result["val/search_turn_hit_rate"], 0.5)
        self.assertEqual(result["val/subgoal_coverage"], 0.5)
        self.assertEqual(result["val/search_turn_hit_rate/turn_1"], 1.0)
        self.assertEqual(result["val/search_turn_hit_rate/turn_2"], 0.0)

        no_specs = self.metrics.summarize_reward_metrics([unannotated], "val")
        self.assertEqual(no_specs["val/reward_spec_coverage"], 0.0)
        self.assertNotIn("val/subreward_activation_rate", no_specs)

        wrong_annotated = {
            "final_reward": 0.0, "subreward": 0.5, "n_subgoals": 1,
            "subgoal_hits": [{"subgoal_id": "b"}],
            "turn_process_rewards": {"2": 0.5}, "search_turn_indices": [2],
        }
        train_result = self.metrics.summarize_reward_metrics(
            [annotated, wrong_annotated], "train"
        )
        self.assertEqual(train_result["train/subreward_mean_when_final_wrong"], 0.5)

    def test_gigpo_diagnostics_require_shared_anchor_and_reward_difference(self):
        result = self.metrics.summarize_gigpo_groups(
            ["q", "q", "q"], ["tool", "tool", "answer"],
            [True, True, False], [0.25, 0.75, 0.25],
        )
        self.assertEqual(result["gigpo/paired_turn_rate"], 1.0)
        self.assertEqual(result["gigpo/nonzero_step_adv_rate"], 1.0)

        one_empty_response = self.metrics.summarize_gigpo_groups(
            ["q", "q"], ["tool", "tool"], [True, True],
            [0.25, 0.75], [True, False],
        )
        self.assertEqual(one_empty_response["gigpo/eligible_turn_count"], 1.0)
        self.assertEqual(one_empty_response["gigpo/nonzero_step_adv_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
