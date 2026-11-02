"""Exact-prefix sharing, branch isolation and GT-logit alignment regressions."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "trace_prefix_scoring_under_test", ROOT / "agentflow/agentflow/trace/prefix_scoring.py")
prefix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prefix
SPEC.loader.exec_module(prefix)


def request(prompt, target):
    return {"prompt_ids": prompt, "target_ids": target}


def packs(rows, **kwargs):
    return prefix.build_prefix_packs(rows, max_tree_tokens=kwargs.pop("max_tree_tokens", 100),
                                     max_mask_bytes=kwargs.pop("max_mask_bytes", 40000),
                                     max_attention_cost_ratio=kwargs.pop("max_attention_cost_ratio", None), **kwargs)


def path(pack, node):
    ancestors = []
    while node >= 0:
        ancestors.append(node)
        node = pack.parents[node]
    return list(reversed(ancestors))


class PrefixTreePlanningTests(unittest.TestCase):
    def test_exact_token_paths_are_shared_across_requests_and_target_boundaries(self):
        rows = [request([1, 2], [3, 4]), request([1, 2, 3], [5]),
                request([1, 2], [3, 4]), request([1, 9], [8])]
        result, stats = packs(rows)
        self.assertEqual(len(result), 1)
        pack = result[0]
        self.assertEqual(stats["original_tokens"], 15)
        self.assertEqual(stats["tree_tokens"], 7)
        self.assertEqual(stats["saved_tokens"], 8)
        for position, original in enumerate(pack.request_indices):
            row = rows[original]
            indices = pack.target_parent_indices[position]
            for offset, parent in enumerate(indices):
                actual = [pack.input_ids[node] for node in path(pack, parent)]
                self.assertEqual(actual, row["prompt_ids"] + row["target_ids"][:offset])
                self.assertEqual(pack.position_ids[parent], len(actual) - 1)
        # The same full token path shares even when a prompt/target boundary
        # differs, since the causal model's hidden states are identical there.
        self.assertEqual(set(pack.request_indices), set(range(len(rows))))

    def test_mask_sees_only_own_ancestors_including_self(self):
        pack = packs([request([1, 2], [4, 5]), request([1, 3], [6, 7])])[0][0]
        tensors = prefix.build_tree_tensors(pack)
        mask = tensors["attention_mask"][0, 0]
        self.assertEqual(tensors["input_ids"].shape, (1, pack.tree_tokens))
        for node in range(pack.tree_tokens):
            visible = torch.where(mask[node] == 0)[0].tolist()
            self.assertEqual(visible, path(pack, node))
            self.assertTrue(torch.isneginf(mask[node, torch.tensor(
                [i for i in range(pack.tree_tokens) if i not in visible], dtype=torch.long)]).all())

    def test_token_and_mask_limits_split_without_losing_rows(self):
        rows = [request([1, i], [9]) for i in range(2, 8)]
        result, stats = packs(rows, max_tree_tokens=7, max_mask_bytes=100)
        self.assertGreater(len(result), 1)
        self.assertEqual(sorted(i for pack in result for i in pack.request_indices), list(range(len(rows))))
        self.assertTrue(all(pack.tree_tokens <= 5 for pack in result))
        self.assertLessEqual(stats["largest_mask_bytes"], 100)

    def test_oversize_single_request_never_truncates(self):
        with self.assertRaisesRegex(prefix.PrefixScoringError, "Request 0 needs 6 tokens"):
            packs([request([1, 2, 3, 4], [5, 6])], max_tree_tokens=5)

    def test_production_default_splits_unrelated_or_weakly_shared_requests(self):
        unrelated = [request([i] + list(range(10, 30)), [40, 41]) for i in (1, 2, 3, 4)]
        weakly_shared = [request([1, i] + list(range(10, 30)), [40, 41]) for i in (2, 3, 4, 5)]
        for rows in (unrelated, weakly_shared):
            with self.subTest(rows=rows):
                result, stats = prefix.build_prefix_packs(
                    rows, max_tree_tokens=200, max_mask_bytes=160000)
                self.assertEqual(len(result), len(rows))
                self.assertEqual(stats["original_attention_cells"],
                                 sum((len(r["prompt_ids"]) + len(r["target_ids"])) ** 2 for r in rows))
                self.assertEqual(stats["attention_cost_ratio"], 1.)
                for pack in result:
                    baseline = sum((len(rows[i]["prompt_ids"]) + len(rows[i]["target_ids"])) ** 2
                                   for i in pack.request_indices)
                    self.assertLessEqual(pack.tree_tokens ** 2, baseline)

    def test_production_default_keeps_strong_prefix_sharing_and_duplicate_requests(self):
        common = list(range(1, 33))
        rows = [request(common + [i], [50, 51]) for i in (40, 41, 42, 43)]
        rows.append(rows[0])
        result, stats = prefix.build_prefix_packs(rows, max_tree_tokens=200, max_mask_bytes=160000)
        self.assertEqual(len(result), 1)
        self.assertEqual(stats["tree_tokens"], 44)
        self.assertEqual(stats["original_attention_cells"], 5 * 35 ** 2)
        self.assertEqual(stats["tree_attention_cells"], 44 ** 2)
        self.assertLess(stats["attention_cost_ratio"], 1.)
        self.assertEqual(stats["saved_tokens"], 5 * 35 - 44)

    def test_attention_cost_ratio_validates_and_explicit_none_disables_only_that_guard(self):
        rows = [request([1, 2], [3]), request([4, 5], [6])]
        result, stats = packs(rows, max_attention_cost_ratio=None)
        self.assertEqual(len(result), 1)
        self.assertEqual(stats["attention_cost_ratio"], 2.)
        result, stats = packs(rows, max_attention_cost_ratio=2.)
        self.assertEqual(len(result), 1)
        result, stats = packs(rows, max_attention_cost_ratio=1.)
        self.assertEqual(len(result), 2)
        for ratio in (True, False, .9, 0, -1, float("nan"), float("inf"), "1"):
            with self.subTest(ratio=ratio), self.assertRaises(prefix.PrefixScoringError):
                packs(rows, max_attention_cost_ratio=ratio)
        self.assertEqual(packs([])[1]["attention_cost_ratio"], 0.)

    def test_production_default_obeys_cost_and_memory_limits_together(self):
        common = list(range(1, 31))
        rows = [request(common + [i], [90, 91]) for i in range(40, 50)]
        result, stats = prefix.build_prefix_packs(rows, max_tree_tokens=42, max_mask_bytes=4 * 40 ** 2)
        self.assertGreater(len(result), 1)
        self.assertEqual(sorted(i for pack in result for i in pack.request_indices), list(range(len(rows))))
        self.assertLessEqual(stats["largest_tree_tokens"], 40)
        self.assertLessEqual(stats["largest_mask_bytes"], 4 * 40 ** 2)
        self.assertLessEqual(stats["attention_cost_ratio"], 1.)

    def test_invalid_tokens_and_limits_fail_before_tensor_construction(self):
        for row in (request([], [1]), request([1], []), request([True], [2]),
                    request([1.5], [2]), request([-1], [2]), {}):
            with self.subTest(row=row), self.assertRaises(prefix.PrefixScoringError):
                packs([row])
        for value in (0, -1, 1.2, True):
            with self.subTest(value=value), self.assertRaises(prefix.PrefixScoringError):
                packs([], max_tree_tokens=value)
        self.assertEqual(packs([])[0], [])

    def test_first_gt_token_uses_prompt_parent_not_preceding_branch(self):
        pack = packs([request([10, 11], [20, 21]), request([10, 12], [30, 31])])[0][0]

        class PositionModel:
            training = False

            def __call__(self, input_ids, position_ids, attention_mask, use_cache):
                logits = torch.zeros(1, input_ids.shape[1], 64)
                for node in range(input_ids.shape[1]):
                    logits[0, node, :] = torch.arange(64) * (node + 1) / 10
                return SimpleNamespace(logits=logits)

        values = prefix.compute_tree_logps(PositionModel(), pack)
        for position, parents in enumerate(pack.target_parent_indices):
            for token_offset, parent in enumerate(parents):
                expected = (torch.arange(64) * (parent + 1) / 10).log_softmax(0)[pack.target_ids[position][token_offset]]
                self.assertEqual(values[position][token_offset], expected)

    def test_nonfinite_model_scores_remain_nonfinite_for_existing_validity_checks(self):
        pack = packs([request([1, 2], [3, 4])])[0][0]

        class InvalidModel:
            training = False

            def __call__(self, input_ids, **kwargs):
                return SimpleNamespace(logits=torch.full((1, input_ids.shape[1], 8), float("nan")))

        self.assertTrue(torch.isnan(prefix.compute_tree_logps(InvalidModel(), pack)[0]).all())


try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@unittest.skipUnless(HAS_TRANSFORMERS, "optional real-model CPU verification requires transformers")
class PrefixTreeRealModelTests(unittest.TestCase):
    def verify_model(self, model):
        rows = [request([3, 4, 5], [31, 32, 33]),
                request([3, 4, 6, 7], [34]),
                request([3, 4, 5, 31], [35, 36]),
                request([3, 4, 5], [31, 32, 33]),
                request([9, 10], [40, 41, 42, 43])]
        model.eval()
        result, stats = packs(rows)
        self.assertLess(stats["tree_tokens"], stats["original_tokens"])
        for pack in result:
            scored = prefix.compute_tree_logps(model, pack)
            kept = prefix.compute_tree_logps(model, pack, logits_to_keep=True)
            for local, original in enumerate(pack.request_indices):
                row = rows[original]
                ids = torch.tensor([row["prompt_ids"] + row["target_ids"]])
                with torch.no_grad():
                    logits = model(input_ids=ids, position_ids=torch.arange(ids.shape[1])[None],
                                   attention_mask=torch.ones_like(ids), use_cache=False).logits
                    start = len(row["prompt_ids"]) - 1
                    wanted = logits[0, start:start + len(row["target_ids"])].float().log_softmax(-1)
                    expected = wanted.gather(1, torch.tensor(row["target_ids"])[:, None]).squeeze(1)
                torch.testing.assert_close(scored[local], expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(kept[local], expected, rtol=1e-5, atol=1e-6)

    def test_qwen2_tree_matches_all_independent_gt_token_scores(self):
        torch.manual_seed(17)
        config = transformers.Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, attention_dropout=0.)
        config._attn_implementation = "eager"
        self.verify_model(transformers.Qwen2ForCausalLM(config))

    def test_qwen3_tree_matches_all_independent_gt_token_scores(self):
        torch.manual_seed(23)
        config = transformers.Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, attention_dropout=0.)
        config._attn_implementation = "eager"
        self.verify_model(transformers.Qwen3ForCausalLM(config))


if __name__ == "__main__":
    unittest.main()
