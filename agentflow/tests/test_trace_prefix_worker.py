"""CPU contracts for reference-only tree scoring and collective schedules."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "trace_prefix_worker_test_package"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "agentflow/agentflow/trace")]
sys.modules[PACKAGE] = package
backend = importlib.import_module(PACKAGE + ".prefix_backend")
scoring = importlib.import_module(PACKAGE + ".prefix_scoring")
credit = importlib.import_module(PACKAGE + ".credit")


class Proto:
    def __init__(self, batch, non_tensors=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensors or {}
        self.meta_info = dict(meta_info or {})

    @classmethod
    def from_dict(cls, tensors, **kwargs):
        return cls(tensors, **kwargs)


class ToyCausalModel(torch.nn.Module):
    """A real masked-attention module, with deterministic reference logits."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(77)
            self.embedding = torch.nn.Embedding(100, 16)
            self.position = torch.nn.Embedding(128, 16)
            self.projection = torch.nn.Linear(16, 100, bias=False)
        self.config = SimpleNamespace(model_type="qwen3", _attn_implementation="flash_attention_2")
        self.tree_calls = 0

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False, logits_to_keep=None):
        hidden = self.embedding(input_ids) + self.position(position_ids)
        weights = hidden @ hidden.transpose(-1, -2) / 4
        if attention_mask.ndim == 4:
            self.tree_calls += 1
            assert self.config._attn_implementation == "eager"
            additive = attention_mask[:, 0]
        else:
            length = input_ids.shape[1]
            allowed = torch.ones(length, length, dtype=torch.bool).tril()[None] & attention_mask[:, None, :].bool()
            additive = torch.zeros_like(weights).masked_fill(~allowed, -torch.inf)
        hidden = (weights + additive.to(weights.dtype)).softmax(-1) @ hidden
        if logits_to_keep is not None:
            hidden = hidden[:, logits_to_keep]
        return SimpleNamespace(logits=self.projection(hidden))


class Worker:
    _is_ref = True
    _is_lora = False
    ulysses_sequence_parallel_size = 1

    def __init__(self, model=None):
        self.model = ToyCausalModel() if model is None else model
        self.model.eval()
        self.ref_policy = SimpleNamespace(actor_module=self.model, use_fused_kernels=False)
        self.config = SimpleNamespace(model={})
        self.native_calls = 0
        self.inject_native_difference = False

    def compute_ref_log_prob(self, data):
        self.native_calls += 1
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=data.batch["input_ids"],
                             attention_mask=data.batch["attention_mask"],
                             position_ids=data.batch["position_ids"], use_cache=False).logits
            width = data.batch["responses"].shape[1]
            logits = out[:, -width - 1:-1].float().log_softmax(-1)
            scores = logits.gather(-1, data.batch["responses"].unsqueeze(-1)).squeeze(-1)
        if self.inject_native_difference:
            scores = scores + .25
        return Proto({"ref_log_prob": scores.to(getattr(self, "output_dtype", scores.dtype))})


def batch(*, verify=True):
    requests = [
        {"prompt_ids": [3, 4, 5, 6, 7, 8], "target_ids": [31, 32]},
        {"prompt_ids": [3, 4, 5, 6, 7, 9], "target_ids": [33, 34, 35]},
        {"prompt_ids": [3, 4, 5, 6, 7, 10], "target_ids": [36]},
    ]
    return Proto(credit.build_score_tensors(requests, 0), meta_info={
        "trace_gt_scoring": True, "temperature": 1., "trace_prefix_verify": verify,
        "trace_prefix_max_tokens": 64, "trace_prefix_max_mask_bytes": 64 ** 2 * 4,
    })


def compute(worker, data, **kwargs):
    return backend.compute_reference_prefix(worker, data, Proto, device_name="cpu", device_id="cpu", **kwargs)


class Consensus:
    """Model a peer rank's agreed reductions without starting an NPU runtime."""
    def __init__(self, *, rounds=None, remote_unsupported=False, fail_validation=False):
        self.max_calls = 0
        self.rounds = rounds
        self.remote_unsupported = remote_unsupported
        self.fail_validation = fail_validation

    def integer(self, value, operation):
        if operation == "MAX":
            self.max_calls += 1
            if self.max_calls == 2 and self.remote_unsupported:
                return 1
            if self.max_calls == 7 and self.rounds is not None:
                return max(int(value), self.rounds)
        if operation == "MIN" and self.fail_validation:
            return 0
        return int(value)

    def maximum(self, value):
        return float(value)


class PrefixWorkerTests(unittest.TestCase):
    def test_tree_scores_match_native_tokens_and_only_first_call_runs_verification(self):
        worker, data = Worker(), batch()
        expected = worker.compute_ref_log_prob(data).batch["ref_log_prob"]
        worker.native_calls = 0
        output = compute(worker, data)
        mask = data.batch["response_mask"].bool()
        torch.testing.assert_close(output.batch["ref_log_prob"][mask], expected[mask], rtol=1e-5, atol=1e-6)
        metrics = output.meta_info["trace_prefix_metrics"]
        self.assertEqual(metrics["trace/prefix_used"], 1)
        self.assertEqual(metrics["trace/prefix_validation_passed"], 1)
        self.assertGreater(metrics["trace/prefix_saved_tokens"], 0)
        self.assertEqual(worker.native_calls, 1)
        self.assertEqual(worker.model.config._attn_implementation, "flash_attention_2")
        again = compute(worker, data)
        self.assertEqual(again.meta_info["trace_prefix_metrics"]["trace/prefix_used"], 1)
        self.assertEqual(worker.native_calls, 1)
        self.assertEqual(again.batch["ref_log_prob"].dtype, expected.dtype)

    def test_peer_pack_count_adds_dummy_forwards_without_extra_scores(self):
        worker, data = Worker(), batch(verify=False)
        actual = compute(worker, data, consensus=Consensus(rounds=4))
        self.assertEqual(worker.model.tree_calls, 4)
        self.assertEqual(actual.batch["ref_log_prob"].shape, data.batch["responses"].shape)
        metrics = actual.meta_info["trace_prefix_metrics"]
        self.assertEqual(metrics["trace/prefix_forward_rounds"], 4)
        self.assertEqual(metrics["trace/prefix_dummy_forwards"], 4 - metrics["trace/prefix_packs"])

    def test_remote_unsupported_rank_falls_back_before_any_tree_forward(self):
        worker, data = Worker(), batch()
        output = compute(worker, data, consensus=Consensus(remote_unsupported=True))
        self.assertEqual(worker.model.tree_calls, 0)
        self.assertEqual(worker.native_calls, 1)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_unsupported_model"], 1)

    def test_memory_limit_falls_back_without_truncating_or_changing_inputs(self):
        worker, data = Worker(), batch()
        before = {key: tensor.clone() for key, tensor in data.batch.items()}
        data.meta_info["trace_prefix_max_mask_bytes"] = 4
        output = compute(worker, data)
        self.assertEqual(worker.model.tree_calls, 0)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_planning"], 1)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, data.batch[key]))

    def test_validation_failure_returns_native_and_disables_following_tree_calls(self):
        worker, data = Worker(), batch()
        worker.inject_native_difference = True
        expected = worker.compute_ref_log_prob(data).batch["ref_log_prob"]
        worker.native_calls = 0
        output = compute(worker, data)
        self.assertTrue(torch.equal(output.batch["ref_log_prob"], expected))
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_validation"], 1)
        self.assertTrue(worker._trace_prefix_disabled)
        tree_calls = worker.model.tree_calls
        again = compute(worker, data)
        self.assertEqual(worker.model.tree_calls, tree_calls)
        self.assertEqual(again.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_validation_disabled"], 1)

    def test_remote_validation_failure_also_disables_this_rank(self):
        worker, data = Worker(), batch()
        output = compute(worker, data, consensus=Consensus(fail_validation=True))
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_validation_passed"], 0)
        self.assertTrue(worker._trace_prefix_disabled)

    def test_native_output_precision_is_preserved_after_verification(self):
        worker, data = Worker(), batch()
        worker.output_dtype = torch.bfloat16
        output = compute(worker, data)
        self.assertEqual(output.batch["ref_log_prob"].dtype, torch.bfloat16)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_validation_passed"], 1)
        second = compute(worker, data)
        self.assertEqual(second.batch["ref_log_prob"].dtype, torch.bfloat16)
        self.assertEqual(worker.native_calls, 1)

    def test_no_shared_nodes_uses_original_backend(self):
        data = Proto(credit.build_score_tensors([
            {"prompt_ids": [1, 2, 3], "target_ids": [31]},
            {"prompt_ids": [4, 5, 6], "target_ids": [32]},
        ], 0), meta_info={"trace_gt_scoring": True})
        worker = Worker()
        output = compute(worker, data)
        self.assertEqual(worker.model.tree_calls, 0)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_fallback_no_sharing"], 1)
        self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_used"], 0)

    def test_unsupported_lora_fused_window_and_rope_do_not_enter_tree(self):
        for setting in ("lora", "fused", "window", "dynamic_rope"):
            with self.subTest(setting=setting):
                worker, data = Worker(), batch()
                if setting == "lora":
                    worker._is_lora = True
                elif setting == "fused":
                    worker.ref_policy.use_fused_kernels = True
                elif setting == "window":
                    worker.model.config.use_sliding_window = True
                else:
                    worker.model.config.rope_scaling = {"rope_type": "dynamic"}
                output = compute(worker, data)
                self.assertEqual(worker.model.tree_calls, 0)
                self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_used"], 0)

    def test_eager_scope_restores_config_and_reshards_when_model_forward_raises(self):
        worker, data = Worker(), batch(verify=False)
        original = worker.model.forward
        def fail(*args, **kwargs):
            if kwargs["attention_mask"].ndim == 4:
                raise RuntimeError("model failure")
            return original(*args, **kwargs)
        worker.model.forward = fail
        resharded = []
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            compute(worker, data, reshard=lambda model: resharded.append(model))
        self.assertEqual(worker.model.config._attn_implementation, "flash_attention_2")
        self.assertEqual(resharded, [worker.model])
        self.assertEqual(worker.native_calls, 0)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "optional real HF model validation")
class RealQwenPrefixWorkerTests(unittest.TestCase):
    def test_qwen2_and_qwen3_match_independent_teacher_forcing(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM
        for config_class, model_class in ((Qwen2Config, Qwen2ForCausalLM), (Qwen3Config, Qwen3ForCausalLM)):
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(81)
                config = config_class(vocab_size=100, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      max_position_embeddings=128, attention_dropout=0.)
                config._attn_implementation = "eager"
                model = model_class(config).eval()
                worker, data = Worker(model), batch()
                output = compute(worker, data)
                self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_used"], 1)
                self.assertEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_validation_passed"], 1)
                self.assertLessEqual(output.meta_info["trace_prefix_metrics"]["trace/prefix_validation_max_diff"], 1e-5)
                self.assertEqual(model.config._attn_implementation, "eager")


if __name__ == "__main__":
    unittest.main()
