# vLLM 0.11 native token IDs for QA

## Confirmed cause

The user verified VERL 0.6.0, vLLM 0.11.0 and vLLM-Ascend 0.11.0.
A curl request through the existing AgentFlow proxy returned non-null
`prompt_token_ids` and `choices[i].token_ids` when `return_token_ids: true`
was present. The previous client neither requested this flag nor read the native
choice field. It incorrectly expected a top-level `response_token_ids` field.
The previous custom server also imported `AsyncvLLMServer`, which is absent in
VERL 0.6.0; that version uses `vLLMHttpServer` and native vLLM HTTP serving.

Upstream source checks:
- https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/serving_chat.py#L1379
- https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/serving_chat.py#L1422
- https://github.com/verl-project/verl/blob/v0.6.0/verl/workers/config/rollout.py#L64

## Minimal correction and rollback

The resulting tree uses main `bf300f9a5aa042d063c424fcfe289a329ef3ea90` as its
baseline, retaining only the QA exact-token data path and its tests/docs.
The commit is an ordinary single-parent commit on `fix`, not a merge or a force
reset. Main is not changed.

- `agentflow/agentflow/engine/vllm.py`: request `extra_body={"return_token_ids": True}`;
  read root prompt IDs and the first choice's `token_ids`. Keep ordinary text
  returns and internal `last_generation_metadata.response_token_ids`.
- `agentflow/agentflow/models/planner.py` and `train-roma/rollout.py`: preserve the
  existing exact-ID logging and Triplet construction. No re-tokenization fallback.
- `agentflow/verl/config.yaml`: set custom server path/name to null, matching
  VERL 0.6's Optional[str] schema/defaults.
- `agentflow/verl/async_server.py`: retain the main legacy code entirely as
  comments; no executable server imports or monkey patch remain in this module.
  Importing this retired module is harmless; it does not provide a replacement
  PatchedvLLMServer alias.
- `agentflow/instrumentation/vllm.py`: restore exactly to main, undoing the previous
  instance-capture framework. This legacy file is not used by the QA native server.
- `agentflow/scripts/check_serving_token_ids.py`: probe the native protocol and
  validate every returned choice.
- Replace the previous monkey-patch tests with native request-to-Triplet tests.

Replaced production statements are retained as `[原代码保留]` comments, with
`[修改目的]` explaining new behavior. Superseded speculative patch code/tests
remain available in Git history. GRPO, rewards, tools, memory, proxy forwarding,
sampling settings, and response length settings are unchanged.

## Tests and remaining runtime checks

Run CPU tests from repository root:

```sh
python -m unittest discover -s agentflow/tests -v
```

Eight tests cover the outgoing HTTP flag, native SDK fields and dump fallback,
exact IDs/EOS in Triplets, missing IDs, length termination without invented EOS,
metadata retention, retired-module import, native server configuration, and probe
validation of all choices. Generation/HTTP transport is simulated; these tests do
not establish success on real Ascend workers or a training run.

After updating the fix checkout actually imported by the runtime, restart the
workers. Against the user's proxy (replace port/model if they changed):

```sh
python agentflow/scripts/check_serving_token_ids.py \
  --base-url http://127.0.0.1:35571/v1 \
  --model Qwen3-4B-Instruct-2507
```

Expected: `PASS: OpenAI client received prompt_token_ids and response token ids`.
The response must contain root `prompt_token_ids` and each `choices[i].token_ids`,
not a synthetic root `response_token_ids` field.

Next run one QA rollout with `AGENTFLOW_TOKEN_DEBUG=1` in the agent worker's
environment. Confirm both internal ID lists are non-null and the Triplet response
IDs equal that request's `choices[0].token_ids`, including EOS if it was generated.
The debug flag prints token lists only when explicitly enabled. Do not infer full
RL success from the probe. Real endpoint and single-rollout verification remain
for the user's Ascend environment.
