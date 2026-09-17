# Exact serving token IDs: QA rollout

This change targets the non-streaming text QA endpoint. It does not change GRPO,
reward, tool, memory, Planner, or rollout encoding. Original replaced production
code is retained as comments marked `[原代码保留]`; new behavior is marked
`[修改目的]`. There is no text re-tokenization fallback.

## Evidence and limits

The reported normal text response with `prompt_token_ids=None` and no
`response_token_ids` establishes that exact IDs did not reach the client. It does
not alone establish whether the wrapper was bypassed (A), the generation output
lacked IDs (B), or serialization dropped fields (C). The actual Ascend runtime was
not accessible during this repair. Repository requirements pin vLLM 0.8.5 but do
not establish the installed VERL/vLLM-Ascend versions. No specific runtime root
cause or successful 910B training run is claimed.

The previous patch targeted one imported class and used a protocol-class identity
check as an installation sentinel. That cannot verify the method on an actual
serving instance with an override or an already-bound method. Replacing the
protocol module's response class also does not replace class aliases imported
elsewhere. Whether aliases cause field loss depends on the actual Pydantic model;
in particular, an `extra=allow` model may already preserve fields.

## Changes

- Bind the wrapper to `self.openai_serving_chat` immediately before non-streaming
  generation, after the actual object exists. Verify the method itself for
  idempotence, and bind its signature instead of copying version-specific args.
- Copy `RequestOutput.prompt_token_ids` and each indexed output's `token_ids`
  before yielding. Align sequences to the returned choice indexes.
- Extend the actual response model with declared `List[int]` and `List[List[int]]`
  fields; explicitly include them in the final HTTP JSON. No global response class
  alias replacement is needed.
- Refuse missing/invalid IDs with a serving error instead of a successful HTTP
  response that fails only later during Triplet creation.

The wrapper supports the cumulative/final `RequestOutput` contract used by the
full non-streaming generator. Unknown method signatures, output schemas, or
non-cumulative sequences fail explicitly; no guessed alternative token field is
substituted. Such failures require adapting against the installed source.

## Runtime verification

1. Update the **fix** checkout and ensure the installed `agentflow` package uses
   that checkout. Restart all serving actors; editing disk alone does not update
   methods already loaded in workers.
2. Set `AGENTFLOW_TOKEN_DEBUG=1` in the environment inherited by serving/Ray actors
   before launching them. Logs show package versions, actual class, method,
   signature/source path, wrapper entry, field names and token lengths. They do
   not log prompts, answers or complete token sequences.
3. Against the actual **model serving** endpoint (not AgentFlow's task queue), run:

   ```sh
   python agentflow/scripts/check_serving_token_ids.py --base-url http://HOST:PORT/v1 --model MODEL
   ```

   Use the served model name and port from the runtime. If authenticated, provide
   `OPENAI_API_KEY` through the environment. The probe disables client retries to
   expose the first failure, validates both fields through the OpenAI client, and
   prints only token counts/prefixes. A length-limited answer need not contain EOS;
   the fix never fabricates one.
4. Run one QA rollout and confirm Planner IDs are non-null and Triplet construction
   succeeds before restarting a full training run.

Interpretation: no `[TOKEN PATCH]` implies this request path has not installed the
patch (or logs are from another process). Patch log without `wrapper entered` on a
completion implies a bypassed method; `[TOKEN B]` and output field diagnostics
identify absent/unsupported raw generation data. `wrapper entered` and valid
counts followed by boundary failure localizes the response/serialization path.
`[TOKEN A/C]` intentionally does not claim to distinguish bypass from field loss
without those preceding logs.

## Local checks

```sh
python -m unittest discover -s agentflow/tests -v
```

These CPU contract tests stub vLLM/VERL generation. They cover response models
that ignore unknown fields, bound methods, subclasses, cumulative outputs,
multiple choices, early iterator exit, concurrency, error responses, missing
IDs, and the endpoint body. They are not a replacement for the runtime probe.
