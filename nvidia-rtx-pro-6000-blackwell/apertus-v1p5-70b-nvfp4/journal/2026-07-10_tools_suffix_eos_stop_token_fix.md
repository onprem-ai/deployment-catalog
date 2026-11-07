# 2026-07-10 -- Fix: <|tools_suffix|> (72) missing from eos_token_id -- generation ran past tool calls

## Symptom

Deterministic at temp 0 (pre-fix probe via POST /api/inference-services/test/chat):
the model emits a correct tool call, then CONTINUES generating -- hallucinated
answer text, raw `<|tools_prefix|>` leaking into `content`, repeated blocks,
`stop_reason: null`. Same family as the "fails to stop after the call"
observation in 2026-07-08_agentic_capability_finding.md (there attributed to
sampling variance; it is actually deterministic with tools attached).

## Root cause

The v1.5 prerelease upstream `generation_config.json` declares
`"eos_token_id": 68` only (68 = `<|assistant_end|>`). The 2509 Instruct
release ships `[2, 68, 72]`, where 72 = `<|tools_suffix|>`. Token 72 exists in
the v1.5 tokenizer (verified in tokenizer.json added_tokens) -- the model is
trained to emit it after a tool-call array, but vLLM was never told it is a
stop token, so generation continued after every tool call.

vLLM mechanism (source-verified via vllm-project/vllm#45591 discussion):
`v1/engine/input_processor.py` loads the model dir's `generation_config.json`
at startup and registers list-valued `eos_token_id`s as per-request stop
tokens via `SamplingParams.update_from_generation_config`. Non-primary EOS ids
become `stop_token_ids`, and a stop on one reports `stop_reason: <token_id>`.

## Rejected alternatives

- `--override-generation-config '{"eos_token_id": [...]}'` in the card: does
  NOT work. The override is only consulted by `get_diff_sampling_param()`,
  which filters through a hardcoded sampling-param allow-list; `eos_token_id`
  is silently dropped (confirmed on v0.23.0 and main in vllm#45591).
- Injecting `stop_token_ids` per model at the ai-gateway: feasible (the proxy
  already rewrites bodies) but puts model-specific serving logic in the proxy
  and misses pod-direct traffic. `stop` STRINGS would be actively wrong:
  vLLM strips matched stop strings before the tool parser runs -- the same
  failure mode the 8B SGLang detector patch fixed
  (apertus-v15-8b-prerelease/journal/2026-06-19_tool_calling_fix.md).
- Client-side stop params (agent config): only covers agent traffic.

## Fix

Set `"eos_token_id": [2, 68, 72]` (mirroring the 2509 Instruct release) in
`generation_config.json` of all three local quantized checkpoints
(ThreeTier, FP8-Dynamic, NVFP4-MLPOnly under /persistent/models/, `.orig`
backups kept next to each), and added `write_tool_call_eos_token_ids()` to
`quantize_apertus_70b_three_tier.py` so both pass outputs carry the fix on
future quantization runs (model_free_ptq copies the source config verbatim).

NOTE: the card's `source.files` list includes `generation_config.json` from
the onprem-ai HF repo -- a re-download would revert the local patch until the
fixed config is uploaded there (open item below).

## Verification

Restarted via POST /api/inference-services/restart (pod restart suffices --
the model dir is a hostPath mount; vLLM reads generation_config.json at
startup). All probes via POST /api/inference-services/test/chat.

Pre-fix baseline (temp 0, get_weather tool): `finish_reason: tool_calls` but
`stop_reason: null`, content = hallucinated weather answer with raw
`<|tools_prefix|>` leaked and the answer repeated. Deterministic.

Post-fix:

| Probe | Result |
|-------|--------|
| Tool call, temp 0 (same as baseline) | finish=tool_calls, **stop_reason: 72**, content empty, 1 call |
| Plain chat, no tools, temp 0 | finish=stop, clean one-sentence answer (no regression from adding 2/72) |
| Tool call, temp 0.7, 3 samples | all: finish=tool_calls, stop_reason 72, content_len 0, 1 call |

`stop_reason: 72` confirms the suffix token is registered as a stop token and
the native apertus parser tolerates the stripped suffix (as expected -- the
upstream 2509 release ships this exact config + parser combination).

## Open items

1. Upload the fixed generation_config.json to the onprem-ai HF repo(s) so
   re-downloads keep the fix.
2. Upstream: report that the Prerelease-2607
   generation_config lacks 72 vs the 2509 release; likely a
   packaging gap that the final v1.5 Instruct should fix.
3. Retest the heavier agentic failure (prompt-example parroting) from
   2026-07-08_agentic_capability_finding.md -- the stop token cannot fix that
   one (no stop config can force a token the model never emits), but the
   milder "continues past the call" half should now be gone.
