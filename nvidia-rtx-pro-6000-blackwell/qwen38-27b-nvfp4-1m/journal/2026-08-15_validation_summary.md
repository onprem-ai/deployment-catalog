# 2026-08-15 - Validation Summary

**Goal:** Consolidate the final state of the 1M deployment after capability
validation and the official performance benchmark suite.

## Running Configuration

The InferenceService is Ready and serves `qwen3.8-27b` with a 1,010,000-token
context, FP8 KV cache, and MTP speculative decoding:

```text
--max-model-len 1010000
--hf-overrides '{"text_config": {"max_position_embeddings": 1010000}}'
--kv-cache-dtype fp8
--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

MTP is therefore active. vLLM loads the checkpoint's bundled
`model_mtp.safetensors` head and proposes two speculative tokens per decode
step. Startup logs confirmed `SpeculativeConfig(method='mtp', model='/model',
num_spec_tokens=2)`.

## Capability Summary

The official capability-test API produced:

| Suite | Result |
|---|---|
| Vision | 8/8 passed |
| Tools | 12/12 passed |
| Vision + tools | 4/4 passed |
| Chat | 6/6 passed on retry |

The first chat run had one transient empty non-streaming developer-message
response; the immediate complete retry passed. The integrity suite could not
run because the digest-pinned Job image contains an older runner that rejects
the API's `integrity` test type.

## Performance Summary

The official benchmark suite measured 129.21 tok/s single-user decode using
50 random words and 500 output tokens. Balanced generation measured 118.16
tok/s, the 3000-word long-input test generated at 50.50 tok/s, and aggregate
generation reached 950.73 tok/s at concurrency 16 with a 100% success rate.

The benchmark client's prefill/TTFT arithmetic returned invalid values for the
prefill and long-input jobs, so no `tps_prefill` was added to the catalog. The
valid 129.21 tok/s decode result is recorded in `inference.yaml`.

## Outcome

The 1M model remains deployed, Ready, and error-free with MTP enabled. Full
results and caveats remain in the dedicated capability and performance journal
entries from this session.
