# 2026-05-06 — Initial Catalog Entry

**Goal:** Create catalog entry for Gemma 4 31B IT Turbo with speculative decoding using Google's official assistant model.

## Research

### Assistant Model: google/gemma-4-31B-it-assistant

The "31B-it-assistant" is NOT the full 31B model — it's a tiny ~500M parameter drafter designed for speculative decoding:

| Attribute | Value |
|-----------|-------|
| Parameters | ~500M (not 31B!) |
| Layers | 4 |
| Attention heads | 32 |
| KV heads | 16 |
| Hidden size | 1024 |
| Context length | 256K tokens |
| Model size | 939 MB (single safetensors file) |
| License | Apache 2.0 |

The model uses mixed attention: 3 sliding window layers + 1 full attention layer.

### vLLM Speculative Decoding Configuration

From vLLM docs (2026), speculative decoding is configured via `--speculative-config`:

```bash
--speculative-config '{"method": "draft_model", "model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 5}'
```

Key parameters:
- `method`: "draft_model" for using a separate smaller model
- `model`: HuggingFace repo name (vLLM downloads automatically)
- `num_speculative_tokens`: Number of tokens to speculate ahead (5 is typical)

### Expected Performance

Speculative decoding is lossless — output quality identical to base model. Expected speedup:
- Decode throughput: ~1.5-2x (51 tok/s → ~80-100 tok/s)
- Prefill unchanged (spec decode only helps decode phase)
- Slight increase in first-token latency (loading 2 models)

## Implementation

Copied `gemma-4-31b-it-nvfp4-turbo` as base and added:

1. New `model_id`: `gemma-4-31b-it-turbo-spec`
2. Added `--speculative-config` to vLLM args
3. Increased `weights_gi` from 18 to 19 (assistant adds ~1 GiB)
4. Added 2 GiB buffer to all memory profiles for assistant overhead
5. Set maturity to `experimental` (needs validation)
6. Updated display name/description

### Memory Estimates

| Profile | Turbo (measured) | Spec (estimated) |
|---------|------------------|------------------|
| 128k context | 32 GiB | 34 GiB |
| 64k context | 30 GiB | 32 GiB |
| 16k context | 27 GiB | 29 GiB |

Note: These are estimates based on turbo + 2 GiB buffer. Need to validate with actual deployment and binary search.

## Open Questions

1. **Download behavior**: vLLM will download the assistant model from HuggingFace on first run. This adds startup latency but avoids complexity of pre-downloading.

2. **num_speculative_tokens**: Started with 5 (typical value). May need tuning — higher values increase speculation but also increase rejection rate.

3. **Compatibility**: Need to verify the NVFP4 turbo model works with the BF16 assistant model. Architecture alignment should be fine but quantization differences may cause issues.

## Outcome

Created catalog entry `gemma-4-31b-it-turbo-spec`.

**Next steps:**
1. Download main model (if not already present)
2. Deploy and validate speculative decoding works
3. Binary search for minimum working `gpu_memory_utilization`
4. Benchmark actual decode throughput
5. Update memory profiles and performance metrics with measured values

## Sources

- [vLLM Speculative Decoding Docs](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [vLLM Speculators Library](https://github.com/vllm-project/speculators)
- [HuggingFace: google/gemma-4-31B-it-assistant](https://huggingface.co/google/gemma-4-31B-it-assistant)
