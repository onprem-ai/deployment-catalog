# 2026-05-06 — Initial Catalog Entry

**Goal:** Create catalog entry for Gemma 4 26B-A4B IT with MTP speculative decoding to achieve ~80 tok/s as demonstrated in Google benchmarks.

## Research Summary

Based on colleague's research via Claude, the 80 tok/s benchmark uses:

### Models Required (Single Catalog Entry with Two Sources)

| Component | Model | Notes |
|-----------|-------|-------|
| Target | `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` | FP8 quantized, fits single GPU |
| Drafter | `google/gemma-4-26B-A4B-it-assistant` | MTP drafter, ~500M params |

Both models are defined in a single catalog entry using the `sources:` list (plural).
Download triggers both HuggingFace downloads. Each downloads to its own directory.

### Performance Numbers (Google/Community Benchmarks)

| Configuration | tok/s | Speedup |
|---------------|-------|---------|
| FP8-it baseline (no MTP) | 40.85 | 1.0x |
| FP8-it + MTP γ=2 | 74.51 | 1.82x |
| FP8-it + MTP γ=4 | 108.78 | 2.66x |
| Concurrency=8, γ=4 | ~84/user | 2x per-user |

### Key Technical Details

1. **MoE Architecture**: 26B total params, only 3.8B active per forward pass
2. **Drafter compatibility**: MUST use `-it` (instruction-tuned) target. Using drafter with base model causes 38% slowdown
3. **Drafter weight sharing**: Drafter shares embeddings and KV cache with target, minimal memory overhead
4. **Image requirement**: MTP support requires `vllm/vllm-openai:gemma4-0505-cu130` or vLLM nightly (PR #41745)

### vLLM Command (from benchmarks)

```bash
vllm serve nvidia/Gemma-4-26B-A4B-NVFP4 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --gpu-memory-utilization 0.55 \
  --speculative-config '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}'
```

### Docker Setup

```bash
# For Blackwell GPUs (RTX PRO 6000, B200, B300)
docker pull vllm/vllm-openai:gemma4-0505-cu130

# For Hopper/Ampere (H100, A100)
docker pull vllm/vllm-openai:gemma4-0505-cu129
```

## Why This Works vs NVFP4

Our earlier test with Gemma 4 31B NVFP4 Turbo + MTP drafter failed because:

1. **NVFP4 strips MTP weights**: The quantization process removes MTP-specific weights
2. **Architecture mismatch**: BF16 drafter with NVFP4 target causes initialization failures

The FP8 Dynamic quantization from RedHat preserves compatibility with the MTP drafter.

## All Official Drafter Models

| Target Model | Drafter (Assistant) |
|--------------|---------------------|
| google/gemma-4-31B-it | google/gemma-4-31B-it-assistant |
| google/gemma-4-26B-A4B-it | google/gemma-4-26B-A4B-it-assistant |
| google/gemma-4-E4B-it | google/gemma-4-E4B-it-assistant |
| google/gemma-4-E2B-it | google/gemma-4-E2B-it-assistant |

## Outcome

Created catalog entry with:
- FP8 Dynamic target model from RedHat
- MTP speculative config with γ=4 (num_speculative_tokens=4)
- Special vLLM image with MTP support
- Memory profiles for 32k/128k/256k context

**Next steps:**
1. Download target model: `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`
2. Deploy and validate MTP works (drafter auto-downloads)
3. Benchmark to confirm ~80-108 tok/s decode speed
4. Binary search VRAM to find minimum working utilization

## Sources

- Google Blog: Multi-token prediction in Gemma 4
- Google AI Docs: Speed-up Gemma 4 with MTP
- ai-muninn.com: Gemma 4 MTP 108 tok/s on DGX Spark
- Millstone AI: Gemma-4-26B-A4B benchmarks on RTX PRO 6000
- vLLM PR #41745: MTP support for Gemma 4
