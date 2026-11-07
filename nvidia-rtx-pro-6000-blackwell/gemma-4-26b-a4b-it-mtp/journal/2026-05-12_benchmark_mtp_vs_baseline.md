# Benchmark: Gemma 4 26B-A4B IT — MTP vs Baseline

**Date:** 2026-05-12
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (96 GiB VRAM)
**vLLM Image:** `vllm/vllm-openai:gemma4-0505-cu130`
**Quantization:** FP8 Dynamic (RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic)
**MTP Drafter:** google/gemma-4-26B-A4B-it-assistant (~500M params, γ=4)
**Memory Profile:** 32k context, fp8 kv cache
**Concurrency:** 1 (single user)
**Tool:** llmapibenchmark (random input per request)

## Results

| Test | Metric | Baseline | MTP (γ=4) | Delta |
|------|--------|----------|-----------|-------|
| Decode (50w in, 500 tok out) | Gen TPS | 206.6 | 102.3 | -50% |
| Decode | Prefill TPS | 10,455 | 6,812 | -35% |
| Decode | TTFT | 20ms | 30ms | +50% |
| Prefill (1500w in, 100 tok out) | Gen TPS | 143.4 | 52.9 | -63% |
| Prefill | Prefill TPS | 28,325 | 7,752 | -73% |
| Prefill | TTFT | 190ms | 690ms | +263% |
| Balanced (500w in, 500 tok out) | Gen TPS | 168.1 | 82.2 | -51% |
| Balanced | Prefill TPS | 3,403 | 3,456 | ~0% |
| Balanced | TTFT | 530ms | 530ms | 0% |
| Long Context (3000w in, 100 tok out) | Gen TPS | 121.9 | 49.5 | -59% |
| Long Context | Prefill TPS | 39,722 | 16,340 | -59% |
| Long Context | TTFT | 270ms | 650ms | +141% |

## Analysis

**MTP speculative decoding is 50-63% SLOWER than baseline on this hardware.**

The baseline FP8 model achieves 207 tok/s decode — already 5x faster than Google's
reported baseline of ~40 tok/s, and 2x faster than their MTP γ=4 result (108 tok/s).

Possible explanations:
1. **Google benchmarked on RTX PRO 6000 (Ada), not Blackwell** — Blackwell's native
   FP8 tensor cores are fast enough that the drafter overhead exceeds the speculation
   savings
2. **Drafter verification overhead** — at 207 tok/s baseline, the verification step for
   4 speculated tokens costs more than just generating them directly
3. **MoE routing overhead** — the drafter shares embeddings but adds routing overhead
   for each speculated token verification

## Round 2: Corrected MTP Config

**Date:** 2026-05-12 (same session)

Research revealed the original config had critical issues:

1. **Wrong speculative-config format:** `"method": "mtp"` routes through generic MTP
   path. Gemma 4 has a dedicated `Gemma4MTPModel` / `Gemma4Proposer` code path in
   vLLM (PR #41745) that handles KV cache sharing and Q-only attention. Dropping
   `"method": "mtp"` lets vLLM auto-detect and use the correct Gemma4-specific path.

2. **`--attention-backend flashinfer` crashes** on `gemma4-0505-cu130` image — not
   available. Falls back to TRITON_ATTN. Community guides show flashinfer is critical
   for Blackwell performance but requires a different vLLM build.

3. **`num_speculative_tokens: 2` instead of 4** — community consensus is 2 is the
   sweet spot on vLLM. Additionally, the Gemma4 assistant model has only 1 MTP layer,
   so γ>1 means running that single layer multiple times (vLLM warns: "may result in
   lower acceptance rate").

4. **vLLM recommends TP=2 for 26B-A4B** — we only have 1 GPU, so TP=1 is forced.

### Corrected Config

```
--speculative-config '{"model": "/drafter", "num_speculative_tokens": 2}'
```
(no `"method": "mtp"`, no flashinfer, γ=2)

### Round 2 Results

| Test | Baseline | Old MTP (method:mtp, γ=4) | Corrected MTP (Gemma4MTP, γ=2) | Corrected vs Baseline |
|------|----------|--------------------------|-------------------------------|----------------------|
| Decode (50w→500tok) Gen TPS | 206.6 | 102.3 | 123.9 | -40% |
| Prefill (1500w→100tok) Gen TPS | 143.4 | 52.9 | 86.4 | -40% |
| Prefill Prefill TPS | 28,325 | 7,752 | 40,815 | +44% |
| Balanced (500w→500tok) Gen TPS | 168.1 | 82.2 | 105.8 | -37% |
| Balanced Prefill TPS | 3,403 | 3,456 | 35,500 | +10x |
| Long Context (3000w→100tok) Gen TPS | 121.9 | 49.5 | 54.3 | -55% |
| Long Context Prefill TPS | 39,722 | 16,340 | 16,506 | -58% |

### Analysis

Fixing the speculative-config format significantly improved results:
- **Gen speed improved ~20%** across all tests vs old MTP config
- **Prefill throughput recovered dramatically** — now exceeds baseline in some tests
  (40,815 vs 28,325 on Prefill test), confirming the old config was using a broken
  code path that crippled prefill

However, **generation speed is still 37-55% slower than baseline** in all tests.

Root causes for remaining gap:
1. **MoE expert routing overhead at batch=1** — verifying drafted tokens requires
   loading additional expert weights. Multiple sources (Google, community) confirm
   MoE needs concurrency 4-8 for MTP to break even. Our baseline TPOT is ~4.8ms
   (207 tok/s) — well below the ~20ms threshold where spec-dec typically helps.
2. **Single MTP layer** — the assistant model predicts 1 token ahead. γ=2 means
   running that layer twice, with diminishing returns. γ=1 may be more efficient
   but limits the speculation window.
3. **TP=1 forced** — vLLM recommends TP=2 for this model. Single GPU means both
   target and drafter compete for the same compute.
4. **No flashinfer** — the `gemma4-0505-cu130` image lacks flashinfer, which is
   critical for optimized Blackwell kernels per community benchmarks.

### Open Questions

- **31B Dense model may be a better MTP candidate** — dense models don't have MoE
  routing overhead during verification. The 31B has a slower baseline TPOT (~17ms
  on similar hardware), putting it in the range where spec-dec helps. Community
  results show 2.42x speedup with spec-dec on Gemma4 31B dense.
- **num_speculative_tokens=1** — untested, may be optimal given single MTP layer.
- **Newer vLLM image with flashinfer** — could unlock significant additional gains.

## Recommendation

**For 26B-A4B MoE on RTX PRO 6000 Blackwell at concurrency=1:** Use baseline (no MTP).
The 207 tok/s baseline already outperforms all MTP configs tested. MoE routing overhead
at batch=1 makes MTP counterproductive on this hardware.

**For higher concurrency (4-8):** MTP may break even or provide modest gains due to
expert weight overlap, but untested.

**For 31B Dense:** MTP is likely beneficial — slower baseline TPOT, no MoE routing
overhead, and community confirms 1.6-2.4x speedups on similar hardware.
