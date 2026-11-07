# Improvement Proposals — Gemma 4 31B IT NVFP4 SGLang MTP

**Date**: 2026-05-14
**Baseline**: 90 tok/s decode, 4,719 tok/s prefill, 32k/128k context profiles
**Hardware**: RTX PRO 6000 Blackwell (96 GiB VRAM, SM100)

---

## 1. trtllm_mha Attention Backend (HIGH PRIORITY)

**Source**: SGLang PR #25006 — "Enable trtllm_mha as gemma4 default attn backend on SM100"

TensorRT-LLM's fused MHA kernel is optimized for Blackwell (SM100). Benchmarks from the PR show:
- +22% output tok/s (text generation)
- +13.9% overall throughput
- Requires flashinfer v0.6.10.post1+

**How to test**:
```
--attention-backend trtllm_mha
```

**Expected impact**: Decode could go from 90 → ~110 tok/s. Prefill may also improve since the attention kernel is used in both phases.

**Risk**: Low — the PR targets Gemma 4 specifically on SM100. Main concern is compatibility with NEXTN MTP (spec decode uses its own attention path for the drafter). Needs verification that drafter doesn't break with trtllm_mha on the main model.

**Availability**: Merged into SGLang main. NOT in our current image (`lmsysorg/sglang:gemma4-mtp` is v0.5.12.dev93, pre-dates this PR). Requires rebuilding from a newer SGLang commit or waiting for v0.5.12+ stable.

---

## 2. FlashInfer Prefill for Blackwell (MEDIUM PRIORITY)

**Source**: SGLang PR #22921 — FlashInfer prefill support for Blackwell

- 5% overall speedup
- 19% speedup on GDN (Gated Dense Network) layers specifically
- Uses `--linear-attn-prefill-backend flashinfer`

**Expected impact**: Prefill could improve from 4,719 → ~5,000-5,600 tok/s. Stacks with trtllm_mha (they target different parts of the forward pass).

**Risk**: Low. Already merged. Same availability constraint as #1 — needs newer SGLang build.

---

## 3. Piecewise CUDA Graphs with Speculative Decoding (MEDIUM PRIORITY)

**Source**: SGLang PR #22128 — "Allow piecewise CUDA graph with speculative decoding"

Our current deployment logs show: `Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set`. The `lmsysorg/sglang:gemma4-mtp` image disables them by default for FROZEN_KV_MTP.

PR #22128 adds support for piecewise CUDA graphs WITH speculative decoding. This could reduce kernel launch overhead during both prefill and decode.

**How to test**: Remove `--disable-piecewise-cuda-graph` (or don't set it) on a newer build that includes PR #22128.

**Expected impact**: Hard to quantify without testing. CUDA graph capture already takes ~28s in our setup (24s main + 4s drafter). Piecewise graphs may add capture time but reduce per-step latency.

**Risk**: Medium — piecewise graphs with spec decode is newer code. Could cause correctness issues or VRAM spikes during graph capture. Needs careful testing with all capabilities (vision, tool calling, reasoning).

---

## 4. Chunked Prefill Size Tuning (LOW PRIORITY)

**Source**: SGLang discussion #2815, PR #2268

SGLang's default `--chunked-prefill-size` is 8192. Larger values (16384, 32768) can improve prefill throughput by reducing the number of prefill chunks and associated overhead.

**How to test**:
```
--chunked-prefill-size 16384
--chunked-prefill-size 32768
```

**Expected impact**: Marginal (5-15% prefill improvement). The turbo entry gets 12,448 tok/s prefill partly because `--max-num-batched-tokens 4096` + `--enforce-eager` avoids CUDA graph overhead. We can't use `--enforce-eager` because it kills decode throughput with MTP.

**Risk**: Low. Larger chunk size uses more temporary VRAM during prefill, which may reduce `max_total_num_tokens` slightly. Re-measure VRAM after changing.

---

## 5. 256K Context Profile (BLOCKED — NEEDS TURBOQUANT)

The model natively supports 256K context. At 128k with FP8 KV, min_gi=79 (85.1 GiB used). Linear extrapolation suggests 256k would need ~138 GiB with FP8 — well beyond 96 GiB.

However, with TurboQuant 4-bit KV (#8), the KV cache shrinks by ~1.94x vs FP8. Revised estimate:
- FP8 KV at 256k: ~138 GiB (not feasible)
- TurboQuant 4-bit at 256k: ~85-90 GiB (potentially feasible within 96 GiB)

Gemma 4's hybrid SWA+full attention architecture helps — SWA layers cap at a fixed window regardless of context length, so scaling is sub-linear.

**Verdict**: Blocked on TurboQuant (#8) being merged. Once available, run a VRAM binary search at 256k with `--kv-cache-dtype turboquant_4bit_uniform`. This is the most impactful use case for TurboQuant on our hardware.

---

## 6. 1M Context (NOT FEASIBLE)

At FP8 KV, 1M context for a 31B dense model would need roughly:
- 48 layers × 16 heads × 2 (K+V) × 1M tokens × 256 dim × 1 byte ≈ 187 GiB (KV cache alone)
- Plus 31 GiB weights + drafter

Total: ~220 GiB. Far exceeds the 96 GiB available.

With TurboQuant 4-bit KV (3.88x compression vs bf16, see #8):
- KV cache at 1M: ~48 GiB (down from ~187 GiB with bf16, or ~94 GiB with FP8)
- Plus 32 GiB weights + drafter + CUDA graphs ≈ ~90-95 GiB total

This is right at the edge of 96 GiB — theoretically possible but with zero headroom for CUDA graphs and runtime overhead. Practically not feasible without TP=2.

**Verdict**: Not feasible on single RTX PRO 6000 Blackwell, even with TurboQuant.

---

## 7. LilaRest Turbo Weights (NOT VIABLE)

**Source**: HuggingFace `LilaRest/gemma-4-31b-it-nvfp4-turbo`

These weights are 18 GiB (vs our 31 GiB) because they use `--language-model-only`, which strips the vision encoder entirely. This is the same restriction as vLLM MTP — the whole point of our SGLang entry is multimodal support.

**Verdict**: Incompatible with our multimodal goal. Already covered by the existing `gemma-4-31b-it-turbo-spec` entry which uses these weights for text-only use cases.

---

## 8. TurboQuant KV Cache Compression (HIGH PRIORITY — NOT YET MERGED)

**Source**: SGLang PRs #21419, #23135, #21617

TurboQuant (Google, ICLR 2026, arXiv 2504.19874) is being actively developed for SGLang with multiple competing implementations:

| PR | Approach | Compression | Decode Throughput | Status |
|----|----------|-------------|-------------------|--------|
| **#23135** | Fused Triton kernels, packed 4-bit read during attention | **3.88x** vs bf16 | 93-105% of bf16 baseline | Open, most mature |
| **#21419** | Original TurboQuant impl (3-4 bit) | up to 4.92x vs bf16 | Not benchmarked | Open |
| **#21617** | PolarQuant + QJL + Lloyd-Max, outlier-aware | 2.69-4.4x vs bf16 | ~100% (throughput equivalent) | Open (WIP) |

**PR #23135 is the most promising** — key details:
- `--kv-cache-dtype turboquant_4bit` or `turboquant_4bit_uniform`
- Fused Triton kernels read packed 4-bit KV directly during attention — no dequant buffer needed
- Full CUDA graph support (`--cuda-graph-max-bs 1024`)
- Uniform quantization variant (`turboquant_4bit_uniform`) is +15% faster decode than codebook variant
- Tested on Mistral-7B: 2.1x more KV cache tokens on same GPU, correct inference preserved

**Impact on our entry (estimated)**:
- Currently at FP8 KV (2x compression vs bf16). TurboQuant 4-bit = 3.88x compression vs bf16 = **~1.94x better than FP8**
- 128k context (min_gi=79): KV cache portion shrinks by ~half → estimated min_gi ~55-60 GiB
- **256K context could become feasible** within 96 GiB (currently impossible with FP8 KV)
- 1M context still not feasible (KV alone would be ~48 GiB + 32 GiB weights + drafter overhead)

**How to test** (once merged):
```
--kv-cache-dtype turboquant_4bit_uniform
```

**Caveats**:
- Prefill locked to Triton backend (cannot use FlashInfer/FA3 for extend path)
- Prefill performance degrades with longer inputs: 94% at 128 tokens, drops to 64% at 4k+ due to WHT rotation + quantize kernel overhead
- Incompatible with `trtllm_mha` attention backend (#1) — TurboQuant requires Triton backend for fused kernels
- **None of these PRs are merged yet** — all still open as of 2026-05-14

**Verdict**: Game-changer for long context. When merged, could unlock 256K context and significantly reduce VRAM for existing profiles. However, the prefill penalty and Triton backend lock-in mean it trades prefill speed for memory capacity — opposite tradeoff from `trtllm_mha` (#1). May need separate catalog entries for "fast decode" vs "long context" configurations.

Also relevant: **PR #22004** adds INT8 KV cache (2x vs bf16, +24.9% output throughput) as a simpler intermediate step, but offers less compression than TurboQuant.

---

## Recommended Action Plan

| Priority | Improvement | Expected Gain | Blocker |
|----------|------------|---------------|---------|
| 1 | trtllm_mha attention (#1) | +13-22% decode & prefill | Needs newer SGLang build (merged) |
| 2 | TurboQuant KV cache (#8) | 1.94x less KV VRAM, unlocks 256K ctx | PRs not yet merged |
| 3 | FlashInfer Blackwell prefill (#2) | +5-19% prefill | Needs newer SGLang build (merged) |
| 4 | Piecewise CUDA graphs + spec decode (#3) | Unknown (reduce overhead) | Needs newer SGLang build (merged) |
| 5 | Chunked prefill size tuning (#4) | +5-15% prefill | Can test on current build |

### Two optimization tracks (mutually exclusive attention backends)

**Track A — Fast Decode** (trtllm_mha + FlashInfer + FP8 KV):
- Best for: 32k/128k context, maximum throughput
- Expected: ~110-115 tok/s decode, ~6,000+ tok/s prefill
- Blocker: newer SGLang build (all PRs merged)

**Track B — Long Context** (TurboQuant 4-bit + Triton backend):
- Best for: 256K context on single GPU
- Expected: ~85-90 tok/s decode (slight penalty), 256K context feasible
- Blocker: TurboQuant PRs not yet merged
- Caveat: prefill performance penalty at long inputs (64-94% of baseline)

These tracks are mutually exclusive because TurboQuant requires the Triton attention backend, while trtllm_mha is a separate backend. May warrant two catalog entries when both become available.

### Immediate next step

Rebuild SGLang image when v0.5.12+ stable releases. Test Track A improvements first (all merged). Monitor TurboQuant PRs (#23135 especially) for Track B.
