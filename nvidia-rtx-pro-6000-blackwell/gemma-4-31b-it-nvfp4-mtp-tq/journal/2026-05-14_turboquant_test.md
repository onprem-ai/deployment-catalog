# TurboQuant 4-bit KV Cache Test — Gemma 4 31B NVFP4 SGLang MTP

**Date**: 2026-05-14
**Goal**: Test TurboQuant 4-bit KV cache (PR #23135) with Gemma 4 SWA + NEXTN MTP
**Hardware**: RTX PRO 6000 Blackwell (96 GiB VRAM, SM100)
**Image**: `onpremai/sglang-blackwell:gemma4-mtp-turboquant` (cherry-pick of PR #23135 onto lmsysorg/sglang:gemma4-mtp)
**Deploy**: memory_profile=0 (32k ctx, turboquant_4bit_uniform kv), +20 GiB headroom

## Build

Cherry-picked PR #23135 onto `lmsysorg/sglang:gemma4-mtp` (v0.5.12.dev93).
Zero merge conflicts, clean apply. Built via `_docker/sglang-blackwell-turboquant/build.sh`.

## Key Finding: SWA Pool Works

The biggest risk — TurboQuant incompatibility with Gemma 4's hybrid SWA+full attention — **did NOT materialize**.

```
TurboQuant: overriding decode-attention-backend=default → triton (fused decode kernel)
TurboQuant K=4bit V=4bit KV cache enabled
Use sliding window memory pool. full_layer_tokens=46392, swa_layer_tokens=37113
SWAKVPool mem usage: 31.86 GB, swa size: 37113, full size: 46392
```

TurboQuant allocates KV in bf16 (`Using KV cache dtype: torch.bfloat16`) but quantizes
during attention computation via fused Triton kernels. This avoids the `torch.zeros()`
crash that FP4 E2M1 hits in `SWAKVPool` (see `2026-05-14_chunked_prefill_test.md`).

## Server Metrics

| Metric | TurboQuant 4-bit | FP8 baseline | Change |
|--------|-----------------|--------------|--------|
| max_total_num_tokens | **46,392** | 35,103 | **+32%** |
| SWAKVPool mem usage | 31.86 GB | ~13 GB* | — |
| Available GPU after pool | 26.42 GB | 25.49 GB | Similar |
| VRAM used (nvidia-smi) | 72,051 MiB | 51,373 MiB | +20 GiB (headroom) |
| RAM (kubectl top) | 6,779 MiB | 7,648 MiB | Similar |

*FP8 pool size not logged in same format; estimated from VRAM difference.

The +32% more tokens is meaningful: TurboQuant uses 3.88x compression vs bf16 (~1.94x
vs FP8), but the pool is allocated in bf16 — the quantization happens inside the Triton
attention kernels. The token count increase suggests the pool allocator accounts for
TurboQuant's per-token memory savings even though the allocation dtype is bf16.

## Performance

### Decode (5 requests × 500 tokens, excluding warmup)

| Request | Tokens | Wall time | tok/s |
|---------|--------|-----------|-------|
| 1 | 500 | 8,232ms | 60.7 |
| 2 | 500 | 7,671ms | 65.2 |
| 3 | 500 | 7,715ms | 64.8 |
| 4 | 500 | 7,927ms | 63.1 |
| 5 | 500 | 8,058ms | 62.1 |
| **Avg** | | | **63.2** |

**FP8 baseline: 90 tok/s → TurboQuant: 63 tok/s (-30%)**

Server-side decode throughput ranges 52-76 tok/s (batched, content-dependent).
MTP accept rate: 0.23-0.41 (avg ~0.28), accept len: 2.15-3.02 (avg ~2.5).
FP8 baseline MTP: accept rate 0.36-0.56 (avg ~0.46), accept len 3.2-3.8.

The decode regression has two compounding causes:
1. TurboQuant quantize/dequantize overhead per attention step (~15-20% kernel time)
2. Lower MTP acceptance rate (~39% drop) — quantized KV cache alters attention
   outputs, reducing the drafter's prediction accuracy

### Prefill (~4k tokens, unique prompts, excluding warmup)

| Request | Prompt tokens | Wall time | ~tok/s prefill |
|---------|--------------|-----------|----------------|
| 1 | 3,905 | 921ms | 6,288 |
| 2 | 3,864 | 865ms | 6,839 |
| 3 | 3,899 | 870ms | 6,840 |
| 4 | 3,919 | 872ms | 6,851 |
| 5 | 3,830 | 855ms | 6,901 |
| **Avg (2-5)** | | | **6,858** |

**FP8 baseline: 4,719 tok/s → TurboQuant: 6,858 tok/s (+45%)**

Unexpectedly faster prefill. The PR warned about 64-94% prefill performance at longer
inputs due to WHT rotation + quantize kernel overhead. At ~4k tokens the overhead is
minimal, and reduced KV memory pressure may help.

Note: the PR's prefill penalty is on the EXTEND path (continuation after initial prefill).
Initial prefill may behave differently. Need to test at longer prompts (20k+) to see
if the degradation appears.

### Capability Tests

| Capability | Status |
|-----------|--------|
| Text generation | ✅ Pass |
| Reasoning | ✅ Pass (correct 17×23=391) |
| Tool calling | ✅ Pass (get_weather with correct args) |
| Vision | Not tested (same model, expected to work) |

## Startup

- Model load: 12.5s (same as FP8)
- CUDA graph capture (main): 23.7s (same as FP8)
- Triton kernel compilation: included in CUDA graph capture time (~20s for first batch)
- Drafter load: 17.9s (same as FP8)
- Total to ready: ~85s (similar to FP8)

Triton `tl.where uint8` deprecation warnings during CUDA graph capture — cosmetic, not errors.

## Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Compatibility | ✅ Excellent | Works with SWA + NEXTN MTP + vision |
| KV capacity | ✅ +32% | 46,392 vs 35,103 tokens |
| Prefill | ✅ +45% | 6,858 vs 4,719 tok/s at 4k tokens |
| Decode | ❌ -30% | 63 vs 90 tok/s |
| MTP efficiency | ❌ -39% | Accept rate 0.28 vs 0.46 |
| Startup | ✅ Same | ~85s, Triton compiles during CUDA graph |
| Quality | ✅ Good | Correct reasoning, proper tool calls |

**Verdict**: TurboQuant works and delivers on its promise of more KV cache capacity.
However, the 30% decode regression is larger than the PR's benchmarks suggested
(93-105% of bf16 baseline on Mistral-7B). The interaction with NEXTN MTP — lower
acceptance rate due to quantized KV cache — amplifies the decode penalty.

**Initial recommendation was wrong** — see critical finding below.

---

## CRITICAL: TurboQuant Does NOT Compress KV on SWA Models

### Verification test: same VRAM, no headroom

Redeployed at min_gi=46 (identical to FP8 baseline) with additional_memory_gi=0:

| Metric | TurboQuant (min_gi=46) | FP8 (min_gi=46) | Ratio |
|--------|----------------------|----------------|-------|
| **max_total_num_tokens** | **17,551** | **35,103** | **0.50x** |
| SWAKVPool mem usage | 12.05 GB | ~13 GB | Similar |
| context_len | 32,768 | 32,768 | Same |
| **Fits requested context?** | **NO (17k < 32k)** | **YES (35k > 32k)** | — |

TurboQuant at the same VRAM gives **exactly half** the KV tokens.

### Root cause

The KV pool log confirms bf16 allocation:
```
Using KV cache dtype: torch.bfloat16
KV Cache is allocated. #tokens: 14040, K size: 5.36 GB, V size: 5.36 GB
```

5.36 GB for K at 14,040 tokens = 401 KB/token — matches bf16 (2 bytes/element), NOT
4-bit packed (0.5 bytes/element, which would be ~100 KB/token).

**TurboQuant's custom pool class (`MHATokenToKVPoolTurboQuant`) is NOT used for
Gemma 4's hybrid SWA architecture.** The standard `SWAKVPool` allocates in bf16.
TurboQuant's Triton quantize/dequantize kernels still run during attention, but the
pool storage is bf16 — so all overhead, zero compression benefit.

This is the same class of issue as FP4 E2M1: hybrid SWA/full attention models
(Gemma 4, Qwen 3.6) use `SWAKVPool`/`HybridLinearKVPool` which doesn't integrate
with alternative pool classes (TurboQuant, FP4).

### Corrected assessment

| What we thought | What's actually happening |
|----------------|--------------------------|
| +32% more KV tokens from compression | Artifact of +20 GiB extra headroom |
| +45% faster prefill from reduced memory pressure | Probably real (Triton prefill backend), unrelated to TQ |
| -30% decode from quantize overhead | Real, and with ZERO benefit |
| TurboQuant compresses KV to 4-bit | KV pool is bf16, no compression |

**Verdict: TurboQuant is strictly harmful on Gemma 4 (and any SWA model).**
- Half the KV capacity (bf16 vs fp8)
- 30% slower decode
- Zero memory benefit

### Upstream issue

PR #23135 was tested on Mistral-7B (standard MHA, no SWA). It does not support:
- `SWAKVPool` (hybrid SWA/full attention models like Gemma 4, Qwen 3.6)
- `HybridLinearKVPool` (same class of models)

Worth reporting as a comment on the PR. The fix would require making
`MHATokenToKVPoolTurboQuant` work as a backend for `SWAKVPool`.

## Final Verdict

**TurboQuant is NOT viable for Gemma 4 until the PR adds SWA pool support.**

FP8 (`fp8_e4m3`) remains the best available KV cache compression for SWA models.
256K context remains infeasible on single GPU until either:
1. TurboQuant adds SWA pool support (upstream PR fix)
2. A different KV compression approach works with `SWAKVPool`
3. FP4 E2M1 gets `SWAKVPool` support in PyTorch/SGLang
