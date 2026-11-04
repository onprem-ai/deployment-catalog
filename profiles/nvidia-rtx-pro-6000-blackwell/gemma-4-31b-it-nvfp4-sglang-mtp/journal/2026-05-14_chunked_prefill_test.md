# Chunked Prefill Size Tuning — Gemma 4 31B NVFP4 SGLang MTP

**Date**: 2026-05-14
**Goal**: Test whether larger `--chunked-prefill-size` improves prefill throughput
**Baseline**: 4,719 tok/s prefill (default chunked-prefill-size=8192)
**Hardware**: RTX PRO 6000 Blackwell (96 GiB VRAM, SM100)
**Image**: `lmsysorg/sglang:gemma4-mtp` (v0.5.12.dev93)
**Deploy**: memory_profile=1 (32k ctx, fp8 kv), +20 GiB headroom

## Background

SGLang's default `--chunked-prefill-size` is 8192. Larger values (16384, 32768) can
improve prefill throughput by reducing the number of prefill chunks and overhead per chunk.
The turbo entry achieves 12,448 tok/s prefill partly via `--max-num-batched-tokens 4096` +
`--enforce-eager`. We can't use `--enforce-eager` (kills decode throughput with MTP).

## Results

All configs: `max_total_num_tokens=92784`, `available_gpu_mem=25.49 GB` (unchanged).

### ~6k token prompts (5 unique requests each, excluding cold-start req 1)

| Config | Prefill tok/s (avg) |
|--------|-------------------|
| **Baseline** (chunked=8192, max_prefill=16384) | **4,795** |
| chunked=16384, max_prefill=16384 | 4,810 |
| chunked=32768, max_prefill=16384 | 4,806 |
| chunked=32768, max_prefill=32768 | 4,810 |

No measurable difference. At 6k tokens, the prompt fits in a single chunk regardless
of chunk size, so there's no chunk boundary overhead to eliminate.

### ~20k token prompts (3 requests each, excluding cold-start req 1)

| Config | Prefill tok/s (avg) |
|--------|-------------------|
| **Baseline** (chunked=8192, max_prefill=16384) | **4,540** |
| chunked=32768, max_prefill=32768 | 3,265 |

Lifting `max_prefill_tokens` from 16384 to 32768 **hurt** performance by 28%. The larger
prefill batch likely causes more memory pressure during attention computation on 20k tokens,
outweighing any chunk-boundary savings.

### Decode throughput (from baseline deployment)

| Request | Prompt | Completion | Decode tok/s |
|---------|--------|------------|-------------|
| 1 | 42 | 500 | 84.7 |
| 2 | 42 | 500 | 77.3 |
| 3 | 42 | 500 | 77.7 |

Decode unchanged from previous measurements (~80-90 tok/s).

## Conclusion

**No improvement from chunked prefill tuning. Keep defaults (chunked=8192, max_prefill=16384).**

The prefill bottleneck is not chunk scheduling overhead — it's the attention computation
itself. The real gains require a different attention backend (trtllm_mha, PR #25006) which
needs a newer SGLang build.

---

## Bonus Test: FP4 KV Cache (fp4_e2m1)

Also tested `--kv-cache-dtype fp4_e2m1` (MXFP4, 4x compression vs bf16) as an alternative
to TurboQuant (which isn't available in our image).

**Result: CRASH** — `NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'`

The SWA KV pool (`SWAKVPool` → `MHATokenToKVPool`) tries to allocate zero-initialized
tensors via `torch.zeros()` in FP4 format, but PyTorch doesn't support `fill` for this
experimental dtype on CUDA.

Gemma 4 uses hybrid SWA + full attention (same as Qwen 3.6), so it hits the same
`SWAKVPool` incompatibility documented in the Qwen 3.6 1M entry:
> "FP4 E2M1 is NOT supported for hybrid GDN models — HybridLinearKVPool lacks the FP4
>  pool class; see SGLang issue #14963"

**Verdict**: FP4 KV cache is not usable with Gemma 4 on this SGLang version. FP8 remains
the best available option. TurboQuant (when merged) uses its own pool class and may avoid
this limitation.

## YAML Changes

None — reverted all experimental flags. inference.yaml unchanged from before this test.
