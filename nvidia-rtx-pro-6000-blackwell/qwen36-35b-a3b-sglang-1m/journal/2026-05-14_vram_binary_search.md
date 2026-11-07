# VRAM & RAM Binary Search Measurement — Qwen3.6 35B-A3B 1M Context

**Date**: 2026-05-14
**Hardware**: NVIDIA RTX PRO 6000 Blackwell (96 GiB VRAM, SM120a)
**Container**: onpremai/sglang-blackwell:v0.5.11 (custom image with pre-compiled NVFP4 JIT kernels)
**Method**: Binary search for minimum working mem-fraction-static (SGLang equivalent of gpu_memory_utilization)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date — is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

---

## Model Characteristics

- **Weights**: 21.85 GiB (NVFP4, CompressedTensorsW4A4Nvfp4MoE)
- **Architecture**: Qwen3_5MoeForConditionalGeneration
  - Hybrid attention: 30 × Gated DeltaNet + 10 × Full Attention (3:1 ratio)
  - MoE: 256 experts, 8 routed + 1 shared per token
  - Vision encoder: 27 blocks, patch_size 16
- **Draft model (EAGLE MTP)**: Qwen3_5ForCausalLMMTP, 3.79 GiB
- **Load time**: ~20s weights + ~5s draft
- **Context extension**: YaRN rope scaling factor 4.0 (262k → 1M)
- **KV cache dtype**: fp8_e5m2 (NOT fp4 — HybridLinearKVPool lacks FP4 pool class, SGLang issue #14963)

## FP4 KV Cache — NOT SUPPORTED for Hybrid GDN Models

Attempted `--kv-cache-dtype fp4_e2m1` but hit:

1. `AssertionError: KV4 MHA expects attention_backend triton/torch_native` → added `--attention-backend triton`
2. `NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'` → FP4 KV only works with `HybridMHAKVPool` / `HybridMLAKVPool`, NOT `HybridLinearKVPool` (used by GDN models)

**Fix is upstream-only** — requires adding FP4 pool class to `HybridLinearKVPool`.

Switched to **fp8_e5m2** which provides 2x compression vs BF16 (vs 3.56x for FP4).

## Theoretical Estimate

```
model_weights:        21.85 GiB
mamba_cache:          12.18 GiB (conv 0.52 + SSM 11.04 + intermediate 0.62)
                      Note: Mamba cache is NOT controlled by mem-fraction-static
kv_cache (main):      12.86 GiB (FP8, K 6.43 + V 6.43, 1.35M tokens at min_gi=48)
draft_model (EAGLE):   3.79 GiB
draft_kv_cache:        1.28 GiB (K 0.64 + V 0.64)
cuda_graphs:           0.49 GiB (main 0.25 + draft 0.24)
---
Total at min_gi=48: ~55 GiB
Available:             41 GiB of 96 GiB

SGLang mem-fraction-static controls: KV token pool allocation
SGLang does NOT control: model weights, mamba cache, EAGLE model
```

**KV budget at different min_gi values:**

The KV pool must hold at least `context_length` tokens (1,048,576) to serve
a single full-context request. At min_gi=48, pool = 1,348,480 tokens (1.29x needed).

## Empirical Results (VRAM)

**Important**: The deploy API calculates `gpu_memory_utilization = min_gi / system_gpu_gi`.
The `gpu_memory_utilization` query param is IGNORED — only `min_gi` in the YAML matters.
To change mem-fraction-static, you must change `min_gi`, reload the catalog, redeploy.

| min_gi | Token Pool | Mamba Seqs | nvidia-smi (GiB) | Status | Notes |
|----------|-----------|------------|-------------------|--------|-------|
| 42 | 1,035,008 | 289 | 47 | **FAIL** | Pool < 1M context (13k tokens short) |
| 43 | 1,088,832 | 303 | 49 | **MIN** | 3.8% above 1M context — minimum working |
| 50 | 1,452,992 | 405 | 55 | PASS | 39% above 1M context |
| 55 | 1,763,264 | 493 | 61 | PASS | 68% above 1M context |

**Key finding**: Lower `min_gi` does NOT always mean less total nvidia-smi VRAM. SGLang
allocates mamba cache proportional to available memory after KV pool. At min_gi=55 vs 50,
mamba cache grew from 13 GiB → 15.8 GiB (max_mamba_cache_size 405 → 493 sequences).

**Minimum working: min_gi=43**
- Token pool 1.09M = 3.8% above 1M context minimum
- nvidia-smi: 49 GiB (47 GiB free of 96 GiB)
- Inference verified: model responds correctly
- User adds more VRAM at deploy time from remaining GPU headroom

## RAM Measurement

After deploying at minimum working min_gi=43 and serving test requests:

```
kubectl top pod <pod-name> -n <namespace>
NAME                CPU(cores)   MEMORY(bytes)
...-5ztzm           1222m        6970Mi
```

**Measured RAM**: 6,970 MiB (6.8 GiB)

Additional measurements at other min_gi levels:

| min_gi | kubectl top (MiB) | Notes |
|----------|-------------------|-------|
| 42 | 8,095 | Higher — larger intermediate buffers |
| 43 | 6,498 | Minimum working — used for catalog |
| 48 | 6,469 | Initial measurement (before binary search) |
| 50 | 6,970 | |
| 55 | 6,034 | |

RAM usage varies 6-8 GiB across min_gi levels. Higher than typical vLLM
models (1-3 GiB) because SGLang + Mamba state management + EAGLE draft model
use more host memory.

| Value | Current | Recommended | Rationale |
|-------|---------|-------------|-----------|
| requests.memory | 8Gi | 8Gi | Covers peak measured (8 GiB) |
| limits.memory | 16Gi | 16Gi | ceil(8 × 2.0) = 16 GiB, also needed for JIT safety |

Note: The 16 GiB limit is necessary because SGLang JIT-compiles NVFP4 kernels
on first startup (if cache is cold). The `cicc` compiler uses up to 7.4 GiB RSS
during compilation. Our custom image has pre-compiled kernels, so normal operation
only needs ~6-8 GiB, but keeping 16 GiB limit provides safety margin.

## Benchmark Results (EAGLE MTP Enabled)

**Concurrency: 1, FP8 KV cache, 1M context, YaRN factor 4.0**

| Test | Input | Output | Gen TPS | Notes |
|------|-------|--------|---------|-------|
| Decode | 50 words | 500 tokens | **300** | Matches 262k sibling (298 tok/s) |
| Prefill | 1500 words | 100 tokens | **156** | Total throughput from progress bar |
| Balanced | 500 words | 500 tokens | **260** | Typical interactive workload |
| Long Context | 3000 words | 100 tokens | **109** | ~3x slower than short context decode |

**Comparison vs no-EAGLE (same config, FP8 KV, 1M context):**

| Test | Without EAGLE | With EAGLE | Speedup |
|------|--------------|------------|---------|
| Decode (50w) | 163 tok/s | 300 tok/s | **1.84x** |
| Long Context (3000w) | 94 tok/s | 109 tok/s | **1.16x** |

EAGLE MTP provides dramatic speedup for short-context decode (1.84x) and modest
improvement for long context (1.16x). The MTP head on Qwen3.6 provides excellent
draft quality.

**Prefill throughput** (from server logs): ~11,292 tok/s chunked prefill.
Note: benchmark tool prompt_throughput metric is unreliable for reasoning models
(produces +Inf or negative values due to `<think>` tag output format).

## Recommended Catalog Values

```yaml
resources:
  weights_gi: 24
  memory:
    # Binary search 2026-05-14: min_working=43 GiB
    # User adds more VRAM at deploy time from remaining GPU headroom.
    - name: "1M context (fp8 kv, EAGLE)"
      params: "--context-length 1048576 --max-running-requests 4"
      min_gi: 43  # Minimum working from binary search
    - name: "512k context (fp8 kv, EAGLE)"
      params: "--context-length 524288 --max-running-requests 8"
      min_gi: 38  # estimated (less KV, more concurrency)
    - name: "262k context (fp8 kv, EAGLE)"
      params: "--context-length 262144 --max-running-requests 16"
      min_gi: 34  # estimated (native context, max concurrency)

# inference_service container resources:
requests:
  cpu: "2"
  memory: "8Gi"   # Measured: 6.8 GiB actual (kubectl top at 0.52)
limits:
  cpu: "8"
  memory: "16Gi"  # ceil(8 × 2.0), also needed for JIT compilation safety
```

## Notes

- Custom Docker image `onpremai/sglang-blackwell:v0.5.11` with pre-compiled NVFP4 JIT kernels
- FP4 KV is NOT supported for hybrid GDN models — only FP8 or BF16
- `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` required to extend context > 262k
- `SGLANG_ENABLE_SPEC_V2=1` required for EAGLE + radix cache + extra_buffer on hybrid GDN
- `SGLANG_MAMBA_SSM_DTYPE=bfloat16` halves mamba cache size (float32 → bfloat16)
- Mamba cache is allocated separately from KV pool — not controlled by mem-fraction-static
- Binary search measures minimum mem-fraction-static that provides enough KV tokens
