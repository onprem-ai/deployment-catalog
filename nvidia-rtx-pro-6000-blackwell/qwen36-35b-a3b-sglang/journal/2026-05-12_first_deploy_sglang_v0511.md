# Session: First Deploy — SGLang v0.5.11 with EAGLE + extra_buffer

**Date:** 2026-05-12
**Model:** Qwen3.6-35B-A3B NVFP4 (RedHatAI), served by SGLang
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, SM120)
**SGLang Image:** `lmsysorg/sglang:v0.5.11`
**Baseline (vLLM sibling):** 195 tok/s decode, 30,000 tok/s prefill

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date — is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

---

## Changes from Initial Catalog Entry (2026-05-09)

1. **Pinned image tag:** `lmsysorg/sglang:latest` → `lmsysorg/sglang:v0.5.11`
   - v0.5.11 is the first release with Qwen3.6 support (PR #23486)
   - CUDA 13 default (SM120 Blackwell compatibility)
   - Spec V2 is default (no longer needs explicit `SGLANG_ENABLE_SPEC_V2=1` but we keep it)
   - Sources: v0.5.11 release notes, Qwen3.6 SGLang cookbook

2. **Switched from array args to bash -c format** to avoid YAML type coercion
   issue where `${GPU_MEMORY_UTILIZATION}` (float) was rejected by K8s
   admission webhook: "cannot unmarshal number into Go struct field
   Container.spec.predictor.containers.args of type string"

3. **Added `${MEMORY_PARAMS}`** template variable to pick up context length
   and max-running-requests from the selected memory profile.

## Community Research Findings

### SGLang vs vLLM for MoE models (general)

- SGLang tends to be **3-5x faster than vLLM** for MoE at high concurrency
- For single-user (concurrency=1), the gap is smaller — some benchmarks show
  vLLM slightly ahead with default settings
- SGLang's advantage comes from: DeepGEMM for FP8, fused MoE Triton kernels,
  overlap scheduler, and radix prefix cache for GDN/SSM state

### EAGLE Spec-Dec on MoE: DISPROVED — 1.5x Speedup Confirmed

Community reports on Qwen3-30B-A3B showed **no speedup** with EAGLE3 due to
low acceptance lengths. Our results **contradict this** for Qwen3.6-35B-A3B:

- **298 tok/s with EAGLE** vs 195 tok/s vLLM baseline = **1.53x speedup**
- EAGLE accept_len=3.4 (of 4 proposed), 80% acceptance rate
- Peak: 356 tok/s at 100% acceptance

The difference vs community reports: Qwen3.6 has a dedicated **MTP head**
(`model_mtp.safetensors`) that provides much better drafts than the generic
EAGLE3 approach used on Qwen3-30B-A3B. The MTP head is specifically trained
for multi-token prediction on this model's architecture.

### NVFP4 Quantization — RESOLVED

The RedHatAI checkpoint uses `compressed-tensors` format. Initial catalog entry
had `--quantization modelopt_fp4` which caused a `ValueError` on startup:

> Quantization method specified in the model config (compressed-tensors) does
> not match the quantization method specified in the 'quantization' argument
> (modelopt_fp4)

**Fix:** Changed to `--quantization compressed-tensors`. SGLang v0.5.11 uses
`CompressedTensorsW4A4Nvfp4MoE` weight loader — confirmed working.

The March 2026 nightly `KeyError: 'model.layers.1.mixer.experts.w2_weight_scale'`
is NOT present in v0.5.11.

### Known Bugs to Watch For

- **VRAM memory leak with extra_buffer + structured JSON** (fixed in PR #20697)
- **GPU memory growth during long sessions** (issue #10306)
- **Chunked prefill snapshot race in overlap scheduler** (issue #24221, open)
- **Tool call params malformed with qwen3_coder on NVFP4** (reported)

## Deploy Configuration

```
Image: lmsysorg/sglang:v0.5.11
Memory profile: 262k context, 64 max-running-requests (default)
GPU memory utilization: 0.50 (mem-fraction-static)

Key flags:
  --quantization compressed-tensors  (was modelopt_fp4 — fixed)
  --mamba-scheduler-strategy extra_buffer
  --page-size 64
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder

Env:
  SGLANG_ENABLE_SPEC_V2=1
  SGLANG_MAMBA_SSM_DTYPE=bfloat16
```

## Status

**COMPLETED** — Deployed, benchmarked, results documented. See Benchmark Results section.

---

## Deployment Log

### Startup Observations (22:29 UTC)

1. **Weight loading**: 18.25s for 3 shards (21.85 GB), `CompressedTensorsW4A4Nvfp4MoE` used
2. **Blackwell auto-detection**: `SM120 detected: auto-selecting fp4-gemm-backend=flashinfer_cudnn`
3. **Mamba cache**: max_size=267, conv_state=0.37GB, ssm_state=7.85GB, intermediate_ssm=6.33GB
4. **KV cache**: 478,912 tokens, K=4.57GB, V=4.57GB (bfloat16, not fp8)
5. **Available VRAM after allocation**: 48.28 GB (of 96 GB) — comfortable headroom
6. **GDN kernel dispatcher**: decode=TritonGDNKernel, extend=TritonGDNKernel, packed_decode=True
7. **CUDA graph capture**: 25 batch sizes [1..53], blocked on JIT compilation

### JIT Kernel Compilation — CRITICAL: 90+ MINUTES

The `sgl_kernel_jit_nvfp4_blockwise_moe` kernel must be JIT-compiled by NVCC for
SM120a (Blackwell). There is **no AOT pre-compiled kernel** — PR #19437/#20012
removed NVFP4 from AOT `sgl-kernel` and moved it to JIT-only.

**Compilation details:**
- `cicc` (CUDA IR → PTX) runs at ~88% CPU, uses ~3.6 GB RAM
- Preprocessed CUDA source is **17 MB** (CUTLASS + FlashInfer + DeepGemm templates)
- Compilation flags: `-O3 -gencode=arch=compute_120a,code=sm_120a`
- Estimated total time: **90-120+ minutes** on first run

**Root cause of long compile time:**
The CUTLASS NVFP4 blockwise MoE kernel (PR #21314) has separate SM100/SM120
implementations with exhaustive tile size heuristics. The template expansion
produces 17 MB of C++ code, which `cicc -O3` must optimize down to PTX.

**Impact:** Every pod restart requires re-compilation since the JIT cache at
`~/.cache/tvm-ffi/sgl_kernel_jit_nvfp4_blockwise_moe_*` lives inside the
container. For production use, consider:
1. PVC-mounted JIT cache to persist compiled kernels across restarts
2. Init container that pre-compiles the kernel
3. Custom image with AOT-compiled kernel baked in

### OOMKilled During JIT Compilation

**First deploy with memory limit 8 GiB was OOMKilled** after ~2.5 hours of `cicc`
compilation. The `cicc` process was stable at 3.6 GB RSS (VmPeak 4.08 GB) but the
total cgroup memory (including page cache from mmap'd model files) exceeded 8 GiB.

**Fix:** Increased container memory limits from `8Gi` → `16Gi` (requests `4Gi` → `8Gi`).
This is necessary for the NVFP4 MoE JIT kernel compilation on SM120a.
CPU limits also increased from `4` → `8` and requests from `1` → `2`.

### Warnings (non-blocking)

- `DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0` — accuracy risk, not crash
- `TokenizersBackend` — tokenizer loaded generically, model-specific attributes may be missing
- `Unexpected error during package walk: cutlass.cute.experimental` — benign CuTeDSL warning
- `Using default MoE kernel config` — no tuned Triton MoE config for RTX PRO 6000 Blackwell

## Benchmark Results

**Hardware:** NVIDIA RTX PRO 6000 Blackwell (96 GiB VRAM, SM120)
**Memory profile:** 262k context, 64 max-running-requests, mem-fraction-static=0.50
**Concurrency:** 1 (single user)
**Tool:** llmapibenchmark (random input per request)

| Test | Gen TPS | EAGLE Accept Len | EAGLE Accept Rate | vs vLLM (195 tok/s) |
|------|---------|-------------------|--------------------|--------------------|
| Decode (50w→500tok) | **297.6** | 3.0-4.0 | 66-100% | **1.53x** |
| Balanced (500w→500tok) | **293.0** | 3.3-4.0 | 76-100% | **1.50x** |
| Long Context (3000w→100tok) | **~93** | 3.4 | 80% | **0.48x** |

**Prefill throughput (from server logs):**
- Chunked prefill (8192-token chunks): ~11,292 tok/s
- vLLM sibling: 30,000 tok/s
- SGLang prefill is ~2.7x slower than vLLM (due to chunked prefill + SSM state management)

**Peak steady-state decode:** 356 tok/s at 100% EAGLE acceptance (4/4 tokens)

### Analysis

**Decode-heavy workloads: SGLang + EAGLE is 1.5x faster than vLLM.**
The EAGLE speculative decoding with the Qwen3.6 MTP head provides excellent
speculation quality — average 3.4 accepted tokens per step (of 4 proposed).
This contradicts community reports of "no speedup with EAGLE on MoE" which were
based on Qwen3-30B-A3B (without the MTP head). Qwen3.6's built-in MTP head
provides much better drafts.

**Long context: SGLang is 2x slower than vLLM.** The 93 tok/s decode at 3000 words
input is a significant degradation. Possible causes:
1. Chunked prefill (8192-token max) requires multiple prefill passes for large contexts
2. The extra_buffer mamba scheduler has higher overhead with larger SSM states
3. Mamba cache management (conv_state + ssm_state + intermediate) for 40 GDN layers
4. EAGLE verification with large KV caches adds proportionally more overhead

**Prefill: SGLang is 2.7x slower than vLLM.** The vLLM sibling processes prompts at
30,000 tok/s vs SGLang's ~11,292 tok/s. This is expected: vLLM uses flash attention
with continuous batching and no SSM state management, while SGLang must snapshot
SSM state at each prefill chunk boundary.

**Recommendation:**
- For **chat/interactive** workloads (short prompts, long generations): Use SGLang (1.5x faster)
- For **RAG/coding** workloads (long prompts, short generations): Use vLLM (2x faster prefill)
- Consider deploying both and routing based on workload characteristics

