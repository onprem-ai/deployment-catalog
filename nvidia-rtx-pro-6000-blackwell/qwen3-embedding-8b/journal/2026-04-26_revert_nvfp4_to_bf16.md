# 2026-04-26 — Revert NVFP4 to BF16

**Goal:** Fix 10+ min cold start caused by NVFP4 JIT kernel compilation on Blackwell.

## Problem

The NVFP4 variant (`alexliap/Qwen3-Embedding-8B-NVFP4`) triggers JIT compilation
of FP4 CUTLASS GEMM kernels for sm120a (Blackwell) inside the vLLM container on
every cold start. The `nvcc` processes compile flashinfer FP4 kernels from source,
taking 10+ minutes with a 4-core CPU limit.

## Research

Searched HuggingFace for alternative quantized variants:

| Variant | Format | Risk on Blackwell |
|---------|--------|-------------------|
| alexliap/Qwen3-Embedding-8B-FP8-DYNAMIC | FP8 W8A8 | Known FP8 + pooling bugs (vllm#33970) |
| boboliu/Qwen3-Embedding-8B-W4A16-G128 | INT4/Marlin | Marlin negative-scale bug on SM12.x |
| alexliap/Qwen3-Embedding-8B-NVFP4 | NVFP4 | JIT compilation 10+ min |

All quantized variants carry risks on the sm120a Blackwell kernel stack.
With 98 GiB VRAM, quantization provides no meaningful benefit.

## Decision

Reverted to original BF16 model (`Qwen/Qwen3-Embedding-8B`).

- Cold start: ~50 seconds (vs 10+ min NVFP4)
- VRAM: 19.9 GiB measured (vs ~8 GiB NVFP4)
- Quality: Full BF16 precision, no degradation

## Deployment test

- Pod ready in ~50s (BF16 loads natively, no kernel JIT)
- 4096-dim embeddings verified via AI Gateway
- VRAM: 19.9 GiB (matches `min_gi: 20` for 8K context profile)
