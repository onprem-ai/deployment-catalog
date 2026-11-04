# 2026-04-26 — Revert NVFP4 to BF16

**Goal:** Fix 10+ min cold start caused by NVFP4 JIT kernel compilation on Blackwell.

## Problem

Same as Qwen3-Embedding-8B: the NVFP4 variant triggers JIT compilation of FP4
CUTLASS GEMM kernels for sm120a on every cold start.

## Decision

Reverted to original BF16 model (`Qwen/Qwen3-VL-Embedding-2B`).

- Cold start: ~50 seconds (vs 10+ min NVFP4)
- VRAM: 7.3 GiB measured (vs ~5 GiB NVFP4)
- Quality: Full BF16 precision, no degradation

## Deployment test

- Pod ready in ~50s
- 2048-dim embeddings verified via AI Gateway
- VRAM: 7.3 GiB (matches `min_gi: 8` for multimodal profile)
