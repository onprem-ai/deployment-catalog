# Session: Q8 vs NVFP4 Benchmark Comparison

**Date**: 2026-04-04

## Summary

Benchmark comparison between Q8_0 (llama.cpp) and NVFP4 (RedHatAI checkpoint, vLLM v0.17.1) on RTX 6000 Blackwell. Key finding: Q8 wins single-user generation speed, NVFP4 wins prefill, TTFT, and multi-user scaling.

## Single-User Performance

| Metric | Q8 (llama.cpp) | NVFP4 RedHatAI (vLLM) | Winner |
|--------|----------------|----------------------|--------|
| **Generation** | **141 tok/s** | 123 tok/s | **Q8 +14%** |
| **Prompt (prefill)** | 3,025 tok/s | **5,557 tok/s** | NVFP4 +84% |
| **TTFT** | 130ms | **70ms** | NVFP4 -46% |
| VRAM | 83 GB | 72 GB | NVFP4 -13% |

## Multi-User Scaling

| Concurrency | Q8 (llama.cpp) | NVFP4 (vLLM) | Winner |
|-------------|----------------|--------------|--------|
| 1 | **141 tok/s** | 123 tok/s | Q8 +14% |
| 2 | 100 tok/s | **215 tok/s** | NVFP4 +115% |
| 4 | 248 tok/s | **389 tok/s** | NVFP4 +57% |

## Analysis

**Q8 (llama.cpp) advantages:**
- 14% faster single-user generation (141 vs 123 tok/s)
- Higher quality quantization (Q8 vs FP4)
- Simpler deployment (single llama-server binary)

**NVFP4 (vLLM) advantages:**
- 84% faster prefill (5.5k vs 3k tok/s) - better for long prompts/context
- 46% lower TTFT (70ms vs 130ms) - feels more responsive
- Much better multi-user scaling via continuous batching
- 13% lower VRAM usage

## Recommendation

| Use Case | Recommended |
|----------|-------------|
| Single user, short prompts, long outputs | **Q8 llama.cpp** |
| Single user, long code context | NVFP4 vLLM (faster prefill) |
| Multi-user / shared deployment | **NVFP4 vLLM** |
