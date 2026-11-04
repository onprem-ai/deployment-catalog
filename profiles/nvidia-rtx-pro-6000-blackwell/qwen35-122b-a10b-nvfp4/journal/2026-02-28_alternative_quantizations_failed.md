# Session: Alternative Quantizations (Failed)

**Date:** 2026-02-28
**Model:** Qwen3.5-122B-A10B NVFP4
**Hardware:** NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date - is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

**Goal:** Try Q4_K_M, AWQ, and MXFP4 as alternatives to NVFP4

**Findings:**

**Q4_K_M (llama.cpp):**
- Model loads and runs at ~112 tok/s
- Output is garbled/nonsensical (mixed Cyrillic, Chinese, random tags)
- Root cause: Docker image built with `-DGGML_CUDA_FORCE_CUBLAS=ON` causes FP16 cuBLAS overflow on SM120
- CPU-only (`-ngl 0`) produces correct output, confirming GPU issue
- GitHub issue: [llama.cpp#19683](https://github.com/ggml-org/llama.cpp/issues/19683)

**AWQ (vLLM):**
- Model loads (71.04 GiB)
- Crashes with `AttributeError: 'RMSNormGated' object has no attribute 'activation'`
- Fundamental architecture incompatibility, not Blackwell-specific

**MXFP4_MOE (llama.cpp):**
- Only ~6GB offloads to GPU, rest stays in CPU memory
- With mmap: 0.85 tok/s (disk I/O bottleneck)
- Without mmap: OOMKilled (64GB model > 31GB RAM)

**Outcome:** Failed - All alternatives have issues
