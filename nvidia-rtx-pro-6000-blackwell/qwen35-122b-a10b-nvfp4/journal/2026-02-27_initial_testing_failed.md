# Session: Initial Testing (Failed)

**Date:** 2026-02-27
**Model:** Qwen3.5-122B-A10B NVFP4
**Hardware:** NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date - is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

**Goal:** Deploy Qwen3.5-122B-A10B-NVFP4 on RTX 6000 Blackwell

**Findings:**
- NVIDIA NGC vLLM 26.01-py3 doesn't recognize `qwen3_5_moe` architecture
- vLLM nightly fails with CUTLASS TMA kernel initialization error on SM120
- `--enforce-eager` doesn't help - same CUTLASS error during autotuner
- `--moe-backend marlin` changes MoE backend but linear attention (GDN) kernels still fail

**Outcome:** Failed - Multiple kernel compatibility issues with SM120

**Errors encountered:**

1. NGC container:
```
The checkpoint you are trying to load has model type `qwen3_5_moe` but Transformers does not recognize this architecture.
```

2. vLLM nightly (CUTLASS TMA error):
```
[TensorRT-LLM][ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm.
Error: Error Internal (/workspace/build/aot/generated/cutlass_instantiations/120/gemm_grouped/120/cutlass_kernel_file_gemm_grouped_sm120_M128_BS_group2.generated.cu:60)
```

**Related Issues:**
- [vllm#33416](https://github.com/vllm-project/vllm/issues/33416) - NVFP4 MoE kernels fail on RTX Blackwell (SM12.0)
- [vllm#31085](https://github.com/vllm-project/vllm/issues/31085) - Add SM120 support for native NVFP4 MoE kernels
