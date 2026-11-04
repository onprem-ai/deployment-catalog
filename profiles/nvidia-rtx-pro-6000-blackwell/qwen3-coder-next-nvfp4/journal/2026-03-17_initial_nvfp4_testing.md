# Session: Initial NVFP4 Testing (Success)

**Date:** 2026-03-17
**Model:** Qwen3-Coder-Next NVFP4
**Hardware:** NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date - is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

---

## Model Overview

Alibaba's specialized coding agent model with NVIDIA FP4 quantization. 80B total params with 3B activated (MoE with 512 experts, 10 active + 1 shared).

## Hardware Requirements

- **GPU**: NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)
- **System RAM**: Minimum 32GB recommended
- **Storage**: ~45GB for model files (NVFP4 quantized, 10 shards)

---

## Goal

Deploy Qwen3-Coder-Next-NVFP4 on RTX 6000 Blackwell.

## Findings

- vLLM v0.17.1-cu130 works with SM120 NVFP4 MoE fix (PR #33417)
- MoE backend correctly selects FLASHINFER_CUTLASS
- Model loads in ~31 seconds, uses 44.31 GiB VRAM
- Requires `--max-cudagraph-capture-size 128` for hybrid DeltaNet architecture
- Requires custom chat template (missing from quantized model)

## Outcome

Success - NVFP4 working on SM120

## Key Logs Confirming Fix

```
Using NvFp4LinearBackend.FLASHINFER_CUTLASS for NVFP4 GEMM
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends:
['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN']
Model loading took 44.31 GiB memory and 31.143594 seconds
```

## Performance

- 500 tokens generated in 4 seconds = **125 tok/s**
- This is significantly better than the FP8 version (19 tok/s with --enforce-eager)

---

## Recommended Configuration

- Image: `vllm/vllm-openai:v0.17.1-cu130`
- Quantization: `compressed-tensors` (NVFP4 format)
- Key flags:
  - `--max-cudagraph-capture-size 128` (required for hybrid DeltaNet architecture)
  - `--gpu-memory-utilization 0.90` (leave headroom for inference)
  - `--max-model-len 32768` (tested working, higher may work)
  - `--kv-cache-dtype fp8` (memory efficiency)
  - `--chat-template /templates/qwen3_coder_next.jinja` (missing from quantized model)
- Environment variables:
  - `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=629145600` (600MB)
  - `VLLM_FLASHINFER_MOE_BACKEND=latency` (avoid SM120 throughput backend issues)

## Memory Profiles (Measured on RTX 6000 Blackwell 96GB)

| Context Length | gpu_memory_util | VRAM Used | KV Cache Capacity | Notes |
|---------------|-----------------|-----------|-------------------|-------|
| 32k | 0.55 | 55 GiB | 121k tokens | Minimum viable |
| 64k | 0.60 | 57 GiB | 225k tokens | Good balance |
| 128k | 0.62 | 60 GiB | 288k tokens | High context |
| 256k | 0.65 | 63 GiB | 350k tokens | Full native context |

**Why so memory-efficient:**
The hybrid DeltaNet architecture means only 12 of 48 layers use full attention (need KV cache).
The other 36 layers use state-based linear attention with minimal memory footprint.

## Known Issues

- Chat template missing from GadflyII/Qwen3-Coder-Next-NVFP4 - must provide via --chat-template
- Cannot use larger cudagraph capture sizes (>128) due to hybrid DeltaNet architecture

## Why `--max-cudagraph-capture-size` Is Required

Qwen3-Next uses a hybrid architecture combining:
1. **DeltaNet Linear Attention**: Linear attention with causal convolution
2. **Full Attention**: Standard transformer attention (every 4th layer)
3. **MoE**: 512 experts, 10 activated + 1 shared = 3B active params

The DeltaNet components have constraints similar to Mamba's causal_conv1d. The default cudagraph capture size exceeds the internal state size, triggering assertion errors.

## References

- [PR #33417](https://github.com/vllm-project/vllm/pull/33417) - Add SM120 device capability checks
- [Qwen3-Next vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html)
- [Original Model](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [NVFP4 Quantized](https://huggingface.co/GadflyII/Qwen3-Coder-Next-NVFP4)
