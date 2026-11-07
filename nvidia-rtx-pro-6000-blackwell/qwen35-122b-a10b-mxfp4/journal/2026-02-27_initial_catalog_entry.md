# Session: Initial Catalog Entry

**Date:** 2026-02-27

## Summary

Added Qwen3.5 122B A10B MXFP4 to the catalog for NVIDIA RTX 6000 Blackwell.

## Configuration

- Container: `onpremai/llama-cpp-blackwell:latest`
- Model: `unsloth/Qwen3.5-122B-A10B-GGUF` (HuggingFace, MXFP4_MOE subdirectory)
- Status: `unsupported`
- Issue: Only 6GB of 64GB model can offload to GPU; runs at 0.85 tok/s with mmap (disk I/O bound) or OOMKilled without mmap
- Runtime: llama.cpp (CUDA backend)
- Quantization: MXFP4_MOE (native Blackwell MoE kernels, `-cmoe` flag)
- Parameters: 122B total, ~10B activated (MoE, 256 experts, 9 activated)
- Context length: 262K native
- Multimodal: no (text-only)
- Reasoning: yes (thinking mode with `<think>` blocks)
- Tool use: yes
- Weights: ~64 GiB (3 GGUF shards)
- KV cache: q8_0 (both K and V)
- Flash attention: enabled
- VRAM profiles:
  - 262k context (q8_0): 96 GiB total
  - 128k context (q8_0): 80 GiB total
  - 64k context (q8_0): 72 GiB total
- License: Apache-2.0
- Release date: 2026-02-25
