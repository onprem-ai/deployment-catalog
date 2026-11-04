# Session: Initial Catalog Entry

**Date:** 2026-02-27

## Summary

Added Qwen3.5 122B A10B AWQ to the catalog for NVIDIA RTX 6000 Blackwell.

## Configuration

- Container: `vllm/vllm-openai:nightly`
- Model: `QuantTrio/Qwen3.5-122B-A10B-AWQ` (HuggingFace)
- Status: `unsupported`
- Issue: vLLM nightly lacks RMSNormGated.activation support for Qwen3.5 hybrid architecture
- Runtime: vLLM
- Quantization: AWQ (4-bit)
- Parameters: 122B total, ~10B activated (MoE)
- Context length: 262K native (64K recommended for single GPU)
- Multimodal: no (text-only)
- Reasoning: yes (thinking mode with `<think>` blocks)
- Tool use: yes
- Weights: ~77 GiB (AWQ 4-bit, 28 shards)
- VRAM profiles:
  - 64k context: 96 GiB total
  - 32k context: 88 GiB total
  - 16k context: 82 GiB total
- License: Apache-2.0
- Release date: 2026-02-25
