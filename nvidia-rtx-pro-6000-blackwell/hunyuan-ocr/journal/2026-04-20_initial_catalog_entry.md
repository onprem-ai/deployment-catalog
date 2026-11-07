# Session: Initial Catalog Entry

**Date:** 2026-04-20

## Summary

Added HunyuanOCR to the catalog for NVIDIA RTX 6000 Blackwell.

## Configuration

- Container: `vllm/vllm-openai:latest`
- Model: `tencent/HunyuanOCR` (HuggingFace)
- Status: `experimental`
- Runtime: vLLM
- Weight format: bfloat16
- Parameters: 1B
- Context length: 32K (32768 tokens)
- Multimodal: yes (vision-language, image input)
- Weights: ~2 GiB (BF16, 4 shards)
- Tensor parallel size: 1
- Prefix caching: disabled (`--no-enable-prefix-caching`)
- VRAM profiles:
  - 32k context: 8 GiB total
  - 16k context: 6 GiB total
- License: Tencent Hunyuan Community License
- Release date: 2025-11-25
