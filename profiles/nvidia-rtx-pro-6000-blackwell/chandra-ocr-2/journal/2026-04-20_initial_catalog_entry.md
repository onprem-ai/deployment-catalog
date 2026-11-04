# Session: Initial Catalog Entry

**Date:** 2026-04-20

## Summary

Added Chandra OCR 2 to the catalog for NVIDIA RTX 6000 Blackwell.

## Configuration

- Container: `vllm/vllm-openai:latest`
- Model: `datalab-to/chandra-ocr-2` (HuggingFace)
- Status: `stable`
- Runtime: vLLM
- Weight format: bfloat16
- Parameters: 5.3B
- Context length: 256K (262144 tokens)
- Multimodal: yes (vision-language, image input)
- Weights: ~11 GiB (BF16)
- KV cache dtype: fp8_e5m2
- Tensor parallel size: 1
- VRAM profiles:
  - 32k context: 20 GiB total
  - 16k context: 18 GiB total
- License: OpenRAIL-M (modified)
- Release date: 2026-03-18
