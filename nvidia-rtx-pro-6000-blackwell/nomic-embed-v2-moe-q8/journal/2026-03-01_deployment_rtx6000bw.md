# Session: Nomic Embed v2 MoE Q8 - NVIDIA RTX 6000 Blackwell Deployment

**Date:** 2026-03-01
**Catalog ID:** `rtx6000bw--nomic-embed-v2-moe-q8`
**Hardware:** NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96GB VRAM)

## Summary

Successfully deployed Nomic Embed Text v2 MoE Q8_0 on NVIDIA RTX 6000 Blackwell using llama.cpp with CUDA backend.

## Performance

| Metric | Value |
|--------|-------|
| Embedding Throughput | ~11,000 tokens/sec |
| Embedding Dimensions | 768 |
| Context Length | 512 tokens |
| VRAM Usage | ~1GB |

## Configuration

- Image: `llama-cpp-blackwell:latest` (local image with SM120 support)
- Backend: CUDA with flash attention
- Quantization: Q8_0 (highest quality)
- Batch processing supported

## Notes

- This is an **embedding-only model** - no chat completions support
- Uses the `/v1/embeddings` endpoint
- Performance is ~12% faster than AMD Strix Halo version (~9,800 tok/s)
- Memory footprint is minimal (~500MB weights + ~1GB runtime)

## AI Gateway Integration

Note: Embedding passthrough via AI Gateway requires OpenAI-compatible endpoint configuration. Direct pod access works correctly.
