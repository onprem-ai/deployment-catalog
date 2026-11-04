# Session: Initial Deployment on RTX 6000 Blackwell

**Date**: 2026-02-28
**Status**: WORKING
**Catalog ID**: `qwen3-coder-next-q6kxl-nvidia-rtx6000bw`

## Summary

Qwen3-Coder-Next is a specialized coding agent model (80B total params, 3B activated MoE) running successfully on RTX 6000 Blackwell with llama.cpp CUDA backend.

## Performance

| Metric | Value |
|--------|-------|
| Generation TPS | ~137 tok/s |
| Prompt Processing | ~800 tok/s (cached) |
| Context Length | 64k tokens |
| Model Size | ~65 GB (Q6_K_XL) |
| VRAM Usage | ~78 GiB |

## Configuration

**Image**: `llama-cpp-blackwell:latest` (custom SM120 build)

**Key Parameters**:
```
-ngl 999                    # All layers on GPU
--flash-attn on             # Flash attention enabled
--no-mmap                   # Load to GPU memory directly
-c 65536                    # 64k context
--cache-type-k q8_0         # KV cache quantization
--cache-type-v q8_0
-b 4096 -ub 4096           # Batch sizes
--jinja                     # Jinja templating for chat
```

## Deployment Steps

1. Model copied from USB drive (already downloaded on AMD system)
2. Deployed via `/api/catalog/{id}/deploy`
3. Tested through AI Gateway

## Notes

- Uses same model files as AMD Strix Halo variant
- Catalog source files updated to use flat filenames (no subdirectory)
- MoE architecture (512 experts, 10 activated) provides excellent TPS despite 80B total params
- 4 parallel slots enabled for concurrent requests
