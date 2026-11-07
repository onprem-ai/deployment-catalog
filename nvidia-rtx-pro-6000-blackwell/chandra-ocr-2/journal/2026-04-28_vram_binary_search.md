# VRAM Binary Search Measurement — Chandra OCR 2

**Date**: 2026-04-28
**Hardware**: NVIDIA RTX PRO 6000 Blackwell (96 GiB)
**Container**: vllm/vllm-openai:latest (v0.20.0)
**Method**: Binary search for minimum working gpu_memory_utilization

## Model Characteristics

- **Weights**: 8.61 GiB (from vLLM log: "Model loading took 8.61 GiB memory")
- **Architecture**: Qwen 3.5 VL (5.3B) — **hybrid Mamba + attention**
  - Uses GDN linear attention (Mamba-style) for most layers
  - Only a subset of layers use full attention → minimal KV cache
  - Mamba cache blocks required per concurrent sequence
- **Load time**: 3.5 seconds

## Critical Discovery: Mamba Block Constraint

The original deployment failure was **NOT** caused by insufficient VRAM for model weights or KV cache.

**Root cause**: Default `max_num_seqs=1024` requires 1024 Mamba cache blocks. At `gpu_memory_utilization=0.19-0.25`, only ~592 blocks fit. vLLM error:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (592).
Each decode sequence requires one Mamba cache block, so CUDA graph capture cannot proceed.
Please lower max_num_seqs to at most 592 or increase gpu_memory_utilization.
```

**Fix**: Add `--max-num-seqs 128` (sufficient for OCR workloads — typically 1-4 concurrent requests).

## Empirical Results (with --max-num-seqs 128)

### 16k context (--max-model-len 16384 --kv-cache-dtype fp8_e5m2)

| GPU Util | Reserved (GiB) | nvidia-smi (MiB) | Status |
|----------|----------------|-------------------|--------|
| 0.02     | 1.9            | 14,521            | PASS   |
| 0.06     | 5.8            | 14,523            | PASS   |
| 0.09     | 8.6            | 14,521            | PASS   |
| 0.10     | 9.6            | 12,443            | PASS   |
| 0.11     | 10.6           | 10,983            | PASS   |
| 0.12     | 11.5           | 12,443            | PASS   |
| 0.14     | 13.4           | 12,443            | PASS   |
| 0.16     | 15.4           | 10,983            | PASS   |

### 32k context (--max-model-len 32768 --kv-cache-dtype fp8_e5m2)

| GPU Util | Reserved (GiB) | nvidia-smi (MiB) | Status |
|----------|----------------|-------------------|--------|
| 0.02     | 1.9            | 14,521            | PASS   |
| 0.06     | 5.8            | 14,523            | PASS   |
| 0.08     | 7.7            | 14,521            | PASS   |
| 0.09     | 8.6            | 14,521            | PASS   |

### Without --max-num-seqs (default 1024)

| GPU Util | Status | Error |
|----------|--------|-------|
| 0.19     | FAIL   | Mamba blocks (592) < max_num_seqs (1024) |
| 0.22     | FAIL   | Mamba blocks < max_num_seqs |
| 0.25     | FAIL   | Mamba blocks < max_num_seqs |

## Key Observations

1. **nvidia-smi reading is ~14.5 GiB regardless of utilization setting**. The model weights (8.6 GiB), CUDA graphs (~1.6 GiB), torch.compile overhead, and external overhead (CUDA context + PyTorch runtime ~3 GiB) dominate. KV cache is negligible for this hybrid architecture.

2. **Context length has no measurable VRAM impact** (16k vs 32k identical). The Mamba layers use fixed-size state, not growing KV cache. Only the few attention layers contribute to KV cache.

3. **gpu_memory_utilization only controls Mamba/KV cache allocation**, not model loading. Model weights load outside the managed budget in vLLM v0.20.0.

4. **The nvidia-smi readings fluctuate** (10,983–14,523 MiB) depending on CUDA graph capture timing and memory fragmentation. Peak reading (14,523 MiB) is the safe reference.

## VRAM Breakdown

```
Model weights:          8.61 GiB (from vLLM logs)
CUDA graphs:            1.58 GiB (estimated by vLLM)
torch.compile:         ~1.0  GiB (observed during compilation)
Encoder cache:         ~0.5  GiB (vision encoder profiling)
External overhead:     ~2.5  GiB (CUDA context, PyTorch runtime)
─────────────────────────────────
Total (nvidia-smi):   ~14.2  GiB (peak: 14,523 MiB)
```

## Recommended Catalog Values

```yaml
resources:
  weights_gi: 11
  memory:
    - name: "32k context"
      params: "--max-model-len 32768 --max-num-seqs 128"
      min_gi: 15  # nvidia-smi peak 14,523 MiB + margin
    - name: "16k context"
      params: "--max-model-len 16384 --max-num-seqs 128"
      min_gi: 15  # Same — context length doesn't affect VRAM
```

Both profiles need the same VRAM because Mamba architecture makes KV cache negligible.

## Required inference.yaml Changes

1. Add `--max-num-seqs 128` to all memory profile params
2. Update `min_gi` to 15 for both profiles
3. Previous values (18/20 GiB estimated, then 21/23 GiB) were wrong — based on standard transformer assumptions that don't apply to hybrid Mamba models
