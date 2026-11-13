# 2026-04-26 — Switch to NVFP4 quantization

**Goal:** Replace BF16 weights with NVFP4 for better VRAM density.

## Decision

Switched from `Qwen/Qwen3-VL-Embedding-2B` (BF16, 4.3 GB) to `LifetimeMistake/Qwen3-VL-Embedding-2B-NVFP4` (NVFP4, 2.1 GB).

This is the best-documented NVFP4 embedding quantization on HuggingFace — includes full FP16 vs NVFP4 benchmarks.

## Quantization details

- Publisher: LifetimeMistake (community)
- Method: NVIDIA ModelOpt (`NVFP4_DEFAULT_CFG`)
- Group size: 16
- Vision encoder: excluded from quantization (stays BF16)
- Calibration: 1000 mixed samples (Polish text, MIRACL multilingual, ViDoRe multimodal, hard-negatives)
- vLLM flags: `--quantization modelopt_fp4 --convert embed`

## Published benchmarks (FP16 vs NVFP4)

| Metric | FP16 | NVFP4 | Delta |
|--------|------|-------|-------|
| nDCG@10 | 0.562 | 0.550 | -2.2% |
| Recall@10 | 0.649 | 0.638 | -1.8% |
| MRR@10 | 0.789 | 0.779 | -1.3% |
| Throughput | 18.4 rps | 21.2 rps | +15.1% |
| Latency | 333 ms | 278 ms | -16.6% |

Probe cosine similarity (NVFP4 vs FP16): mean 0.917 — high fidelity.
Note: ViDoRe image benchmarks show more regression than text tasks.

## VRAM impact

| Profile | BF16 | NVFP4 |
|---------|------|-------|
| Multimodal | 10 GiB | 5 GiB |
| Text-only | 6 GiB | 3 GiB |

## Outcome

Catalog entry updated to use NVFP4 variant. Needs deployment testing to verify:
1. `--quantization modelopt_fp4 --convert embed` on NGC 26.02
2. Vision inputs work correctly with quantized model
3. Actual VRAM measurement
