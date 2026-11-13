# 2026-04-26 — Switch to NVFP4 quantization

**Goal:** Replace BF16 weights with NVFP4 for better VRAM density.

## Decision

Switched from `Qwen/Qwen3-Embedding-8B` (BF16, 14.1 GB) to `alexliap/Qwen3-Embedding-8B-NVFP4` (NVFP4, 4.8 GB).

Rationale: embedding models are prefill-only (compute-bound, not memory-bandwidth-bound), so NVFP4 doesn't help throughput much. But the VRAM savings are significant — 8 GiB vs 20 GiB means this model can co-locate with a chat LLM on the same 48 GB GPU.

## Quantization details

- Publisher: alexliap (community)
- Method: llmcompressor / compressed-tensors
- Format: W4A4 (weights AND activations quantized to FP4)
- Group size: 16
- Scale dtype: float8_e4m3fn
- Calibration: 100 samples from CohereLabs/aya_dataset (Greek subset)
- vLLM flag: `--quantization compressed-tensors`

## Risk

No published benchmark comparison vs BF16 baseline. Quality is unverified.
The calibration dataset (100 Greek samples) is small and language-specific.
6,302 downloads suggest community adoption but no guarantees.

## VRAM impact

| Profile | BF16 | NVFP4 |
|---------|------|-------|
| 8k context | 20 GiB | 8 GiB |
| 32k context | 28 GiB | 14 GiB |
| 4k context | 18 GiB | 7 GiB |

## Outcome

Catalog entry updated to use NVFP4 variant. Needs deployment testing to verify:
1. `--quantization compressed-tensors` works with `--runner pooling` on NGC 26.02
2. Embedding quality spot-check vs BF16 baseline
3. Actual VRAM measurement
