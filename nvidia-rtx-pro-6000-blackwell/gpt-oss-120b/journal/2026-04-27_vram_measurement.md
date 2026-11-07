# 2026-04-27 — VRAM Binary Search Measurement

**Goal:** Measure correct VRAM profiles for GPT-OSS-120B on RTX 6000 Blackwell (96 GiB).

## Model Characteristics

- **Architecture**: MoE — 120B total, 4 active experts per token, 128 experts
- **Layers**: 36 (18 full attention + 18 sliding window @ 128 tokens)
- **Attention**: 64 heads, 8 KV heads, head_dim 64
- **Quantization**: MXFP4 (microscaling FP4) on MLP expert weights via Marlin backend
- **Model loading**: 65.97 GiB (from vLLM log)
- **Non-KV overhead**: ~8.5 GiB (activations, buffers, CUDA graphs)
- **Container**: nvcr.io/nvidia/vllm:26.02-py3 (vLLM 0.15.1)

Previous catalog had `weights_gi: 92` and a single `min_gi: 92` profile — both incorrect.
Actual disk size is ~60 GiB (MXFP4), loaded size is 65.97 GiB (Marlin dequantization).

## KV Cache Calculation

Only 18 of 36 layers use full attention (the other 18 are sliding window @ 128 tokens).
In practice, vLLM allocates blocks for all 36 layers uniformly, but the concurrency
calculation accounts for sliding window layers needing fewer tokens.

| KV Dtype | Per token (all layers) | 131k full request | 65k full request |
|----------|----------------------|-------------------|------------------|
| BF16 | 72 KiB | ~9.0 GiB | ~4.5 GiB |
| FP8 | 36 KiB | ~4.5 GiB | ~2.25 GiB |

## Binary Search Results

**Hardware**: RTX 6000 BW (96 GiB, nvidia-smi reports 97,887 MiB)

| Profile | min min_gi | min_util | concurrency@min | recommended min_gi | rec util |
|---------|-------------|----------|-----------------|---------------------|----------|
| 131k BF16 | 79 | 0.82 | 1.12x | 91 | 0.95 |
| 131k FP8 KV | 76 | 0.79 | 1.05x | 88 | 0.92 |
| 65k BF16 | 77 | 0.80 | 1.37x | 89 | 0.93 |
| 65k FP8 KV | 76 | 0.79 | 1.99x | 88 | 0.92 |

### Failed attempts (concurrency < 1.0x)

| Profile | min_gi | concurrency | Result |
|---------|----------|-------------|--------|
| 131k BF16 | 78 | 0.92x | FAIL |
| 131k FP8 | 75 | 0.26x | FAIL |
| 65k BF16 | 76 | 0.99x | FAIL |
| 65k FP8 | 75 | 0.49x | FAIL |

## nvidia-smi at Recommended Utilization

| Profile | min_gi | util | nvidia-smi (MiB) | nvidia-smi (GiB) |
|---------|----------|------|------------------|-------------------|
| 131k BF16 | 91 | 0.95 | 94,505 | 92.3 |
| 131k FP8 KV | 88 | 0.92 | 91,591 | 89.4 |
| 65k BF16 | 89 | 0.93 | 92,561 | 90.4 |
| 65k FP8 KV | 88 | 0.92 | 91,591 | 89.4 |

## Key Observations

- Model weights dominate: 65.97 GiB loaded + 8.5 GiB overhead = ~74.5 GiB non-KV
- KV cache differences between profiles are modest (1-4 GiB) relative to model size
- Marlin FP4 backend warning: "GPU does not have native FP4 support, using weight-only compression"
- MXFP4 is auto-detected from config.json (quant_method: mxfp4)
- Previous `weights_gi: 92` was incorrect — corrected to 60 (disk) / 66 (loaded)

## Model Source

Copied from a local copy instead of HuggingFace download:
```
rsync -ahv --bwlimit=265M <local-model-copy>/openai--gpt-oss-120b--model/ /persistent/models/openai--gpt-oss-120b--model/
```
60 GiB transferred at ~260 MB/s in ~4 minutes.

## Disk Space Management

Freed space by removing:
- `RedHatAI--Qwen3-Coder-Next-NVFP4--model` (45G, backup copy kept)
- `bg-digitalservices--Apertus-8B-Instruct-2509-NVFP4A16--model` (5.7G, unsupported)
- Backed up `bg-digitalservices--Apertus-70B-Instruct-2509-NVFP4--model` (40G) first
