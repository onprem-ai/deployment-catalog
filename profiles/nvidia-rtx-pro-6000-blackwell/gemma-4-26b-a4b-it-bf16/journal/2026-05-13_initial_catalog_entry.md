# 2026-05-13 — Initial Catalog Entry

**Goal:** Create catalog entry for Google's original Gemma 4 26B-A4B-it in BF16 (full precision),
as an alternative to the currently deployed RedHat FP8 Dynamic variant.

## Background

The FP8 variant (`RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic`) is currently deployed and running
well at 207 tok/s decode. However, FP8 quantization causes measurable benchmark regression
(GPQA 68.2 vs 82.3, MMLU-Pro 79.5 vs 82.6). The BF16 variant preserves full model quality
and adds multimodal (image) support that the FP8 variant lacks.

## Model Research

Source: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
Base model: [google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B)
License: Apache-2.0

### Architecture
- 30 layers, MoE with 128 total experts + 1 shared, 8 active per forward pass
- Total parameters: 25.2B (26B with embeddings)
- Active parameters: 3.8B per forward pass
- Sliding window: 1024 tokens
- Context length: 256K tokens
- Vocabulary: 262K
- Vision encoder: ~550M params
- Architecture class: `Gemma4ForConditionalGeneration` (multimodal)

### Weight Format
- BF16 (bfloat16) — native precision, no quantization
- 2 safetensors shards: model-00001-of-00002 (~49.9 GB) + model-00002-of-00002 (~1.6 GB)
- Total on disk: ~49 GiB

### Benchmarks (from model card)
| Benchmark | BF16 (this) | FP8 (RedHat) | Delta |
|-----------|-------------|--------------|-------|
| GPQA Diamond | 82.3 | 68.2 | +14.1 |
| MMLU-Pro | 82.6 | 79.5 | +3.1 |

### Key Differences from FP8 Variant
| Property | FP8 | BF16 |
|----------|-----|------|
| Weights size | ~27 GiB (1 shard) | ~48 GiB (2 shards) |
| Multimodal | No | Yes (image input) |
| Quantization | FP8 Dynamic (llm-compressor) | None |
| Runtime | vLLM | vLLM |

## Catalog Entry

Created `inference.yaml` based on the existing FP8 entry with adjusted:
- `source.repo` → `google/gemma-4-26B-A4B-it`
- `source.weight_format` → `bfloat16`
- `capabilities.multimodal` → `true`
- `resources.weights_gi` → `48`
- `display.benchmarks` → updated to full-precision scores

### Memory Estimates (NOT YET MEASURED)

VRAM estimates are extrapolated from FP8 measurements by adding ~21 GiB for the
larger BF16 weights. These need empirical validation via the VRAM binary search
methodology.

| Context | FP8 Measured | BF16 Estimated |
|---------|-------------|----------------|
| 32k (fp8 kv) | 40 GiB | ~61 GiB |
| 128k (fp8 kv) | 50 GiB | ~71 GiB |
| 256k (fp8 kv) | 65 GiB | ~86 GiB |

## Download

Downloaded via `POST /api/catalog/download?config_id=rtx6000bw--gemma-4-26b-a4b-it-bf16`.
All files stored at `/persistent/models/google--gemma-4-26B-A4B-it--model/`.

## TODO

- [ ] Measure actual VRAM requirements using binary search
- [ ] Measure RAM usage with `kubectl top pod` after deployment
- [ ] Update `min_gi` values in inference.yaml with measured data
- [ ] Test multimodal (image) input
- [ ] Compare decode speed vs FP8 variant (expect slower due to larger weights)
