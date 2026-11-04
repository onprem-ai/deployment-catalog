# 2026-04-27 — Initial Catalog Entry

**Goal:** Create catalog entry for Apertus 8B Instruct FP8 Dynamic on RTX 6000 Blackwell,
replacing the broken NVFP4A16 variant.

## Background

The NVFP4A16 variant (bg-digitalservices/Apertus-8B-Instruct-2509-NVFP4A16) produces NaN logits
due to xIELU's x² activation amplifying FP4 quantization noise. The model's trained alpha_p
values reach up to 174 in early layers, making NVFP4 (4-bit) precision insufficient.
The superseded NVFP4A16 catalog entry documented the full investigation before
it was removed from the catalog.

## Model Research

Source: [RedHatAI/Apertus-8B-Instruct-2509-FP8-dynamic](https://huggingface.co/RedHatAI/Apertus-8B-Instruct-2509-FP8-dynamic)
Base model: [swiss-ai/Apertus-8B-Instruct-2509](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509)
Quantized by: Red Hat AI using [llm-compressor](https://github.com/vllm-project/llm-compressor)

### Architecture (same as base model)
- 32 layers, 32 attention heads, 8 KV heads, head_dim 128
- hidden_size: 4096, intermediate_size: 21504
- vocab_size: 131,072
- max_position_embeddings: 65,536
- model_type: `apertus` / `ApertusForCausalLM`
- Dense transformer with xIELU activation

### Quantization
- FP8 W8A8 dynamic: 8-bit weights (channel symmetric) + 8-bit activations (token symmetric, dynamic)
- Format: `compressed-tensors` (auto-detected by vLLM from config.json `quant_method`)
- lm_head excluded from quantization
- ~8.5 GiB on disk (two safetensors shards: 4.58 + 3.93 GiB)
- Validated on Red Hat OpenShift AI (RHOAI 3.2) and RHAIIS 3.2.5

### Why FP8 should work where NVFP4 failed
- FP8 (E4M3) has 4 exponent + 3 mantissa bits vs NVFP4's 2+1 bits
- Much smaller quantization error → xIELU x² amplification stays bounded
- Red Hat validated the model on vLLM — implies xIELU works correctly with FP8 precision
- No `--quantization` flag needed — vLLM auto-detects `compressed-tensors` from config.json

### Deployment notes from HF README
- Simple: `vllm serve RedHatAI/Apertus-8B-Instruct-2509-FP8-dynamic`
- RHAIIS deployment uses `--enforce-eager --max-model-len 32768`
- No special xIELU flags — Python fallback should work with FP8 precision

## Catalog Entry Decisions

- **Runtime**: vLLM with auto-detected compressed-tensors quantization
- **Image**: `vllm/vllm-openai:cu130-nightly` (standard image, no custom xIELU build needed)
- **No --enforce-eager**: Try without first — FP8 is well-supported by CUDA graphs on SM120
- **No --quantization flag**: compressed-tensors auto-detected from config.json
- **Tool calling**: `--enable-auto-tool-choice --tool-call-parser hermes` (matching 70B sibling)
- **ai_gateway**: served_model_name since model_id differs from vLLM served name

## VRAM Measurements (2026-04-27)

**Hardware**: NVIDIA RTX 6000 Blackwell (96 GiB)
**Container**: vllm/vllm-openai:cu130-nightly
**Method**: Binary search for minimum working gpu_memory_utilization

### Model Characteristics
- **Weights**: 8.52 GiB (from vLLM log)
- **CUDA graphs**: 0.40 GiB
- **KV cache per token (BF16)**: 2 × 32 layers × 8 kv_heads × 128 head_dim × 2 bytes = 128 KiB

### Binary Search Results

| Profile | min_util | ×1.15 margin | recommended_util | min_gi | nvidia-smi |
|---------|----------|-------------|------------------|----------|------------|
| 65k context | 0.21 | 0.2415 | 0.25 | 24 | 25,083 MiB |
| 32k context | 0.16 | 0.184 | 0.18 | 18 | — |
| 16k context | 0.14 | 0.161 | 0.16 | 16 | — |

## Benchmark Results (2026-04-27)

**Method**: llmapibenchmark via benchmark API, random input, concurrency=1

| Test | num_words | max_tokens | gen_tps | prompt_tps | TTFT |
|------|-----------|------------|---------|------------|------|
| Decode | 50 | 500 | 153.84 | — | — |
| Prefill | 1500 | 100 | — | 32,755 | — |
| Balanced | 500 | 500 | 149.11 | 34,233 | 60ms |
| Long Context | 3000 | 100 | 82.06 | 24,104 | 490ms |

**Catalog values**: tps_decode=153.84, tps_prefill=32755

**Notes**:
- Long context (3000 words ≈ 2918 tokens) shows lower generation speed (82 tok/s) due to larger KV cache
- Prefill throughput drops from ~33k to ~24k tok/s with larger inputs (memory bandwidth bound)
- TTFT scales linearly: 60ms at 556 tokens → 490ms at 2918 tokens

## Outcome

Catalog entry complete with measured VRAM profiles and benchmark data.
- FP8 W8A8 dynamic quantization works correctly with xIELU (no NaN)
- CUDA graphs enabled (no --enforce-eager needed)
- Standard vllm/vllm-openai image (no custom xIELU build needed)
- NVFP4A16 sibling marked as `unsupported`
