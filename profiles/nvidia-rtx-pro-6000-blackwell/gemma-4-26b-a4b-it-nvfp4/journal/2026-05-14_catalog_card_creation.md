# Session: Catalog Card Creation

**Date:** 2026-05-14
**Model:** Gemma 4 26B-A4B IT NVFP4 (`RedHatAI/gemma-4-26B-A4B-it-NVFP4`)
**Hardware:** 1x NVIDIA RTX 6000 Blackwell (96GB VRAM)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML frameworks.
> Information 2+ weeks old may be outdated. Verify against current releases.

**Goal:** Create catalog card for RedHatAI's NVFP4-quantized Gemma 4 26B-A4B model.

**Outcome:** Card created. NOT YET DEPLOYED — requires testing with vLLM image.

---

## Model Analysis

### Source & Quantization

- **HuggingFace repo:** `RedHatAI/gemma-4-26B-A4B-it-NVFP4`
- **Base model:** `google/gemma-4-26B-A4B-it`
- **Quantization method:** compressed-tensors (llm-compressor) — NVFP4 weights + FP8 activations
- **Format:** `nvfp4-pack-quantized` (per `config.json` quantization_config)
- **This is NOT NVIDIA modelopt** — RedHat uses their own llm-compressor pipeline

### Key Differences from Other Variants

| Variant | Repo | Weights | Quant Method | vLLM Flag |
|---------|------|---------|-------------|-----------|
| FP8 Dynamic | `RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic` | 27 GiB | compressed-tensors | (auto-detected) |
| **NVFP4** | `RedHatAI/gemma-4-26B-A4B-it-NVFP4` | **16 GiB** | compressed-tensors | `--quantization compressed-tensors` |
| NVIDIA 31B NVFP4 | `nvidia/Gemma-4-31B-IT-NVFP4` | 31 GiB | modelopt | `--quantization modelopt` |

### Architecture

- 26B total parameters, 3.8B active per forward pass
- 128 MoE experts (from `config.json`)
- Text-only (A4B variant has no vision encoder, unlike 31B)
- 262K native context length

### Weight Files

Single safetensors file:
- `model.safetensors`: 16.42 GB (15.29 GiB)

Total download: ~16.5 GB

---

## Decisions Made

### Image Selection: `gemma4-0505-cu130`

Chose the same image as the FP8 Dynamic variant of this model. However, the HuggingFace README
states: "Run it with `vllm serve` on vllm nightly as [PR #39045](https://github.com/vllm-project/vllm/pull/39045) is needed."

**Risk:** The `gemma4-0505-cu130` image may not include this PR. If the model fails to load,
try `vllm/vllm-openai:v0.19.0-cu130` or a newer nightly build.

### Quantization Flag: `--quantization compressed-tensors`

The FP8 Dynamic variant of this model works without an explicit quantization flag (auto-detected).
However, following the pattern from `qwen3-coder-next-nvfp4-redhat` (also a RedHat compressed-tensors
model), we use `--quantization compressed-tensors` explicitly for reliability.

### No `--chat-template` Flag

RedHat models preserve the tokenizer config from the base model, so the chat template is
embedded in `tokenizer_config.json`. Unlike NVIDIA modelopt cards, no explicit `--chat-template`
is needed. (Same pattern as the FP8 Dynamic variant.)

### Memory Profiles: Estimated

Memory profiles are **estimated** by subtracting the weight size difference from the FP8 Dynamic
card's measured values:
- FP8 Dynamic: 27 GiB weights → 40 GiB at 32k context
- NVFP4: 16 GiB weights → ~30 GiB at 32k context (11 GiB savings)

Added a 96k context profile since the HuggingFace README specifically uses `--max-model-len 96000`.

**TODO:** Measure actual VRAM after first deployment using the binary search methodology
(see `gemma-4-31b-it-nvfp4/journal/2026-04-27_vram_measurement_binary_search.md`).

---

## TODO Before Production

1. **Test deployment** — download model, deploy with the card, verify vLLM loads successfully
2. **Verify vLLM image compatibility** — if `gemma4-0505-cu130` fails, try `v0.19.0-cu130`
3. **Measure VRAM** — replace estimated memory profiles with measured values
4. **Run benchmarks** — measure tps_decode and tps_prefill, add to display.performance
5. **Run capability tests** — vision (should fail, text-only) and tool calling
