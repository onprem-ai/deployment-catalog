# 2026-04-22 — Initial Catalog Entry

**Goal:** Create catalog entry based on model README benchmarks.

## Model Info

- **Model**: LilaRest/gemma-4-31B-it-NVFP4-turbo
- **Base**: nvidia/Gemma-4-31B-IT-NVFP4 → google/gemma-4-31B-it
- **Source**: https://huggingface.co/LilaRest/gemma-4-31B-it-NVFP4-turbo
- **Architecture**: Gemma4ForCausalLM (text-only, vision/audio stripped)
- **Hardware**: 1x NVIDIA RTX 6000 Blackwell (96GB)
- **Weights**: ~18 GiB (4 shards, down from 31 GiB for NVIDIA original)

## Findings

- README reports 18.5 GiB GPU memory at 16k context on RTX PRO 6000
- README reports ~180K max context on RTX PRO 6000 (96GB)
- README recommends `cu130-nightly` image with `--quantization modelopt`
- Text-only variant uses `Gemma4ForCausalLM` (not `Gemma4ForConditionalGeneration`)

Key differences from `gemma-4-31b-it-nvfp4` (NVIDIA original):
- No `--chat-template` mount needed (Gemma4ForCausalLM has built-in template)
- `cu130-nightly` image instead of `gemma4-cu130`
- 18 GiB weights vs 31 GiB — more room for KV cache

## Memory Estimates (from README, not measured)

| Context | Estimated min_gi | Notes |
|---------|-------------------|-------|
| 16k | 25 GiB | README: 18.5 GiB GPU memory |
| 64k | 45 GiB | Interpolated |
| 128k | 70 GiB | Interpolated from 180K max on 96GB |

## Outcome

Catalog entry created, pending download and deployment for verification.
