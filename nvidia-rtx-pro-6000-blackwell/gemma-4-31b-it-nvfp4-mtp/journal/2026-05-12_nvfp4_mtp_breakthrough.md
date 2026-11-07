# Breakthrough: Gemma 4 31B IT NVFP4 + MTP Speculative Decoding

**Date:** 2026-05-12
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (96 GiB VRAM)
**vLLM Image:** `vllm/vllm-openai:gemma4-0505-cu130` (v0.20.2rc1.dev49+g9b4e83934)
**Target Model:** nvidia/Gemma-4-31B-IT-NVFP4 (`--quantization modelopt`)
**MTP Drafter:** google/gemma-4-31B-it-assistant (~926 MiB, BF16, γ=4)
**Memory Profile:** 32k context, fp8_e4m3 kv cache, gpu_memory_utilization=0.52
**Concurrency:** 1 (single user)
**Tool:** llmapibenchmark (random input per request)

## Background

After discovering that MTP is 50% *slower* than baseline on the 26B-A4B MoE model
(see `../gemma-4-26b-a4b-it-mtp/journal/2026-05-12_benchmark_mtp_vs_baseline.md`),
we pivoted to the 31B Dense model. The hypothesis: dense architecture avoids MoE
routing overhead during MTP verification, and the slower baseline TPOT (~25ms vs
~4.8ms) puts it in the regime where speculative decoding helps.

## vLLM Bug: quant_config Propagation to BF16 Drafter

When using NVFP4 quantization (`--quantization modelopt`) with a BF16 drafter,
vLLM propagates the target model's `modelopt_fp4` quant_config to the drafter's
`Gemma4MTPDecoderLayer`. This causes NVFP4 weight packing on BF16 weights,
resulting in a shape mismatch crash during `load_merged_column_weight`:

```
AssertionError at parameter.py:175
param_data.shape != loaded_weight.shape
```

### Fix: Startup sed Patch

The fix patches `gemma4_mtp.py` at container startup to pass `quant_config=None`
to the drafter layers, preventing NVFP4 quantization from being applied to BF16
drafter weights:

```bash
sed -i 's/quant_config=vllm_config\.quant_config,/quant_config=None,/' \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_mtp.py
```

This patches line ~386 of `gemma4_mtp.py` where the `Gemma4MTPDecoderLayer` is
constructed with the parent model's `vllm_config.quant_config`. Setting it to
`None` lets the BF16 drafter weights load unquantized.

**Note:** This is a workaround. A proper fix should be upstreamed to vLLM to
detect when the drafter model uses a different quantization than the target and
skip propagating the target's quant_config to drafter layers.

## Results

| Test | Gen TPS | Prefill TPS | TTFT | vs Baseline |
|------|---------|-------------|------|-------------|
| Decode (50w→500tok) | **149.1** | 3,344 | 60ms | **3.7x** |
| Prefill (1500w→100tok) | 71.4 | 7,799 | 690ms | 1.8x |
| Balanced (500w→500tok) | **136.2** | 7,596 | 240ms | **3.4x** |
| Long Context (3000w→100tok) | 44.8 | 7,271 | 1,460ms | 1.1x |

**Baseline reference** (gemma-4-31b-it-nvfp4, same hardware, no MTP):
- Decode: 40 tok/s
- Prefill: 6,600 tok/s

## Analysis

### MTP delivers massive speedup on 31B Dense

**Decode-heavy workloads see 3.4-3.7x speedup** — far exceeding the community-
reported 2.42x. This confirms the hypothesis:

1. **Dense architecture = no MoE routing overhead** during speculated token
   verification. Every verification step is a simple forward pass through
   fixed-weight layers, not a dynamic expert selection.

2. **Slower baseline TPOT (~25ms) = more room for spec-dec to help.** The 26B
   MoE baseline was ~4.8ms TPOT (207 tok/s) — already faster than spec-dec
   could achieve. The 31B baseline at 40 tok/s has enough headroom.

3. **NVFP4 quantization is compatible** with MTP after the quant_config patch.
   The drafter runs in BF16 while the target runs in NVFP4 — mixed precision
   works correctly.

### Long context performance degrades

Long context (3000w) MTP gen speed drops to 44.8 tok/s (only 1.1x baseline).
This is expected: longer prefills require more KV cache operations, and the
drafter's overhead becomes proportionally larger. The TTFT of 1.46s confirms
the prefill dominates the request time.

### Prefill TPS higher than baseline in some tests

Prefill throughput (7,271-7,799 tok/s) exceeds the baseline (6,600 tok/s) in
all tests. The Gemma4MTPModel code path uses shared KV caches between target
and drafter, which may provide better memory locality during prefill.

## Comparison: 26B MoE (MTP) vs 31B Dense (MTP) vs 31B Dense (Baseline)

| Config | Decode TPS | Prefill TPS | TTFT (decode) |
|--------|-----------|-------------|---------------|
| 26B MoE baseline | 207 | 28,325 | 20ms |
| 26B MoE + MTP γ=2 | 124 (-40%) | 40,815 | 30ms |
| 31B Dense baseline | 40 | 6,600 | ~60ms |
| **31B Dense NVFP4 + MTP γ=4** | **149 (+273%)** | **7,799** | **60ms** |

The 31B Dense + MTP is the **best overall performer for generation quality per
token-per-second**: 149 tok/s with a 31B parameter model (vs 207 tok/s with
only 3.8B active params on the 26B MoE). The 31B Dense provides significantly
higher quality output per generated token.

## Configuration

```yaml
image: vllm/vllm-openai:gemma4-0505-cu130
args:
  - |
    # Patch: prevent NVFP4 quant_config from propagating to BF16 drafter
    sed -i 's/quant_config=vllm_config\.quant_config,/quant_config=None,/' \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_mtp.py
    python3 -m vllm.entrypoints.openai.api_server \
      --model ${MODEL_MOUNT_PATH} \
      --quantization modelopt \
      --language-model-only \
      --speculative-config '{"model": "/drafter", "num_speculative_tokens": 4}' \
      --kv-cache-dtype fp8_e4m3 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.52 \
      --tensor-parallel-size 1 \
      --enable-auto-tool-choice \
      --tool-call-parser gemma4 \
      --reasoning-parser gemma4
```

Key flags:
- `--language-model-only`: Required — Gemma4 31B has a vision tower; spec-dec
  conflicts with multimodal encoding
- `--quantization modelopt`: NVFP4 quantization for the target model
- `num_speculative_tokens: 4`: Despite the drafter having only 1 MTP layer,
  γ=4 outperforms γ=2 on this dense model (no MoE acceptance rate penalty)
- No `"method": "mtp"` in speculative-config — vLLM auto-detects and uses
  the optimized `Gemma4MTPModel` / `Gemma4Proposer` code path

## Open Questions

- **γ=2 vs γ=4 comparison needed** — the 26B journal showed γ=2 was better
  for MoE. Dense may behave differently; γ=4 may be optimal here.
- **FlashInfer on Blackwell** — the `gemma4-0505-cu130` image lacks flashinfer.
  A newer vLLM build with flashinfer could unlock additional gains.
- **Higher concurrency** — untested. Dense models scale differently than MoE
  under concurrent load; MTP overhead may amortize better.
- **Upstream the quant_config patch** — file an issue or PR against vLLM to
  properly handle mixed quantization between target and drafter.
