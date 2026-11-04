# 2026-07-07 -- Two-pass three-tier quantization SUCCEEDED; benchmarks vs FP8

## Run summary (attempt 4)

Container: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime + `pip install
llmcompressor` (0.12.0), memory-capped at 30 GB, GPU freed (qwen36 undeployed
via app API for the duration).

Fixes accumulated across attempts (each has its own journal entry or note):
1. `pip uninstall torchvision torchaudio` after installing llmcompressor
   (llmcompressor upgrades torch to 2.12, stranding the image's torchvision;
   transformers then fails to import).
2. `disable_fused_global_scale_mappings()` runtime patch -- llm-compressor's
   microscale fused-set matcher assumes (q,k,v) and (gate,up) sets; Apertus
   has no gate_proj and its shards split q/k/v across files. See
   2026-07-07_fused_mapping_crash_and_patch.md.
3. `apt-get install gcc` -- the NVFP4 compressor triton-compiles a kernel;
   the -runtime pytorch image ships no C compiler (InductorError). The FP8
   pass never needed this (float compressor does not compile).
4. `max_workers=1` -- shard validation loads a full ~50 GB shard onto the GPU
   per worker (learned during the FP8 baseline run).

Timings: PASS 1 (NVFP4A16 on MLP) 1.9 min -> 60.2 GiB intermediate
(MLP-only NVFP4, kept at
/persistent/dev/apertus-70b-fp8-conversion/output/apertus-v15-70b-nvfp4-mlp-only).
PASS 2 (FP8_DYNAMIC on attention) 0.9 min -> 49.0 GiB final.

## Post-processing: quantization_config surgery (REQUIRED)

`create_or_update_quant_config` merged the two passes into
`format: mixed-precision` with both compressors present, BUT both
config_groups carried `targets: ["Linear"]` and the global `ignore` retained
pass 1's attention patterns. As generated, vLLM would resolve attention
modules as UNQUANTIZED (they match ignore) while their tensors are F8 --
guaranteed load failure or corruption. Fixed by rewriting:

- nvfp4-pack-quantized group -> targets `re:.*mlp\.(up|down)_proj$`
- float-quantized group -> targets `re:.*self_attn\.(q|k|v|o)_proj$`
- ignore -> `["lm_head", "model.embed_tokens"]`

The pre-surgery config is preserved as `config.json.as-generated` in the
checkpoint dir. This mirrors how upstream authors their mixed examples
(per-group regex targets). Candidate upstream issue #2 (besides the fused
mapping crash): two-pass model_free_ptq produces ambiguous targets/ignore.

## Verification results

vLLM v0.19.1-cu130, RTX PRO 6000 Blackwell 96 GB, 16k max-len, util 0.90:

| Metric | Three-tier | FP8 baseline | BF16 |
|--------|-----------|--------------|------|
| Checkpoint size | 49.0 GiB | 71.9 GiB | 145.6 GiB |
| Weights in VRAM | 49.14 GiB | 71.98 GiB | does not fit |
| KV cache at util 0.90 | 31.7 GiB / 103,872 tok | 8.87 GiB / 29,056 tok | -- |
| Decode TPS (conc 1) | 29.7 | 20.2 | -- |
| CUDA graphs | yes | yes | -- |

Decode is FASTER than FP8 because decode is memory-bandwidth bound and the
weights are 32% smaller.

Quality -- teacher-forced comparison against the FP8 checkpoint (189 positions
of fixed mixed-domain text; FP8 is the local stand-in for BF16, which measured
cos_sim 0.999 vs BF16 at 8B):

- top-1 next-token agreement: 94.2%
- mean KL(fp8||three-tier) over top-10: 0.168; median 0.042; max 6.30
- greedy generations: coherent and factually correct on all 5 smoke prompts
  (English science, Python code, French translation, EU AI Act, arithmetic);
  French translation matched FP8's output token-for-token.

Interpretation: distributions track FP8 closely on typical positions with a
small heavy tail (rare positions disagree strongly) -- the expected NVFP4
signature. Better behaved than the 8B three-tier was vs its baseline
(kl_div 0.755 full-vocab), consistent with larger models tolerating 4-bit
MLPs better.

## Checkpoint locations

| Variant | Path |
|---------|------|
| Three-tier (this card) | /persistent/models/onprem-ai--Apertus-v1.5-70B-Prerelease-2607-ThreeTier--model/ |
| FP8_DYNAMIC baseline | /persistent/dev/apertus-70b-fp8-conversion/output/apertus-v15-70b-fp8-dynamic/ |
| NVFP4 MLP-only intermediate | /persistent/dev/apertus-70b-fp8-conversion/output/apertus-v15-70b-nvfp4-mlp-only/ |
| BF16 assembled source | /persistent/dev/apertus-70b-fp8-conversion/bf16-assembled/ (symlinks into /persistent/models) |

## Open items

1. Measure memory profiles beyond 16k (32k/65k/131k are estimates in the card).
2. Benchmark prefill TPS and concurrency scaling.
3. Try vLLM v0.23.0 (the 8B FP8 entry's image) with the mixed format.
4. Try fp8_e4m3 KV cache on top (tested BF16 KV only) -- would add another
   ~2x KV capacity if the mixed checkpoint accepts it.
5. File upstream llm-compressor issues: (a) microscale fused-set matcher
   crashes on gateless MLPs and cross-shard q/k/v; (b) two-pass
   model_free_ptq writes ambiguous targets/ignore.
6. Consider cataloging the FP8 variant as the quality-critical option
   (mirrors the 8B fp8/nvfp4 pairing).
