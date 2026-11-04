# 2026-07-07 -- Three-tier (NVFP4+FP8) quantization of Apertus v1.5 70B: design and plan

**Goal:** Port the 8B three-tier mixed-precision design to the 70B, entirely on-node
(RTX 6000 Blackwell 96 GB, 64 GB RAM), using llm-compressor instead of ModelOpt.

## Context

- BF16 source: `apertus-ai/Apertus-v1.5-70B-Prerelease-2607` (145.6 GB, 3 shards).
  Support files (config/tokenizer) were missing from the original download and were
  fetched per-file afterwards. Root cause: the repository file listing used for the
  download returned only weight files (support files were never added), so the
  original download request never included configs.
- An FP8_DYNAMIC baseline was produced first (2026-07-06) via llm-compressor 0.12.0
  `model_free_ptq`: 71.9 GiB, 1.8 min conversion, ~20 tok/s decode on vLLM
  v0.19.1-cu130, 8.87 GiB KV at 16k. Located at
  `/persistent/dev/apertus-70b-fp8-conversion/output/apertus-v15-70b-fp8-dynamic`.
- Motivation for three-tier: FP8 weights (72 GB) leave only ~9 GiB KV on a 96 GB
  card. Three-tier (~53 GB estimated) should roughly triple KV headroom.

## Design (mirrors the 8B three-tier, adapted to llm-compressor)

| Tier | Layers | Scheme | Rationale |
|------|--------|--------|-----------|
| NVFP4 | `mlp.up_proj`, `mlp.down_proj` x 80 layers | NVFP4A16 (weight-only) | 8B lesson: MLP NVFP4 is safe. Weight-only keeps the recipe data-free (W4A4 activation global scales would need calibration) |
| FP8 | `self_attn.{q,k,v,o}_proj` x 80 layers | FP8_DYNAMIC | 8B lesson: full NVFP4 attention produced NaN logits (xIELU x^2 amplifies FP4 noise); FP8 attention is proven safe |
| BF16 | `lm_head`, `embed_tokens`, norms, xIELU params | ignored | 266k vocab head is too sensitive; norms/xIELU too small to matter |

Tooling: llm-compressor non-uniform quantization (config_groups, supported since
v0.7.0), producing a compressed-tensors `mixed-precision` checkpoint.
Reference: examples/quantization_non_uniform/quantization_nvfp4_fp8.py upstream.

**Runtime constraint:** vLLM only. SGLang does not support the compressed-tensors
mixed-precision format (sgl-project/sglang#16276). The 8B three-tier card runs
ModelOpt format on SGLang; this 70B card will run compressed-tensors on vLLM.

## Differences vs the 8B run

1. ModelOpt -> llm-compressor. The 8B FP8 journal already showed llm-compressor
   compressed-tensors + vLLM hits the fast CUTLASS path (1.6x decode vs ModelOpt/SGLang).
2. `model_free_ptq` (used for the FP8 baseline) cannot express mixed schemes,
   so this uses the `oneshot` recipe pathway, which loads the model as a module.
   146 GB does not fit in 64 GB RAM -> accelerate disk offload + llm-compressor
   sequential onloading. Data-free recipe = no calibration forward passes.
3. The 8B `gate_proj` crash (llm-compressor fused-layer code assumed LLaMA MLPs)
   appears refactored away in 0.12.0 (no gate_proj references in
   modifiers/quantization/calibration.py or modeling/fuse.py at tag 0.12.0).
   If any remnant survives, Apertus (no gate_proj) will crash it immediately.

## Verification plan

1. vLLM loads the checkpoint, no NaN in generations (smoke prompts).
2. Logit comparison vs the FP8_DYNAMIC checkpoint (NOT vs BF16 -- the BF16 model
   cannot run on this node). Justification: at 8B, FP8 measured cos_sim 0.999 vs
   BF16, so FP8 serves as a near-lossless proxy baseline.
3. Measure VRAM, KV headroom, decode/prefill TPS; compare against the FP8 numbers.

## Known risks

- 8B data point: three-tier was the WORST quality variant (cos_sim 0.979,
  kl_div 0.755 vs BF16) and was never promoted beyond experimental. 70B models
  typically tolerate 4-bit MLPs better (the third-party v1 70B NVFP4 W4A4 entry
  works), but this must be measured, not assumed.
- oneshot + disk offload on a 146 GB model is the least-proven part of the plan.
  Expected runtime 1-3 h, I/O bound.

## Outcome

Design documented. Implementation next (see following journal entries).
