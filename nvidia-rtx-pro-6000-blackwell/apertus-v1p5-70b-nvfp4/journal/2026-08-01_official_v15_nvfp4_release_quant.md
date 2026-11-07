# 2026-08-01 -- NVFP4 quant of swiss-ai/Apertus-v1.5-70B (official release)

## Summary

Downloaded the official `swiss-ai/Apertus-v1.5-70B` BF16 release (135 GiB,
2026-07-24, Apertus1p5ForConditionalGeneration architecture) and applied the
same two-pass model_free_ptq recipe from the RC (2026-07-13) to produce a
48 GiB NVFP4 (three-tier mixed-precision) checkpoint.

Total wall time: ~12 min (download) + 3.4 min (quantization).

## Changes vs RC quant script

The official v1.5 release differs from `Apertus-v1.5-70B-RC-2607` in two
ways that broke the old `quantize_apertus_70b_three_tier.py`:

1. **Architecture prefix `model.language_model.*`**. The RC used bare
   `ApertusForCausalLM` where `embed_tokens.weight` lived at
   `model.embed_tokens`. The official `Apertus1p5ForConditionalGeneration`
   wraps the transformer under `model.language_model.*` so
   `embed_tokens` is at `model.language_model.embed_tokens`. The old
   ignore entry `"model.embed_tokens"` no longer matched, and the NVFP4
   compressor tried to quantize the 8.14 GiB lm_head's workspace tensor,
   causing OOM (31-51 GiB stale PyTorch allocation from prior crash +
   workspace).

2. **Vision/audio tokenizer shards included inline.** The official release
   ships `model-vision_tokenizer-model.safetensors` (833 MiB) and
   `model-wavtokenizer-model.safetensors` (292 MiB) as separate shard files.
   Their conv/LSTM/codebook weights are 4D, not 2D Linear, and cause
   `validate_weight_for_quantization` to crash. Fixed by adding
   `"re:.*vision_tokenizer.*"` and `"re:.*audio_tokenizer.*"` to the ignore
   list.

New script: `quantize_apertus_v15_70b_nvfp4_vllm.py` in this directory.

## Download

Source: `swiss-ai/Apertus-v1.5-70B` (gated, access granted 2026-08-01)
Token: a Hugging Face read token with access to the gated repository
Method: `snapshot_download` via `huggingface_hub` (max_workers=4, resume)
Target: `/persistent/models/swiss-ai/Apertus-v1.5-70B/`
Size: 135 GiB (3 shards of ~47/44/44 GiB + vision/audio tokenizers)

## Quantization

Environment: `/tmp/quant-venv/` with `llmcompressor==0.12.0`
GPU: NVIDIA RTX PRO 6000 Blackwell (96 GiB VRAM), fully free
RAM: 62 GiB total, ~49 GiB available at start

Key parameters:
- `max_workers=1`
- `device="cuda:0"`
- `disable_fused_global_scale_mappings()` (Apertus has no gate_proj)

### Pass 1: NVFP4A16 on MLP (1.9 min)

```
input:  /persistent/models/swiss-ai/Apertus-v1.5-70B/
output: /persistent/models/swiss-ai/Apertus-v1.5-70B-NVFP4-pass1-mlp-only/
scheme: NVFP4A16
ignore: lm_head, model.language_model.embed_tokens, model.embed_tokens,
        re:.*vision_tokenizer.*, re:.*audio_tokenizer.*,
        re:.*self_attn\.(q|k|v|o)_proj$
size:   59.2 GiB
```

### Pass 2: FP8_DYNAMIC on attention (1.3 min)

```
input:  /persistent/models/swiss-ai/Apertus-v1.5-70B-NVFP4-pass1-mlp-only/
output: /persistent/models/swiss-ai/Apertus-v1.5-70B-NVFP4/
scheme: FP8_DYNAMIC
ignore: lm_head, model.language_model.embed_tokens, model.embed_tokens,
        re:.*vision_tokenizer.*, re:.*audio_tokenizer.*,
        re:.*mlp\.(up|down)_proj$
size:   48.0 GiB
```

### Quantization_config surgery (required)

Per the RC journal (2026-07-07_two_pass_success_and_benchmarks.md),
`model_free_ptq` writes both config_groups with `targets: ["Linear"]` and
leaves pass-1's attention patterns in the global ignore list. This would
cause vLLM to resolve FP8 attention tensors as unquantized.

Applied `fix_three_tier_quant_config.py` which rewrites:
- `nvfp4-pack-quantized` group -> targets = MLP up/down regex
- `float-quantized` group -> targets = attention q/k/v/o regex
- ignore -> `["lm_head", "model.embed_tokens"]`

Backup preserved as `config.json.as-generated`.

## Output

Path: `/persistent/models/swiss-ai/Apertus-v1.5-70B-NVFP4/`
Size: 48 GiB (2.8x compression vs 135 GiB BF16)
Shards: 5 (3 model + 2 tokenizer copies, same structure as input)
Config: mixed-precision, compressed-tensors format

## Notes

- The intermediate pass-1 output at `...-NVFP4-pass1-mlp-only/` is 59 GiB and
  can be deleted after confirming the final checkpoint loads correctly.
- This checkpoint is based on the official v1.5 release, NOT the RC.
  Architecture: `Apertus1p5ForConditionalGeneration` (vs RC's
  `ApertusForCausalLM`). The `--hf-overrides` hack is no longer needed at
  serving time.
- The final checkpoint name is `Apertus-v1.5-70B-NVFP4` -- the "three-tier"
  naming is internal only. "NVFP4" is what people search for.
- NOT uploaded to HuggingFace (pending operator approval).
- vLLM only: SGLang does not support compressed-tensors mixed-precision
  checkpoints (sgl-project/sglang#16276).
