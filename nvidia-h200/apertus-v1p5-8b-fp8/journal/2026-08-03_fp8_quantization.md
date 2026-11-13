# 2026-08-03 -- FP8 and NVFP4 quantization of swiss-ai/Apertus-v1.5-8B

## Summary

Downloaded swiss-ai/Apertus-v1.5-8B (BF16, 18 GiB, 4 shards) and produced
both FP8 and NVFP4 quantized checkpoints using identical recipes as the 70B.

## Download

Source: swiss-ai/Apertus-v1.5-8B (gated)
Token: a Hugging Face read token with access to the gated repository
Method: snapshot_download (huggingface_hub 1.26.0, max_workers=4)
Target: /persistent/models/swiss-ai/Apertus-v1.5-8B/
Size: 18 GiB (4 model shards of 3.8/4.4/3.6/4.4G + vision 833M + audio 292M)
Duration: ~4 min
Files: 21 (including PDFs, assets, configs)

Architecture: Apertus1p5ForConditionalGeneration, model_type=apertus1p5,
hidden_size=4096, num_hidden_layers=32, vocab_size=266752, output_vocab_size=131072.
Same structure as 70B: text_config sub-object, model.language_model.* prefix.

## Quant environment

venv: /tmp/quant-venv-8b/ (created 2026-08-03)
llm-compressor: 0.12.0
torch: 2.12.0+cu130
cuda: 13.2
python: 3.12.3

## FP8 quantization

Script: quantize_apertus_v15_8b_fp8.py (single-pass FP8_DYNAMIC)

Input:  /persistent/models/swiss-ai/Apertus-v1.5-8B/ (18 GiB BF16)
Output: /persistent/models/onprem-ai--Apertus-v1.5-8B-FP8--model-apertus-model/ (10.7 GiB)
Duration: 0.2 min
Scheme: FP8_DYNAMIC, max_workers=1, device=cuda:0
Ignore: lm_head, model.language_model.embed_tokens, model.embed_tokens,
        re:.*vision_tokenizer.*, re:.*audio_tokenizer.*

Output: 10.7 GiB (4 shards of 1.9/3.3/2.3/2.2G + vision 833M + audio 292M)
Compression: ~1.7x vs BF16

Config: format=float-quantized, targets=["Linear"]
Note: FP8 single-pass does NOT require config surgery (only two-pass NVFP4 does).

## NVFP4 quantization

Script: quantize_apertus_v15_8b_nvfp4.py (two-pass NVFP4+FP8)

Pass 1: NVFP4A16 on MLP (1.9 min)
  Input:  /persistent/models/swiss-ai/Apertus-v1.5-8B/ (18 GiB BF16)
  Output: /persistent/models/...-pass1-mlp-only/ (9.6 GiB intermediate)
  Scheme: NVFP4A16 on mlp.up_proj/down_proj, ignore attention + always-ignored

Pass 2: FP8_DYNAMIC on attention (0.1 min)
  Input:  ...-pass1-mlp-only/ (9.6 GiB)
  Output: /persistent/models/onprem-ai--Apertus-v1.5-8B-NVFP4--model-apertus-model/ (8.4 GiB)
  Scheme: FP8_DYNAMIC on attention q/k/v/o_proj, ignore MLP + always-ignored

Always ignored: lm_head, model.language_model.embed_tokens, model.embed_tokens,
                re:.*vision_tokenizer.*, re:.*audio_tokenizer.*

Critical: disable_fused_global_scale_mappings() -- same Apertus gate_proj issue.

Total duration: 0.4 min (9.1 GiB saved, ~2.1x compression vs BF16)
Intermediate pass1 deleted after successful pass2.

### Post-processing: quantization_config surgery (REQUIRED)

Applied fix_three_tier_quant_config.py (same as 70B):
- nvfp4-pack-quantized group -> targets = MLP up/down regex
- float-quantized group -> targets = attention q/k/v/o regex
- ignore -> ["lm_head", "model.embed_tokens"]
Backup: config.json.as-generated

## Pre-upload preparation

- Chat templates replaced with curated jinja (same as 70B repos, md5 9f851113)
- README.md written for each repo (modeled on 70B READMEs)
- Configs verified: FP8 has float-quantized, NVFP4 has mixed-precision with correct groups
- All safetensors, tokenizer files, configs, legal docs present
- config.json.as-generated kept in NVFP4 dir for reference

## Disk state after quantization

- /persistent/models/swiss-ai/Apertus-v1.5-8B/ (18G, source BF16)
- /persistent/models/onprem-ai--Apertus-v1.5-8B-FP8--model-apertus-model/ (11G)
- /persistent/models/onprem-ai--Apertus-v1.5-8B-NVFP4--model-apertus-model/ (8.4G)
Total: ~37 GiB used for 8B quant artifacts

## Upload targets

- onprem-ai/Apertus-v1.5-8B-FP8 (currently 1 file, empty placeholder)
- onprem-ai/Apertus-v1.5-8B-NVFP4 (currently 1 file, empty placeholder)

Plan: use hf upload-large-folder with HF_HUB_DISABLE_XET=1 per the 70B journal.

## Catalog scripts added

- catalog/rtx6k/apertus-v1p5-8b-nvfp4/quantize_apertus_v15_8b_fp8.py
- catalog/rtx6k/apertus-v1p5-8b-nvfp4/quantize_apertus_v15_8b_nvfp4.py
- catalog/rtx6k/apertus-v1p5-8b-nvfp4/fix_three_tier_quant_config.py
- catalog/rtx6k/apertus-v1p5-8b-nvfp4/teacher_forced_compare.py
- catalog/rtx6k/apertus-v1p5-8b-nvfp4/chat_template.jinja
- catalog/rtx6k/apertus-v1p5-8b-nvfp4/journal/2026-08-03_fp8_and_nvfp4_quantization.md
