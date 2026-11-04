# 2026-04-25 — Diffusers Community Weights Failed

**Goal:** Load LTX-2.3 distilled weights via diffusers LTX2Pipeline.

## What Happened

### Community Conversion: CalamitousFelicitousness/LTX-2.3-distilled-Diffusers

Downloaded 68 GiB of diffusers-format weights. Loading failed with shape mismatches.

**Error 1: Transformer audio_scale_shift_table**
```
ValueError: transformer_blocks.0.audio_scale_shift_table expected shape [6, 2048], but got [9, 2048]
```
- **Root cause:** Config used wrong field names (`video_gated_attn`, `video_cross_attn_adaln`)
  instead of what diffusers expects (`gated_attn`, `cross_attn_mod`)
- **Fix attempted:** Renamed config fields to match diffusers expectations
- **Result:** Fixed this error, but revealed a second error

**Error 2: VAE decoder upsampler**
```
ValueError: decoder.up_blocks.0.upsamplers.0.conv.conv.bias expected shape [2048], but got [4096]
```
- **Root cause:** VAE config has different `upsample_type` ordering and `upsample_residual`
  values compared to the HF-staff-maintained dg845 conversion
- **Unfixable:** The VAE weights themselves are structured for a different architecture
  variant than what diffusers main supports. The conversion was done for an unreleased
  diffusers PR/fork.

### Why The Community Conversion Is Broken

The CalamitousFelicitousness conversion was created against a development branch of diffusers
that includes LTX-2.3-specific architectural changes not yet merged to main:
- Different `upsample_type` ordering in VAE (reversed)
- `upsample_residual: true` vs `false`
- Different config field naming conventions
- Diffusers 0.38.0.dev0 from main still has the mismatch

### Alternative: dg845/LTX-2.3-Diffusers (HF Staff)

This conversion (maintained by HF staff member YiYiXu) has working configs, but:
- Only has the **dev** model (not distilled) — 94 GiB, 40 inference steps
- Text encoder is 45 GiB (appears to be FP32)
- Too large for our use case (we want the fast 8-step distilled variant)

## Decision: Switch to Official ltx-pipelines

The official Lightricks `ltx-pipelines` Python package:
- Loads raw `.safetensors` checkpoints directly (no diffusers config translation)
- Supports FP8 quantization natively (`--quantization fp8-cast`)
- Has a clean Python API and CLI
- Maintained by Lightricks (the model authors)
- Works with the official distilled checkpoint

## Cleanup

- Deleted 68 GiB of CalamitousFelicitousness community weights
- Started downloading official Lightricks checkpoint files:
  - `ltx-2.3-22b-distilled-1.1.safetensors` (43 GiB)
  - `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (7 GiB)
  - `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` (0.9 GiB)

## Lessons Learned

1. Community diffusers conversions can have config mismatches with diffusers main
2. Config field naming matters — wrong names cause silent "unexpected config" warnings,
   then architecture shape mismatches at load time
3. The "official" diffusers support for LTX-2.3 is not yet fully landed
4. For bleeding-edge models, the author's own inference code is the safest bet
