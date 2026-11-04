# 2026-04-25 — Initial Catalog Entry

**Goal:** Add Lightricks LTX-2.3 (22B) audio-video generation model to the catalog.

## Model Research

- **Official repo:** Lightricks/LTX-2.3 (HuggingFace)
- **Type:** DiT-based (Diffusion Transformer) audio-video foundation model
- **Parameters:** 22B
- **Capabilities:** text-to-video, image-to-video, video-to-video, audio generation
- **License:** ltx-2-community-license-agreement (custom, NOT Apache/MIT)
- **Architecture:** Diffusion Transformer with synchronized audio+video generation

## Key Differences from LLM Catalog Entries

This is NOT an LLM. Major differences:
1. No OpenAI-compatible API — served via ComfyUI (workflow-based)
2. No vLLM/SGLang/llama.cpp support
3. Output is MP4 video files, not text tokens
4. Requires multiple model components (transformer, VAE, text encoder, LoRA)
5. Components spread across 3 HuggingFace repos

## Weight Sources

Using community-repackaged ComfyUI-compatible weights:

**Primary: Kijai/LTX2.3_comfy** (downloaded via catalog API)
| File | Size | Purpose |
|------|------|---------|
| diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors | 23.49 GiB | Main transformer (FP8) |
| vae/LTX23_audio_vae_bf16.safetensors | 0.34 GiB | Audio VAE |
| vae/LTX23_video_vae_bf16.safetensors | 1.35 GiB | Video VAE |
| vae/taeltx2_3.safetensors | 0.02 GiB | Preview VAE |
| text_encoders/ltx-2.3_text_projection_bf16.safetensors | 2.15 GiB | Text projection |
| loras/ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors | 2.55 GiB | Distilled LoRA |

**Separate download: Comfy-Org/ltx-2** (text encoder)
| File | Size | Purpose |
|------|------|---------|
| split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors | 12.30 GiB | Gemma 3 12B text encoder (FP8) |

**Separate download: Lightricks/LTX-2.3** (upscaler)
| File | Size | Purpose |
|------|------|---------|
| ltx-2.3-spatial-upscaler-x2-1.1.safetensors | 0.93 GiB | 2x spatial upscaler |

**Total: ~43 GiB (FP8 versions)**

## Container Image

Using `ls250824/run-comfyui-ltx:25042026`:
- ComfyUI-based serving environment
- Includes Lightricks/ComfyUI-LTXVideo custom nodes
- Designed for RunPod but should work in K8s
- Port 8188 (ComfyUI default)
- GitHub: jalberty2018/run-comfyui-ltx

## ComfyUI Model Directory Structure

Models mount at `/workspace/ComfyUI/models/` with subdirectories:
```
models/
  diffusion_models/   — transformer weights
  vae/                — audio + video VAE
  text_encoders/      — Gemma text encoder + projection
  loras/              — distilled LoRA, control LoRAs
  latent_upscale_models/ — spatial/temporal upscalers
```

## Configuration Decisions

- **FP8 weights** chosen over BF16 to fit within 96 GiB VRAM alongside other models
- **Distilled v1.1** chosen — recommended version, only 8 inference steps needed
- **ComfyUI** as serving layer — only practical option, has built-in LTXVideo support
- **envoy routing** — non-OpenAI API (workflow-based ComfyUI API)

## Known Challenges

1. **Multi-repo downloads** — our catalog API only supports single-repo downloads. Text encoder and upscaler need manual download.
2. **ComfyUI API** — workflow-based (POST /prompt with JSON workflow), not simple REST
3. **Memory management** — ComfyUI manages model loading/offloading dynamically
4. **VRAM estimates are rough** — actual usage depends on resolution and frame count
5. **Container image** — community-maintained, designed for RunPod, may need adaptation for our K8s setup

## Outcome

Catalog entry created. Next steps:
1. Download primary files from Kijai/LTX2.3_comfy
2. Manually download Gemma text encoder from Comfy-Org/ltx-2
3. Manually download upscaler from Lightricks/LTX-2.3
4. Deploy and test ComfyUI serving
5. Figure out model directory mount mapping
6. Test video generation via ComfyUI API
