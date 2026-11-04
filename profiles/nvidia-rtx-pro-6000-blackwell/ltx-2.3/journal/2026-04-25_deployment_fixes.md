# 2026-04-25 — Deployment Fixes

**Goal:** Get LTX-2.3 ComfyUI container running in K8s with our model files.

## Issue 1: Volume Mount Interferes with Container Startup

- **Error:** Container startup script (`/start.sh`) copies ComfyUI to `/workspace/ComfyUI/` via `comfyui-on-workspace.sh`. Mounting a hostPath volume at `/workspace/ComfyUI/models` created the directory structure early, causing the copy script to either skip or fail.
- **Root cause:** The container is designed for RunPod where `/workspace/` starts empty. Our volume mount pre-created `/workspace/ComfyUI/` which confused the copy logic.
- **Fix:** Mount models at `/mnt/models` instead of `/workspace/ComfyUI/models`. Use a background watcher script to inject `extra_model_paths.yaml` into `/workspace/ComfyUI/` once the directory exists (after `comfyui-on-workspace.sh` creates it). ComfyUI reads this config at startup and discovers models at `/mnt/models/`.

### extra_model_paths.yaml Configuration

```yaml
ltx_models:
  base_path: /mnt/models
  diffusion_models: diffusion_models
  vae: vae
  clip: text_encoders
  loras: loras
  upscale_models: latent_upscale_models
```

### Timing

The `/start.sh` script has a ~10s gap between creating `/workspace/ComfyUI/` (line 60) and starting ComfyUI (line 129). The background watcher injects the yaml within ~1s of directory creation, well before ComfyUI reads it.

## Issue 2: Readiness Probe Returns 401 (Unauthorized)

- **Error:** K8s httpGet readiness probe on `GET /` port 8188 received HTTP 401. Pod never became Ready.
- **Root cause:** ComfyUI has built-in authentication that requires a password/token. All HTTP endpoints return 401 for unauthenticated requests.
- **Fix:** Changed readiness probe from `httpGet` to `tcpSocket` on port 8188. This checks that the port is accepting connections without needing authentication.

## Issue 3: ComfyUI Authentication Required for API Access

- **Status:** UNRESOLVED — ComfyUI API requires authentication token
- **Impact:** Cannot call ComfyUI API (e.g., `/prompt`, `/system_stats`) without setting up auth
- **Possible fixes:**
  1. Set `COMFYUI_EXTRA_ARGUMENTS` env var to include auth-bypass flags
  2. Configure a known token via ComfyUI's user settings
  3. Use the ComfyUI Manager config to set `security_level` to `weak` or `none`
- **Note:** The "Please set up your password" messages in logs suggest first-run auth setup is needed

## Container Image Details

- **Image:** `ls250824/run-comfyui-ltx:25042026`
- **Entrypoint:** `/start.sh` (bash script, ~660 lines)
- **Startup sequence:**
  1. GPU detection (nvidia-smi)
  2. Copy ComfyUI to `/workspace/ComfyUI/` via `comfyui-on-workspace.sh`
  3. Copy additional files via `files-on-workspace.sh`, `test-on-workspace.sh`, `docs-on-workspace.sh`
  4. Start code-server on port 9000
  5. CUDA availability check
  6. Start ComfyUI on port 8188 with args from `$COMFYUI_EXTRA_ARGUMENTS`
  7. Wait for ComfyUI to respond
  8. HuggingFace/CivitAI model provisioning (skipped since we pre-mount)
  9. Workflow provisioning
  10. `sleep infinity`
- **Python:** 3.11.14
- **PyTorch:** 2.9.1+cu128
- **CUDA:** 12.8

## Verified Working

- Pod starts and reaches 1/1 Ready state
- ComfyUI detects all mounted model files at `/mnt/models/`:
  - `diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors`
  - `vae/LTX23_audio_vae_bf16.safetensors`, `LTX23_video_vae_bf16.safetensors`, `taeltx2_3.safetensors`
  - `text_encoders/gemma_3_12B_it_fp8_scaled.safetensors`, `ltx-2.3_text_projection_bf16.safetensors`
  - `loras/ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors`
  - `latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- GPU detected: NVIDIA RTX PRO 6000 Blackwell (95 GiB VRAM, High VRAM mode)

## Next Steps

1. Resolve ComfyUI authentication for API access
2. Test video generation via ComfyUI workflow API
3. Measure actual VRAM usage during inference
