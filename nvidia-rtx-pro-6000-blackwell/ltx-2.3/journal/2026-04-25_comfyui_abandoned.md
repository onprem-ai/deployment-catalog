# 2026-04-25 — ComfyUI Approach Abandoned

**Goal:** Document why ComfyUI was abandoned in favor of a diffusers-based serving approach.

## What Worked

- Container image `ls250824/run-comfyui-ltx:25042026` started successfully
- GPU detected (RTX PRO 6000 Blackwell, 95 GiB VRAM, CUDA 12.8, PyTorch 2.9.1)
- Model files mounted at `/mnt/models` via `extra_model_paths.yaml` injection
- ComfyUI discovered all model components correctly:
  - `diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors`
  - `vae/` (audio, video, preview)
  - `text_encoders/` (Gemma 3 12B FP8 + text projection)
  - `loras/` (distilled LoRA)
  - `latent_upscale_models/` (spatial upscaler)
- Pod reached 1/1 Ready state with `tcpSocket` readiness probe

## Problems Encountered

### 1. Volume Mount Conflict with Startup Script
- `/start.sh` copies ComfyUI from `/ComfyUI` to `/workspace/ComfyUI/` at startup
- Mounting hostPath at `/workspace/ComfyUI/models` pre-created the directory, breaking the copy
- **Fix**: Mount at `/mnt/models` + inject `extra_model_paths.yaml` via background watcher

### 2. Readiness Probe 401 Unauthorized
- `httpGet` probe on `/` returned 401 due to authentication middleware
- **Fix**: Changed to `tcpSocket` probe on port 8188

### 3. ComfyUI-Login Custom Node (BLOCKING)
- The `ComfyUI-Login` custom node installs an aiohttp middleware that blocks ALL HTTP requests
- Requires password setup at `/login` before any API access
- Middleware loads at Python import time — renaming/removing the directory after import has no effect
- Renaming before container copy (`comfyui-on-workspace.sh`) only works on first run
- On subsequent runs, the persisted `/workspace/ComfyUI/` already has the node
- Even renaming to `.disabled-Login` didn't help — ComfyUI loads hidden directories too
- The middleware intercepts every request and returns 401 for unauthenticated calls
- Bearer token auth requires the PASSWORD file to exist first (chicken-and-egg)

### 4. Fundamental Design Mismatch
- ComfyUI is designed for interactive GUI use, not headless API serving
- The container (8.6 GiB) includes code-server, dozens of unneeded custom nodes
- API requires constructing complex 50+ node workflow JSON graphs
- No simple "text in, video out" endpoint

## Decision: Switch to Diffusers

**Reasons:**
1. `diffusers >= 0.37.0` has native `LTX2Pipeline` support
2. Simple Python API: `pipe(prompt) → (video, audio)`
3. No authentication hassle
4. No 50-node workflow graphs
5. Built-in FP8 layerwise casting for VRAM efficiency
6. Built-in CPU offloading for memory management
7. Lightweight — just PyTorch + diffusers, no ComfyUI overhead

**Trade-offs:**
- Need to download new weights in diffusers format (~67 GiB BF16)
- Community-converted weights (CalamitousFelicitousness/LTX-2.3-distilled-Diffusers)
- Kijai ComfyUI-format weights (44 GiB) are NOT compatible with diffusers

## Cleanup

- Deleted the ComfyUI deployment
- Deleted 44 GiB of Kijai ComfyUI-format weights from `/persistent/models/`
- Started download of diffusers-format weights from HuggingFace

## Diffusers Weight Sizes

| Component | Size (BF16) | Purpose |
|-----------|-------------|---------|
| transformer (4 shards) | 35.4 GiB | Main 22B DiT model |
| text_encoder (5 shards) | 22.7 GiB | Gemma 3 12B |
| connectors | 5.9 GiB | Text→video/audio bridges |
| vae | 1.4 GiB | Video VAE |
| latent_upsampler | 0.9 GiB | 2x spatial upscaler (optional) |
| vocoder | 0.2 GiB | Audio synthesis |
| audio_vae | 0.1 GiB | Audio VAE |
| **Total** | **~66.7 GiB** | |

Note: With FP8 layerwise casting, VRAM usage is roughly halved during inference.
