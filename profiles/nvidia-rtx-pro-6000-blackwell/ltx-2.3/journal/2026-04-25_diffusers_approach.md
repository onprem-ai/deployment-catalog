# 2026-04-25 — Diffusers Approach

**Goal:** Switch from ComfyUI to Hugging Face diffusers for LTX-2.3 serving.

## Why Diffusers

ComfyUI was abandoned (see `2026-04-25_comfyui_abandoned.md`) due to:
- Unresolvable authentication middleware in ComfyUI-Login custom node
- Complex 50+ node workflow JSON required for API calls
- 8.6 GiB bloated container with code-server, dozens of unneeded custom nodes
- Designed for interactive GUI use, not headless API serving

Diffusers provides:
- Native `LTX2Pipeline` class (diffusers >= 0.37.0)
- Simple Python API: `pipe(prompt) → (video, audio)`
- Built-in FP8 layerwise casting for VRAM efficiency
- Built-in CPU offloading for memory management
- Clean, minimal dependencies

## Diffusers API Reference

### Pipeline Class
```python
from diffusers import LTX2Pipeline
pipe = LTX2Pipeline.from_pretrained(
    "CalamitousFelicitousness/LTX-2.3-distilled-Diffusers",
    torch_dtype=torch.bfloat16,
)
```

### Text-to-Video (Minimal)
```python
video, audio = pipe(
    prompt="A cat walking on a beach at sunset",
    negative_prompt="worst quality, blurry, jittery, distorted",
    width=768,
    height=512,
    num_frames=121,       # (N*8)+1
    frame_rate=24.0,
    num_inference_steps=8, # distilled model: 8 steps
    guidance_scale=1.0,    # distilled model: CFG=1
    output_type="np",
    return_dict=False,
)
```

### LTX-2.3 Specific Parameters
```python
video, audio = pipe(
    prompt=prompt,
    negative_prompt="worst quality, blurry, jittery, distorted",
    width=768, height=512, num_frames=121,
    frame_rate=24.0,
    num_inference_steps=8,
    guidance_scale=3.0,
    stg_scale=1.0,
    modality_scale=3.0,
    guidance_rescale=0.7,
    audio_guidance_scale=7.0,
    audio_stg_scale=1.0,
    audio_modality_scale=3.0,
    audio_guidance_rescale=0.7,
    spatio_temporal_guidance_blocks=[28],  # [28] for 2.3, [29] for 2.0
    use_cross_timestep=True,              # True for 2.3, False for 2.0
    output_type="np",
    return_dict=False,
)
```

### Saving Output
```python
from diffusers.pipelines.ltx2.export_utils import encode_video

encode_video(
    video[0],
    fps=24.0,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="output.mp4",
)
```

### Memory Optimization
```python
# Option 1: CPU offloading (recommended for 96 GiB VRAM)
pipe.enable_model_cpu_offload()

# Option 2: FP8 layerwise casting (halves VRAM for transformer)
from diffusers import AutoModel
transformer = AutoModel.from_pretrained(..., subfolder="transformer", torch_dtype=torch.bfloat16)
transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

# Option 3: VAE tiling (prevents OOM on large videos)
pipe.vae.enable_tiling()
```

## Weights

### Source
Community-converted diffusers-compatible weights:
`CalamitousFelicitousness/LTX-2.3-distilled-Diffusers`

### Why Not Official Weights
The official `Lightricks/LTX-2.3` repo is missing `model_index.json` —
`from_pretrained()` fails with `RemoteEntryNotFoundError: 404`.
The community conversion adds the required directory structure and config files.

### Components and Sizes (BF16)
| Component | Size | Files |
|-----------|------|-------|
| transformer | 35.4 GiB | 4 shards |
| text_encoder (Gemma 3 12B) | 22.7 GiB | 5 shards |
| connectors | 5.9 GiB | 1 file |
| vae (video) | 1.4 GiB | 1 file |
| latent_upsampler (2x spatial) | 0.9 GiB | 1 file |
| vocoder (audio) | 0.2 GiB | 1 file |
| audio_vae | 0.1 GiB | 1 file |
| **Total** | **~66.7 GiB** | |

### Download Location
`/persistent/models/ltx-2.3-distilled-diffusers/`

### Kijai ComfyUI Weights — DELETED
The 44 GiB of Kijai/LTX2.3_comfy weights were deleted. They are NOT compatible
with diffusers `from_pretrained()` (different format, single-file vs directory layout).

## Container Image (TODO)

Need to build a custom container with:
- Python 3.12+
- PyTorch >= 2.7 with CUDA 12.8
- `diffusers` >= 0.37.0 (or installed from main)
- `transformers`, `accelerate`, `safetensors`
- Minimal FastAPI server exposing:
  - `POST /generate` — text-to-video (accepts prompt, returns MP4)
  - `GET /health` — health check

## Key Constraints

- **num_frames**: Must be (N*8)+1 — e.g., 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121
- **width/height**: Must be divisible by 32
- **Distilled model**: Uses only 8 inference steps (vs 40 for base model)
- **CFG**: Distilled model uses `guidance_scale=1.0` (no classifier-free guidance)
- **CUDA**: Requires >= 12.7
- **Python**: Requires >= 3.12

## Next Steps

1. Wait for weight download to complete (~67 GiB)
2. Test inference locally with a Python script
3. Build container image with FastAPI wrapper
4. Deploy via catalog API
5. Measure actual VRAM usage and performance
