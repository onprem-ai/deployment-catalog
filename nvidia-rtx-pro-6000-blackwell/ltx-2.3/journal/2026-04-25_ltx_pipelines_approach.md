# 2026-04-25 — Official ltx-pipelines Approach

**Goal:** Serve LTX-2.3 distilled model via the official Lightricks `ltx-pipelines` package.

## Why ltx-pipelines

After two failed approaches:
1. **ComfyUI** — auth middleware blocked all API access (see `2026-04-25_comfyui_abandoned.md`)
2. **Diffusers community weights** — config/architecture mismatch (see `2026-04-25_diffusers_failed.md`)

The official `ltx-pipelines` package is the safest path:
- Maintained by Lightricks (the model authors)
- Loads raw `.safetensors` checkpoints directly — no diffusers config translation
- Built-in FP8 quantization (`QuantizationPolicy.fp8_cast()`)
- Clean Python API via `DistilledPipeline` class
- PyPI package: `pip install ltx-pipelines` (v1.0.0)

## Architecture

### Pipeline Class

Using `DistilledPipeline` (not `TI2VidTwoStagesPipeline`) because:
- We have the fully distilled checkpoint, not the dev model
- 8 predefined inference steps (fast)
- No guidance parameters needed — simpler API
- Fastest inference among all pipeline options

### Constructor

```python
from ltx_pipelines.distilled import DistilledPipeline
from ltx_core.quantization import QuantizationPolicy

pipeline = DistilledPipeline(
    distilled_checkpoint_path="/models/ltx-2.3-22b-distilled-1.1.safetensors",
    spatial_upsampler_path="/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    gemma_root="/models/gemma-3-12b-it-qat-q4_0-unquantized",
    loras=[],
    quantization=QuantizationPolicy.fp8_cast(),
)
```

### Inference Call

```python
video_iter, audio = pipeline(
    prompt="A cat walking on a beach at sunset",
    seed=42,
    height=512,
    width=768,
    num_frames=121,
    frame_rate=25.0,
    images=[],
    enhance_prompt=False,
)
```

### Saving Output

```python
from ltx_pipelines.utils.media_io import encode_video

encode_video(
    video=video_iter,
    fps=25,
    audio=audio,
    output_path="output.mp4",
    video_chunks_number=1,
)
```

## Model Files Required

| File | Size | Source |
|------|------|--------|
| `ltx-2.3-22b-distilled-1.1.safetensors` | 43 GiB | Lightricks/LTX-2.3 |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 0.9 GiB | Lightricks/LTX-2.3 |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | ~23 GiB | Lightricks/gemma-3-12b-it-qat-q4_0-unquantized |
| **Total** | **~67 GiB** | |

**Note:** The distilled LoRA (`ltx-2.3-22b-distilled-lora-384-1.1.safetensors`) is NOT needed.
It's only for `TI2VidTwoStagesPipeline` which uses the dev checkpoint + LoRA.
The `DistilledPipeline` uses the fully distilled checkpoint directly.

**Gemma 3 text encoder:** The official Google repo (`google/gemma-3-12b-it-qat-q4_0-unquantized`)
is gated and requires HF auth. Lightricks provides a non-gated mirror at
`Lightricks/gemma-3-12b-it-qat-q4_0-unquantized`.

## FastAPI Server

Built a custom FastAPI wrapper at `_docker/ltx-2.3-server/server.py`:

- `GET /health` — health check (returns GPU info, VRAM usage)
- `POST /generate` — accepts JSON `{prompt, seed, height, width, num_frames, frame_rate}`,
  returns MP4 binary

Pipeline is loaded once at startup (lifespan handler). Generation is synchronous
(single request at a time — model is too large for concurrent inference).

## Container Image

Dockerfile at `_docker/ltx-2.3-server/Dockerfile`:
- Base: `nvidia/cuda:12.8.1-devel-ubuntu24.04` (builder) → `runtime` (final)
- Multi-stage build for smaller runtime image
- Installs: PyTorch 2.7 cu128, ltx-pipelines, FastAPI, ffmpeg
- Model weights mounted at `/models` via hostPath volume

## Key Constraints

- **num_frames**: Must satisfy `(num_frames - 1) % 8 == 0` — e.g., 9, 17, 25, ..., 121
- **width/height**: Must be divisible by 32
- **CUDA**: Requires >= 12.7
- **Python**: Requires >= 3.12
- **PyTorch**: Requires ~= 2.7
- **Environment**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` required

## Directory Structure (On Host)

All model files organized under one directory for single K8s volume mount:

```
/persistent/models/ltx-2.3/
├── ltx-2.3-22b-distilled-1.1.safetensors
├── ltx-2.3-spatial-upscaler-x2-1.1.safetensors
└── gemma-3-12b-it-qat-q4_0-unquantized/
    ├── config.json
    ├── model-*.safetensors
    └── tokenizer files...
```

## Next Steps

1. Wait for downloads to complete
2. Organize model files into unified directory
3. Build container image
4. Test locally: `curl -X POST http://localhost:8000/generate -d '{"prompt": "test"}'`
5. Deploy via catalog API
6. Measure actual VRAM usage and generation time
