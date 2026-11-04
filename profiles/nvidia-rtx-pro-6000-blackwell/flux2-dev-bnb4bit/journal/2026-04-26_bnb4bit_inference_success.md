# 2026-04-26 — bnb-4bit Inference Test: SUCCESS

**Goal:** Test FLUX.2-dev bnb-4bit inference in a Kubernetes pod with memory limits, measure actual VRAM usage at multiple resolutions.

## Background

Previous session established that NVFP4 single-file loading is incompatible with diffusers (packed 4-bit tensors can't be converted by the checkpoint pipeline). Manual `load_state_dict` approach also failed — `Flux2Transformer2DModel.from_config()` allocates the full 32B-param model on CPU (~128 GiB float32), causing pod OOMKill at 16 GiB limit. Even if RAM were sufficient, NVFP4 packed tensor shapes don't match the standard dense parameters.

Pivoted to the bnb-4bit pipeline from `diffusers/FLUX.2-dev-bnb-4bit`, which uses bitsandbytes NF4 quantization natively supported by diffusers `from_pretrained`.

## Test Environment

- **Pod**: `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` with `gcc` installed for bitsandbytes triton compilation
- **Memory limits**: 12 GiB request, 16 GiB limit (pod cgroup enforced)
- **GPU**: NVIDIA RTX PRO 6000 Blackwell, 95 GiB VRAM
- **Model files**: `/persistent/models/flux2-dev-bnb4bit/` (pre-downloaded, ~32 GiB on disk)
  - Transformer: 17 GiB (2 shards, NF4)
  - Text encoder: 15 GiB (4 shards, Mistral3 NF4)
  - VAE: 321 MiB (bfloat16)

## Issues Encountered

### 1. Pod scheduling failure (20 GiB request)
Node has ~24 GiB allocatable with ~7 GiB already requested. 20 GiB request exceeded available.
**Fix:** Reduced to 12 GiB request / 16 GiB limit — actual RAM usage is low since model loads to VRAM.

### 2. bitsandbytes triton compilation needs gcc
`pytorch:2.7.0-cuda12.8-cudnn9-runtime` image lacks a C compiler. bitsandbytes 0.49.2 triton module fails with `RuntimeError: Failed to find C compiler`.
**Fix:** Added `apt-get install -y gcc` before pip install.

### 3. `device_map="balanced"` incompatible with bnb-4bit
accelerate 1.13.0 `dispatch_model` passes `_is_hf_initialized` kwarg that `bitsandbytes.Params4bit.__new__()` doesn't accept.
**Fix:** Removed `device_map` parameter entirely. diffusers handles bnb-4bit CUDA placement automatically when loading a pre-quantized checkpoint.

### 4. `total_mem` attribute error
PyTorch 2.7 uses `total_memory` not `total_mem` on `_CudaDeviceProperties`.
**Fix:** Corrected attribute name.

## Results

### Loading
```
Pipeline load time: 12-13s
Versions: diffusers 0.37.1, transformers 5.6.2, bitsandbytes 0.49.2, PyTorch 2.7.0+cu128
```

### VRAM Measurements

| State | VRAM Allocated | VRAM Peak |
|-------|---------------|-----------|
| After pipeline load | 31.43 GiB | 31.90 GiB |
| After 512x512, 20 steps | 31.44 GiB | 32.29 GiB |
| After 1024x1024, 28 steps | 31.44 GiB | 33.83 GiB |

### Performance

| Resolution | Steps | Time | Throughput |
|-----------|-------|------|------------|
| 512x512 | 20 | 7.7s | ~2.7 it/s |
| 1024x1024 | 28 | 31.2s | ~0.9 it/s |

### Image Quality
Both resolutions produced high-quality photorealistic images. Tested prompts:
- "A beautiful sunset over mountains, photorealistic" (512x512 and 1024x1024)
- "A cyberpunk city at night, neon lights, rain" (512x512)

## Loading Pattern (Confirmed Working)

```python
from diffusers import Flux2Pipeline
pipe = Flux2Pipeline.from_pretrained(
    "/models/flux2-dev-bnb4bit",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
```

No `device_map`, no manual state_dict loading, no `nvidia_modelopt`. Just standard `from_pretrained` with the pre-quantized bnb-4bit checkpoint.

## Outcome

- FLUX.2-dev bnb-4bit works on RTX PRO 6000 Blackwell with ~34 GiB peak VRAM (of 95 GiB available)
- Renamed catalog entry from `flux2-dev-nvfp4` to `flux2-dev-bnb4bit`
- Updated inference.yaml with measured VRAM and performance numbers
- Updated Dockerfile: removed `nvidia_modelopt`, added `gcc` + `sentencepiece`
- Updated server.py for bnb-4bit loading pattern
- Pod memory limits (12 GiB request / 16 GiB limit) are sufficient — all heavy work happens in VRAM
