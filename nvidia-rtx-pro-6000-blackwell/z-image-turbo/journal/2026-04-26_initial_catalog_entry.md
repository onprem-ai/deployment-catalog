# 2026-04-26 — Z-Image-Turbo Initial Catalog Entry

**Goal:** Add Z-Image-Turbo (6B) text-to-image model to the RTX 6000 Blackwell catalog. Download, test inference in a pod, measure VRAM, and create OpenAI-compatible server.

## Model Research

- **Name:** Z-Image-Turbo
- **Org:** Tongyi-MAI (Alibaba)
- **Released:** 2025-11-27
- **License:** Apache 2.0
- **Paper:** arXiv:2511.22699
- **HuggingFace:** `Tongyi-MAI/Z-Image-Turbo`

### Architecture

- **Transformer:** ZImageTransformer2DModel — 6B params, Single-Stream DiT (S3-DiT)
  - Concatenates text, visual semantic, and image VAE tokens into a single stream
- **Text encoder:** Qwen3 4B (`Qwen3Model` + `Qwen2Tokenizer`)
- **VAE:** Standard `AutoencoderKL` (168 MiB)
- **Scheduler:** `FlowMatchEulerDiscreteScheduler` (shift=3.0)

### Key Inference Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| `num_inference_steps` | 9 | Results in 8 DiT forward passes |
| `guidance_scale` | 0.0 | **Must be 0** — CFG-free turbo model |
| `max_sequence_length` | 512 | Max prompt tokens |
| Resolution | Up to 4MP | Any aspect ratio |

### Files on Disk (~33 GiB)

| Component | Size |
|-----------|------|
| Transformer (3 shards) | ~24.6 GiB |
| Text encoder (3 shards) | ~8.1 GiB |
| VAE | ~168 MiB |
| Tokenizer | ~15.9 MiB |

### Diffusers Integration

Requires `ZImagePipeline` — merged in diffusers PRs #12703 and #12715. **Not in stable pip release** — must install from source:

```bash
pip install git+https://github.com/huggingface/diffusers
```

Loading code:
```python
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")
image = pipe(prompt="...", num_inference_steps=9, guidance_scale=0.0).images[0]
```

### VRAM Measured (RTX PRO 6000 Blackwell)

| State | VRAM Allocated | VRAM Peak |
|-------|---------------|-----------|
| After pipeline load | 19.18 GiB | 19.18 GiB |
| After 512x512, 9 steps | 19.18 GiB | 19.78 GiB |
| After 1024x1024, 9 steps | 19.18 GiB | 21.56 GiB |

### Performance Measured

| Resolution | Steps | Time | Throughput |
|-----------|-------|------|------------|
| 512x512 | 9 | 0.6s | ~18.5 it/s |
| 1024x1024 | 9 | 2.5s | ~4.9 it/s |

Pipeline load time: 15.6s. diffusers 0.38.0.dev0 (from source).

## Issues Encountered

1. **No `git` in runtime image**: `pip install git+...` requires git. Fixed by adding `apt-get install git`.
2. **OOMKilled at 12 GiB RAM**: Transformer shards are 9.3 GiB each, pass through CPU during loading. Fixed by increasing to 16 GiB limit and using `device_map="balanced"`.
3. **`ZImagePipeline` not in stable diffusers**: Requires `pip install git+https://github.com/huggingface/diffusers` (merged in PRs #12703, #12715, version 0.38.0.dev0).

## Outcome

Z-Image-Turbo works on RTX PRO 6000 Blackwell. Only 19.2 GiB VRAM (of 95 GiB), generates 1024x1024 in 2.5s at 9 steps. Exceptional speed — nearly 5x faster than FLUX.2-dev bnb-4bit at comparable quality. Apache 2.0 licensed.

Updated inference.yaml with measured values. Container needs 14-16 GiB RAM for model loading (shard staging through CPU), but runtime VRAM is modest.
