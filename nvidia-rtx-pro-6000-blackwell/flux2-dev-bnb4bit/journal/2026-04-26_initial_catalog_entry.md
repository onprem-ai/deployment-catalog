# 2026-04-26 — Initial Catalog Entry

**Goal:** Add FLUX.2-dev NVFP4 (32B) text-to-image model to catalog for RTX 6000 Blackwell.

## Model Research

- **Model:** FLUX.2 [dev] — 32B parameter rectified flow transformer by Black Forest Labs
- **Source:** https://huggingface.co/black-forest-labs/FLUX.2-dev-NVFP4
- **Task:** Text-to-image generation, image editing, image combining
- **License:** flux-dev-non-commercial-license

### Architecture (FLUX.2, NOT FLUX.1)

Major architectural change from FLUX.1:
- FLUX.1 used dual text encoders: CLIP + T5-XXL (~9.5 GiB)
- FLUX.2 uses a single **Mistral3** vision-language model as text encoder
- New pipeline class: `Flux2Pipeline` (not `FluxPipeline`)
- New VAE: `AutoencoderKLFlux2` (~336 MB)
- New transformer: `Flux2Transformer2DModel` (32B params, MMDiT)

### Quantization

| Variant | File | Size | Source |
|---------|------|------|--------|
| NVFP4 (pure) | `flux2-dev-nvfp4.safetensors` | 19.6 GiB | FLUX.2-dev-NVFP4 repo |
| NVFP4/BF16 mixed | `flux2-dev-nvfp4-mixed.safetensors` | 21.2 GiB | FLUX.2-dev-NVFP4 repo |
| BF16 (full) | `flux2-dev.safetensors` | 64.4 GiB | FLUX.2-dev repo |
| BNB 4-bit | multiple shards | ~16.9 GiB | diffusers/FLUX.2-dev-bnb-4bit |

NVFP4 = NVIDIA FP4 quantization format. Requires `nvidia_modelopt[hf]` package.

### VRAM Estimate

| Component | Size |
|-----------|------|
| NVFP4 transformer | ~20 GiB |
| Mistral3 text encoder (bnb 4-bit) | ~15 GiB |
| VAE (BF16) | ~0.3 GiB |
| Inference activations | ~5-10 GiB |
| **Total estimate** | **~35-45 GiB** |

Fits within 95 GiB RTX 6000 Blackwell with significant headroom.
Comparison: LTX-2.3 (22B) OOM'd at 94.9/95 GiB — FLUX.2 is ~2x smaller in VRAM.

## Approach: diffusers + FastAPI

- Load NVFP4 transformer via `Flux2Transformer2DModel.from_single_file()` with `NVIDIAModelOptConfig`
- Load rest of pipeline (text encoder, VAE, tokenizer, scheduler) from `diffusers/FLUX.2-dev-bnb-4bit` (non-gated)
- The main `FLUX.2-dev` repo is gated — using the bnb-4bit repo avoids auth requirement
- Wrap with FastAPI: `POST /generate` (prompt → PNG), `GET /health`
- Enable `pipe.enable_model_cpu_offload()` for memory safety

### Key Code Pattern

```python
from diffusers import Flux2Pipeline, Flux2Transformer2DModel, NVIDIAModelOptConfig

transformer = Flux2Transformer2DModel.from_single_file(
    "flux2-dev-nvfp4.safetensors",
    quantization_config=NVIDIAModelOptConfig(quant_type="NVFP4", quant_method="modelopt"),
    torch_dtype=torch.bfloat16,
)

pipe = Flux2Pipeline.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
```

### Dependencies

- diffusers >= 0.37.0 (for `Flux2Pipeline`)
- transformers (for Mistral3 text encoder)
- nvidia_modelopt[hf] (for NVFP4 loading)
- bitsandbytes (for bnb-4bit text encoder)
- PyTorch >= 2.7 with CUDA 12.8

## Downloads

- NVFP4 transformer: `black-forest-labs/FLUX.2-dev-NVFP4` → `/persistent/models/flux2-dev-nvfp4/`
- Text encoder + VAE: `diffusers/FLUX.2-dev-bnb-4bit` (excluding transformer) → `/persistent/models/flux2-dev-bnb4bit/`

## Files Created

- `inference.yaml` — catalog entry with experimental maturity
- `_docker/flux2-server/server.py` — FastAPI wrapper
- `_docker/flux2-server/Dockerfile` — container image definition

## Outcome

Catalog entry created. Downloads in progress. Need to:
1. Wait for downloads to complete
2. Install diffusers + nvidia_modelopt in test venv
3. Test inference locally
4. Measure actual VRAM usage
5. Update performance numbers in inference.yaml
