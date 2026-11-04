# Session: Initial Framework Testing (vLLM, SGLang, TRT-LLM)

**Date:** 2026-03-16
**Model:** NVIDIA Nemotron-3 Super 120B A12B NVFP4
**Hardware:** NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date - is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

## Hardware Requirements

- **GPU**: NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)
- **System RAM**: Minimum 128GB recommended
- **Storage**: ~75GB for model files (NVFP4 quantized)

---

## vLLM Testing (Failed)

**Goal:** Deploy Nemotron-3 Super NVFP4 on RTX 6000 Blackwell using vLLM

**Findings:**
- vLLM's NVFP4 MoE backend selection doesn't recognize SM120 (compute capability 12.0)
- Code checks for SM9.0 (Hopper) and SM10.x (datacenter Blackwell), but not SM12.0 (RTX Blackwell)
- Falls back to MARLIN backend instead of native NVFP4 kernels
- MARLIN fails with CUBLAS_STATUS_INVALID_VALUE errors

**Outcome:** Failed - MoE backend incompatibility

**Error signature:**
```
RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm(...)`
```

**Environment variables tested (all failed):**

| Variable | Value | Result |
|----------|-------|--------|
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Still falls back to MARLIN |
| `VLLM_MXFP4_USE_MARLIN` | `1` | Same CUBLAS error |
| `VLLM_NVFP4_GEMM_BACKEND` | `cutlass` | Only affects linear GEMM, MoE still uses MARLIN |

**Related Issues:**
- [#33416](https://github.com/vllm-project/vllm/issues/33416) - NVFP4 MoE kernels fail on RTX Blackwell (SM12.0)
- [#31085](https://github.com/vllm-project/vllm/issues/31085) - Add SM120 support for native NVFP4 MoE kernels

---

## SGLang Testing (Failed)

**Goal:** Try SGLang as alternative to vLLM for NVFP4 MoE

**Findings:**
- SGLang v0.5.9 doesn't support NVFP4 format (only FP8)
- SGLang nightly (`lmsysorg/sglang:nightly-dev-20260310-0fd9a57d`) fails with weight loading error
- SGLang correctly detects SM120, but weight key mapping is broken

**Outcome:** Failed - NVFP4 weight format not supported

**Error (nightly):**
```
KeyError: 'model.layers.1.mixer.experts.w2_weight_scale'
```

**Note:** SGLang correctly recognizes SM120:
```
SM120 (Blackwell) detected: auto-selecting fp4-gemm-backend=flashinfer_cudnn
```

**Related Issues:**
- [#4598](https://github.com/sgl-project/sglang/issues/4598) - KeyError: w2_weight_scale
- [#3575](https://github.com/sgl-project/sglang/issues/3575) - w2_weight_scale_inv KeyError

---

## TensorRT-LLM Testing (Failed)

**Goal:** Try TensorRT-LLM for NVFP4 MoE deployment

**Findings:**
- TRT-LLM requires pre-converted checkpoint format, not HuggingFace format
- Missing `quant_cfg.json` in the HuggingFace checkpoint
- NVIDIA notes configs "depend on changes not yet merged into 1.3.0rc7 release"

**Outcome:** Failed - Requires checkpoint conversion and custom TRT-LLM build

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/model/quant_cfg.json'
```

**To use TRT-LLM (not tested):**
1. Convert checkpoint using NVIDIA Model Optimizer
2. Build engine with `trtllm-build`
3. Serve with `trtllm-serve`

**References:**
- [TRT-LLM Checkpoint Format](https://nvidia.github.io/TensorRT-LLM/architecture/checkpoint.html)
- [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)

---

## Community avarok Image (Not Applicable)

**Goal:** Test community `avarok/vllm-nvfp4-gb10-sm120` Docker image

**Findings:**
- Image is for **SM121** (DGX Spark GB10), not SM120 (RTX Blackwell)
- Architecture mismatch: ARM64 (Grace CPU) vs x86_64
- Patches address SM121-specific missing PTX instruction, not relevant to SM120

**Outcome:** Not applicable - wrong architecture

| GPU | Compute Capability | Architecture | Platform |
|-----|-------------------|--------------|----------|
| DGX Spark GB10 | SM121 (12.1) | ARM64 | avarok image |
| RTX 6000 Blackwell | SM120 (12.0) | x86_64 | vLLM v0.15.1+ |

**References:**
- [NVFP4 W4A4 MoE Inference on NVIDIA Blackwell GB10 (Blog)](https://blog.avarok.net/nvfp4-w4a4-moe-inference-on-nvidia-blackwell-gb10-1a83e85d0f9e)
- [avarok/vllm-nvfp4-gb10-sm120 Docker Image (NVIDIA Forums)](https://forums.developer.nvidia.com/t/new-bleeding-edge-vllm-docker-image-avarok-vllm-nvfp4-gb10-sm120/354231)

---

## vLLM v0.17.1-cu130 with PR #33417 (Success)

**Goal:** Test official vLLM v0.17.1 with SM120 fix (PR #33417)

**Findings:**
- PR #33417 merged 2026-01-31, cherry-picked to v0.15.1
- Adds `is_device_capability_family(120)` checks for SM120 support
- vLLM v0.17.1-cu130 includes the fix
- MoE backend correctly selects FLASHINFER_CUTLASS instead of MARLIN
- Model loads successfully, inference works

**Outcome:** Success - NVFP4 MoE working on SM120

**Key log confirming fix:**
```
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends:
['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTLASS', 'VLLM_CUTLASS', 'MARLIN']
```

**Performance metrics:**
- Model weights: 69.54 GiB VRAM
- Loading time: ~32 seconds
- Thinking mode (`<think>...</think>` blocks): Working

**Working configuration:**
```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/model:/model \
  vllm/vllm-openai:v0.17.1-cu130 \
  --model /model \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.94
```

**References:**
- [PR #33417](https://github.com/vllm-project/vllm/pull/33417) - Add SM120 device capability checks

---

## Current Status (as of this session)

**Status:** Working (vLLM v0.17.1-cu130)

**Recommended Configuration:**
- Image: `vllm/vllm-openai:v0.17.1-cu130`
- Key flags: `--quantization modelopt_fp4 --kv-cache-dtype fp8`
- MoE Backend: FLASHINFER_CUTLASS (automatic with v0.15.1+)

**Memory profiles:**

| Context Length | VRAM Required | Notes |
|---------------|---------------|-------|
| 128k | ~88Gi | Tested working |
| 64k | ~82Gi | Lower memory profile |
| 262k | ~96Gi | Full native context (may need tuning) |

**Known Issues:**
- NVFP4 performance characteristics differ from Q4/Q8 quantization
- MTP architecture uses token-level parallelism - batch size 1 optimal for latency

**Alternative Options (not recommended):**
1. **SGLang**: Weight key mapping for NVFP4 MoE still broken
2. **TensorRT-LLM**: Requires checkpoint conversion and custom build
3. **BF16 version**: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (requires more VRAM)

## References

- [NVIDIA Nemotron Page](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b)
- [Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [HuggingFace Model](https://huggingface.co/nvidia/Nemotron-3-Super-120B-A12B-NVFP4)
