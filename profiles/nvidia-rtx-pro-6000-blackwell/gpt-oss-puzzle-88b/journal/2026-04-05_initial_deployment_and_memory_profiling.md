# Session: Initial Deployment & Memory Profiling

**Date**: 2026-04-05

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML frameworks.
> Information 2+ weeks old may be outdated. Verify against current releases.

## Model Info

- **Model**: GPT-OSS-Puzzle-88B
- **Source**: `nvidia/gpt-oss-puzzle-88B`
- **Architecture**: Heterogeneous MoE (88B total, 5B active per token) with hybrid attention
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
- **Weights**: ~50 GiB

## Final Working Configuration

```bash
# Requires custom vLLM build with PR #38135 and flashinfer
export PYTORCH_ALLOC_CONF=expandable_segments:True

vllm serve /model \
  --served-model-name gpt-oss-puzzle-88b \
  --host 0.0.0.0 --port 8080 \
  --trust-remote-code \
  --chat-template /model/chat_template.jinja \
  --kv-cache-dtype fp8 \
  --stream-interval 20 \
  --gpu-memory-utilization 0.95 \
  --max-cudagraph-capture-size 8 \
  --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 8192
```

**Key settings**:
- `--trust-remote-code` - Required for custom model architecture files
- `--chat-template /model/chat_template.jinja` - Model-specific chat template
- `--kv-cache-dtype fp8` - FP8 KV cache for memory efficiency
- `--max-cudagraph-capture-size 8` - Limits CUDA graph memory overhead
- `PYTORCH_ALLOC_CONF=expandable_segments:True` - Required per model README

---

## Requirements

### Custom vLLM Build

This model requires vLLM PR #38135 which is not in the main release. The inference container installs it at startup:

```bash
VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation \
  'git+https://github.com/vllm-project/vllm.git@refs/pull/38135/head'
```

### FlashInfer

FlashInfer must be installed for optimal performance:

```bash
pip install flashinfer-cubin==0.6.6 flashinfer-jit-cache==0.6.6 \
  --extra-index-url https://flashinfer.ai/whl/cu<cuda-version>
```

---

## Memory Profiles

**All measurements on RTX 6000 Blackwell (96GB), FP8 KV cache**

| Profile | Context | Min GPU Util | VRAM (min) | Concurrency |
|---------|---------|--------------|------------|-------------|
| 131k    | 131072  | 58%          | **57 GiB** | 2.06x       |
| 65k     | 65536   | 56%          | **55 GiB** | 1.41x       |
| 32k     | 32768   | 56%          | **55 GiB** | ~8x         |

**Recommended configuration (95% GPU util):**
| Context | VRAM Used | Concurrency |
|---------|-----------|-------------|
| 131k    | 92 GiB    | 24.51x      |

### Memory Breakdown (2026-04-05)

- **Model weights**: 50.46 GiB (fixed)
- **CUDA graphs**: ~0.6 GiB (fixed)
- **KV cache**: Scales with context and `--gpu-memory-utilization`

**KV cache requirements (FP8, single request):**
- 32K context: 0.63 GiB
- 65K context: ~1.3 GiB
- 131K context: 1.57 GiB

**Note**: The minimum GPU utilization is ~55-56% regardless of context length because model weights (50.46 GiB) dominate. The difference between 32K and 131K is minimal since KV cache for a single request is small compared to model weights.

---

## VRAM Measurement Methodology

vLLM pre-allocates KV cache based on `--gpu-memory-utilization`. To find minimum requirements:

1. Deploy with low `--gpu-memory-utilization` (e.g., 50%)
2. If OOM error: "To serve at least one request with max seq len (X), Y GiB KV cache needed"
3. Increase utilization until model loads successfully
4. Measure actual VRAM with `nvidia-smi`

---

## Current Status

**Experimental** - Requires custom vLLM build, startup takes ~5 minutes for dependency installation.

## Constraints

- **Startup time**: ~5 minutes (deps install + model load)
- **Custom build**: Requires vLLM PR #38135
- **Trust remote code**: Model uses custom architecture files
