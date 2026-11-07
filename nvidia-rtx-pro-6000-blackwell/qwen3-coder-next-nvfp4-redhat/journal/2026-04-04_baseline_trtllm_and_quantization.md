# Session: Baseline Configuration, TensorRT-LLM Testing, and Quantization Comparison

**Date:** 2026-04-04
**Model:** Qwen3-Coder-Next NVFP4 RedHatAI
**Hardware:** NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date - is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

---

## Hardware Requirements

- **GPU**: NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)
- **System RAM**: Minimum 32GB recommended
- **Storage**: ~45GB for model files (NVFP4 quantized, 10 shards)

---

## Part 1: Document Current vLLM Configuration

**Goal:** Document baseline vLLM configuration before TensorRT-LLM optimization

**Current Configuration (vLLM v0.17.1-cu130):**

```yaml
# Serving command
python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_MOUNT_PATH} \
  --served-model-name qwen3-coder-next-nvfp4-redhat \
  --host 0.0.0.0 --port 8080 \
  --trust-remote-code \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
  ${MEMORY_PARAMS} \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --chat-template ${MODEL_MOUNT_PATH}/chat_template.jinja \
  --max-cudagraph-capture-size 128

# Environment variables
VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=629145600  # 600MB
VLLM_FLASHINFER_MOE_BACKEND=latency  # Avoid SM120 throughput backend issues
```

**Key vLLM Flags Explained:**

| Flag | Value | Reason |
|------|-------|--------|
| `--quantization` | compressed-tensors | NVFP4 format (FP4 weights, FP8 scales) |
| `--kv-cache-dtype` | fp8 | Memory efficiency for KV cache |
| `--max-cudagraph-capture-size` | 128 | Required for hybrid DeltaNet architecture |
| `--tool-call-parser` | qwen3_coder | Enables function calling / tool use |
| `--enable-auto-tool-choice` | - | Auto-detect tool calling in prompts |

**Memory Profiles (from inference.yaml):**

| Context Length | VRAM Used | Parameters |
|---------------|-----------|------------|
| 256k (native) | 63 GiB | --max-model-len 262144 --max-num-seqs 64 |
| 128k | 60 GiB | --max-model-len 131072 --max-num-seqs 128 |
| 64k | 57 GiB | --max-model-len 65536 |
| 32k (minimum) | 55 GiB | --max-model-len 32768 |

**Baseline Performance:**
- Model weights: ~44 GiB (NVFP4 quantized)
- VRAM usage: 72 GiB (with default memory profile)

| Concurrency | Generation Speed | Prompt Throughput | TTFT |
|-------------|-----------------|-------------------|------|
| 1 | 123 tok/s | 5,557 tok/s | 70ms |
| 2 | 215 tok/s | 11,143 tok/s | 70ms |
| 4 | 389 tok/s | 21,843 tok/s | 70ms |

**Why RedHatAI version:**
- RedHatAI/Qwen3-Coder-Next-NVFP4 includes proper chat_template.jinja
- May have different quantization calibration than GadflyII version

---

## Part 2: TensorRT-LLM Testing

**Goal:** Evaluate TensorRT-LLM for improved throughput on Blackwell GPUs

**Result: INCOMPATIBLE**

TensorRT-LLM 1.0.0 (NVIDIA Release 25.06) does not support the RedHatAI `compressed-tensors` quantization format:

```
NotImplementedError: Unsupported quantization_config: {'config_groups': {'group_0': {'format': 'nvfp4-pack-quantized', ...}}}
```

**Root Cause:**
- RedHatAI uses `compressed-tensors` format (vLLM-native quantization library)
- TensorRT-LLM requires `modelopt` format from TensorRT Model Optimizer
- These are incompatible NVFP4 checkpoint formats

**Alternative: NVIDIA-Compatible NVFP4 Checkpoint**

For TensorRT-LLM/SGLang compatibility, use `modelopt` format:
- [vincentzed-hf/Qwen3-Coder-Next-NVFP4](https://huggingface.co/vincentzed-hf/Qwen3-Coder-Next-NVFP4) - TRT Model Optimizer format
- Serve with: `sglang serve --model-path vincentzed-hf/Qwen3-Coder-Next-NVFP4 --quantization modelopt_fp4`

**Known TRT-LLM Issues with Qwen3-Next:**

| Issue | Version | Error | Status |
|-------|---------|-------|--------|
| Quantization format | 1.0.0 (25.06) | `NotImplementedError: Unsupported quantization_config: nvfp4-pack-quantized` | Blocker for RedHatAI checkpoint |
| Hybrid cache assertion | 1.3.0rc7 | `AssertionError: mamba hybrid cache requires block reuse to be disabled` | Requires `enable_block_reuse: false` |
| Config rejection | 1.3.0rc7 | `ValueError: LLM got invalid argument: pytorch_backend_config` | `use_cuda_graph`, `autotuner_enabled` rejected |
| ARM Triton kernel | 1.3.0rc7 | `RuntimeError: Triton Error [CUDA]: an illegal instruction` | x86 kernels incompatible with aarch64 (DGX Spark) |
| AutoDeploy patterns | 1.3.0rc9 | N/A | Only `sdpa` attention has pattern matchers, not DeltaNet ([#10849](https://github.com/NVIDIA/TensorRT-LLM/issues/10849)) |

**Detail: Quantization Format Incompatibility**
```
# RedHatAI uses compressed-tensors (vLLM-native)
quantization_config.format = "nvfp4-pack-quantized"
quantization_config.quant_method = "compressed-tensors"

# TRT-LLM requires modelopt format from TensorRT Model Optimizer
# These formats are NOT interchangeable
```

**Detail: Hybrid Attention Cache (reported 2026-03-14)**
Qwen3-Next uses DeltaNet linear attention (36 layers) + full attention (12 layers). The hybrid cache manager requires:
- `enable_block_reuse: false` in kv_cache_config
- Standard KV cache optimizations don't apply to linear attention layers

**Detail: ARM Triton Kernel Crash (reported 2026-03-14)**
The `fused_qkvzba_split_reshape_cat_kernel` Triton kernel compiled for x86/SM10x crashes on ARM aarch64 (DGX Spark GB10). This is platform-specific - x86 Blackwell may work differently.

**Decision:** Continue with vLLM - stable, proven performance, good support for compressed-tensors format. Revisit TRT-LLM when:
1. `compressed-tensors` format support is added, OR
2. We switch to a `modelopt` format checkpoint

---

## Part 3: Scheduler Tuning (Failed)

**Goal:** Improve single-concurrency latency with vLLM scheduler flags

**Tested flags:**
```bash
--num-scheduler-steps 1           # Process immediately, don't batch
--enable-chunked-prefill false    # Avoid prefill chunking overhead
--disable-log-requests            # Reduce I/O
```

**Results: WORSE performance**

| Concurrency | Baseline | With Flags | Change |
|-------------|----------|------------|--------|
| 1 | 123 tok/s | 117 tok/s | **-5%** |
| 2 | 215 tok/s | 188 tok/s | **-12%** |
| 4 | 389 tok/s | 372 tok/s | **-4%** |

TTFT at concurrency 2 degraded from 70ms to 360ms.

**Analysis:** The hybrid DeltaNet+MoE architecture benefits from vLLM's default scheduling. Disabling chunked prefill and forcing single-step scheduling removes optimizations that help this model. The `VLLM_FLASHINFER_MOE_BACKEND=latency` environment variable (already set) is the correct latency optimization for this architecture.

**Conclusion:** Keep default scheduler settings. The baseline configuration is already well-optimized for this model.

---

## Part 4: Q8 vs NVFP4 Benchmark

**Goal:** Compare llama.cpp Q8_0 vs vLLM NVFP4 for single-concurrency latency

**Results:**

| Concurrency | Q8 (llama.cpp) | NVFP4 (vLLM) | Winner |
|-------------|----------------|--------------|--------|
| **1** | **141 tok/s** | 123 tok/s | **Q8 +14%** |
| 2 | 100 tok/s | 215 tok/s | NVFP4 +115% |
| 4 | 248 tok/s | 389 tok/s | NVFP4 +57% |

**Full comparison at concurrency 1:**

| Metric | Q8 (llama.cpp) | NVFP4 RedHatAI (vLLM) | Winner |
|--------|----------------|----------------------|--------|
| **Generation** | **141 tok/s** | 123 tok/s | Q8 +14% |
| **Prompt (prefill)** | 3,025 tok/s | **5,557 tok/s** | NVFP4 +84% |
| **TTFT** | 130ms | **70ms** | NVFP4 -46% |
| VRAM | 80 GB | 72 GB | NVFP4 -10% |

**Analysis:**
- **Q8 llama.cpp is 14% faster at generation** (141 vs 123 tok/s)
- **NVFP4 vLLM is 84% faster at prefill** (5.5k vs 3k tok/s) with lower TTFT
- NVFP4 scales much better with concurrent requests (continuous batching)
- For coding tasks with long prompts/context, NVFP4 may feel faster despite slower generation

**Recommendation:**
- **Short prompts, long outputs:** Use Q8 with llama.cpp (faster generation)
- **Long prompts (code context), multi-user:** Use NVFP4 with vLLM (faster prefill, better scaling)

---

## Part 5: MTP Speculative Decoding

**Status:** NOT AVAILABLE for this checkpoint

### What is MTP?

Multi-Token Prediction (MTP) is a speculative decoding method native to Qwen3-Next models. The model predicts multiple tokens ahead, then verifies them - reducing time-per-output-token at low concurrency.

### Why NVFP4 Doesn't Support MTP

Standard NVFP4 quantization via llm-compressor **strips MTP weights** because `AutoModelForCausalLM` doesn't load them. The RedHatAI checkpoint does not include `mtp.*` weights.

| Checkpoint | MTP Support | Reason |
|------------|-------------|--------|
| RedHatAI/Qwen3-Coder-Next-NVFP4 | No | MTP weights stripped during quantization |
| Qwen/Qwen3-Coder-Next-FP8 | Yes | FP8 preserves all weights |
| Qwen/Qwen3-Coder-Next (BF16) | Yes | Original model has full MTP |

### How to Enable MTP on NVFP4 (Advanced)

To get MTP working on NVFP4, you would need to:

1. Extract MTP weights from original BF16 model:
   ```python
   # Load BF16 model, extract mtp.* weights
   # Save as mtp_weights.safetensors
   ```

2. Add MTP weights to NVFP4 checkpoint directory

3. Re-quantize with MTP layers in ignore list (kept as BF16)

This approach is used by [SPARK_Qwen3.5-122B-A10B-NVFP4](https://github.com/JungkwanBan/SPARK_Qwen3.5-122B-A10B-NVFP4) and achieves 3.3x speedup on GB10.

### Is MTP Worth It for Qwen3-Coder-Next?

**Probably not a priority:**

- Qwen3-Coder-Next is a **non-thinking** model (no `<think>` blocks)
- Optimized for fast agentic responses, not extended reasoning
- MTP speedup is biggest for thinking/reasoning models with long outputs
- Current NVFP4 performance (123 tok/s) is already solid for coding tasks

**Consider MTP if:**
- You need lower latency for single-user interactive coding
- You're willing to maintain a custom checkpoint with grafted MTP weights
- You can use the FP8 variant (but it has CUDA graph issues)

### Alternatives

| Option | MTP | Performance | Notes |
|--------|-----|-------------|-------|
| NVFP4 RedHatAI (current) | No | 123 tok/s | Stable, recommended |
| FP8 + MTP | Yes | ~19 tok/s* | *CUDA graph bug, needs --enforce-eager |
| llama.cpp Q8 | No | 141 tok/s | Faster generation, no prefill advantage |

**References:**
- [vLLM MTP Documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/)
- [MTP + NVFP4 weight mismatch bug #35031](https://github.com/vllm-project/vllm/issues/35031)
- [SPARK Qwen3.5 NVFP4 with MTP](https://github.com/JungkwanBan/SPARK_Qwen3.5-122B-A10B-NVFP4)
- [GadflyII MTP-enabled models](https://huggingface.co/GadflyII/GLM-4.7-Flash-MTP-NVFP4)

---

## Session Summary

**Outcome:** vLLM is the recommended runtime. TensorRT-LLM is incompatible with this checkpoint format.

**Recommended Configuration (vLLM v0.17.1-cu130):**
- Image: `vllm/vllm-openai:v0.17.1-cu130`
- Quantization: `compressed-tensors` (NVFP4 format)
- Key flags:
  - `--max-cudagraph-capture-size 128` (required for hybrid DeltaNet architecture)
  - `--gpu-memory-utilization 0.90` (leave headroom for inference)
  - `--kv-cache-dtype fp8` (memory efficiency)
  - `--tool-call-parser qwen3_coder` (tool calling support)
- Environment variables:
  - `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=629145600` (600MB)
  - `VLLM_FLASHINFER_MOE_BACKEND=latency` (avoid SM120 throughput backend issues)

**References:**
- [RedHatAI NVFP4 Model](https://huggingface.co/RedHatAI/Qwen3-Coder-Next-NVFP4)
- [TensorRT-LLM Release Notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)
- [TRT-LLM Qwen3-Next Quick Start](https://nvidia.github.io/TensorRT-LLM/deployment-guide/quick-start-recipe-for-qwen3-next-on-trtllm.html)
- [Alternative modelopt checkpoint](https://huggingface.co/vincentzed-hf/Qwen3-Coder-Next-NVFP4)
