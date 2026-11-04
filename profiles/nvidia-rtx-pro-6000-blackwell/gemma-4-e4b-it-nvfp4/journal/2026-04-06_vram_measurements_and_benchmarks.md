# Session: VRAM Measurements and Benchmarks

**Date**: 2026-04-06
**Model**: cosmicproc/gemma-4-E4B-it-NVFP4
**Hardware**: 1x NVIDIA RTX 6000 Blackwell (96GB)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML frameworks.
> Information 2+ weeks old may be outdated. Verify against current releases.

## Model Info (updated)

- **Weights**: ~10 GiB (NVFP4 quantized, measured)

## Known Issues (updated)

3. **GPU Memory Utilization calculation**: The catalog service calculates `gpu_memory_utilization = min_gi / system_gpu_gi`. The `min_gi` value in inference.yaml must include external overhead (~3 GiB) to ensure sufficient KV cache allocation. Fixed by storing nvidia-smi readings (not vLLM reserved amount) in catalog.

---

## VRAM Measurements (2026-04-06)

**Measured on RTX 6000 Blackwell (96GB)**
**Methodology**: a binary search over the engine's GPU memory setting, keeping the lowest value that serves the full context.

### FP8 KV Cache (--kv-cache-dtype fp8_e4m3)
| Context | Min Util | Recommended Util | Catalog min_gi | Notes |
|---------|----------|------------------|------------------|-------|
| 128K    | 16%      | 18%              | **20 GiB**       | 1.17 GiB KV cache needed |
| 64K     | 16%      | 18%              | **20 GiB**       | 0.67 GiB KV cache needed |
| 32K     | 14%      | 16%              | **18 GiB**       | 0.42 GiB KV cache needed |

### FP16 KV Cache (--kv-cache-dtype auto)
| Context | Min Util | Recommended Util | Catalog min_gi | Notes |
|---------|----------|------------------|------------------|-------|
| 128K    | 18%      | 20%              | **22 GiB**       | 2.33 GiB KV cache needed |
| 64K     | 17%      | 19%              | **21 GiB**       | 1.33 GiB KV cache needed |
| 32K     | 16%      | 18%              | **20 GiB**       | 0.83 GiB KV cache needed |

**Model loading**: 9.9 GiB, 3.9 seconds
**External overhead**: ~3 GiB (CUDA context + PyTorch runtime)

**Catalog formula**: `min_gi = (recommended_util × 96) + 3` (nvidia-smi reading, not vLLM reserved)

**Note**: FP16 KV cache uses ~2x memory for KV portion compared to FP8. Model weights (~10 GiB) + CUDA graphs (~0.6 GiB) dominate minimum VRAM.

**Comparison with 31B**:
| Metric | E4B | 31B |
|--------|-----|-----|
| Model weights | 10 GiB | 31 GiB |
| 128K context | 23 GiB | 82 GiB |
| Load time | 3.9s | 9s |
| Decode TPS | 153 | 40 |

---

## Standard 4-Test Benchmark Suite (2026-04-06)

Using the API benchmark tool with `num_words` parameter (generates new random prompt per request).

**Container:** vllm/vllm-openai:gemma4-cu130 (v0.18.2rc1.dev73)
**GPU Memory Utilization:** 0.90
**Context Length:** 32k (fp8 kv cache)

| Test | num_words | max_tokens | Decode TPS | Prefill TPS | TTFT |
|------|-----------|------------|------------|-------------|------|
| Decode | 50 | 500 | 146 | 11,236 | 20ms |
| Prefill | 1500 | 100 | 131 | **66,942** | 80ms |
| Balanced | 500 | 500 | **150** | 37,190 | 50ms |
| Long context | 3000 | 100 | 103 | 44,581 | 240ms |

**Key Observations:**
- **Decode TPS:** 100-150 tok/s depending on workload (best: 150 on balanced)
- **Prefill TPS:** Very fast - 37k-67k tok/s range (small model + NVFP4 = exceptional prefill)
- **TTFT scales linearly:** 20ms (68 tokens) → 80ms (1352 tokens) → 240ms (2628 tokens)

**Catalog Values (inference.yaml):**
- `tps_decode`: 150 (balanced workload measurement)
- `tps_prefill`: 50,000 (conservative mid-point of 37k-67k range)
