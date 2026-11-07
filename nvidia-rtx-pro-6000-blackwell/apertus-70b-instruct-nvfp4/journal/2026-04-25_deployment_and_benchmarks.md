# 2026-04-25 — Deployment and Benchmarks

**Goal:** Deploy Apertus 70B NVFP4, measure VRAM, benchmark performance.

## Deployment Issues

### Issue 1: fp8_e5m2 KV cache incompatible with NVFP4
- **Error:** `ValueError: fp8_e5m2 kv-cache is not supported with fp8 checkpoints`
- **Root cause:** NVFP4 ModelOpt checkpoints contain FP8 scales internally. vLLM rejects `--kv-cache-dtype fp8_e5m2` when the checkpoint already has FP8 components.
- **Fix:** Removed `--kv-cache-dtype fp8_e5m2` from all memory profiles. BF16 KV cache is used instead.

### Issue 2: Insufficient VRAM for KV cache
- **Error:** `ValueError: 5.0 GiB KV cache is needed, which is larger than the available KV cache memory (0.4 GiB)`
- **Root cause:** Initial estimates assumed 35 GiB weights on disk = 35 GiB in VRAM. Actual GPU memory for weights: 40.04 GiB (NVFP4 weights + FP8 scales + activation buffers).
- **Fix:** Updated all memory profiles after VRAM binary search.

### Issue 3: xIELU activation warning (benign)
- **Warning:** `CUDA-fused xIELU not available (No module named 'xielu') – falling back to a Python version`
- **Impact:** None — Python fallback works correctly. Optional CUDA optimization available via `pip install git+https://github.com/nickjbrowning/XIELU`

## VRAM Binary Search Results

GPU: NVIDIA RTX 6000 Blackwell (96 GiB)
Image: vllm/vllm-openai:v0.19.1-cu130
Weights in VRAM: 40.04 GiB
Overhead (CUDA context, NCCL, activations): ~4.85 GiB

| Profile   | Context  | Min VRAM | KV Cache Available | Concurrency |
|-----------|----------|----------|--------------------|-------------|
| 16k       | 16,384   | 50 GiB   | 5.15 GiB           | 1.03x       |
| 32k       | 32,768   | 55 GiB   | 10.77 GiB          | 1.08x       |
| 65k       | 65,536   | 65 GiB   | 20.27 GiB          | 1.01x       |

**Failed attempts:**
- 48 GiB (0.50 util) → 3.25 GiB KV → FAIL (needs 5.0 GiB for 16k)
- 49 GiB (0.51 util) → 4.20 GiB KV → FAIL
- 53 GiB (0.55 util) → 7.92 GiB KV → FAIL (needs 10.0 GiB for 32k)
- 54 GiB (0.56 util) → 8.87 GiB KV → FAIL
- 64 GiB (0.67 util) → 19.32 GiB KV → FAIL (needs 20.0 GiB for 65k)

## Performance Benchmarks

Benchmark at concurrency 1, 16k context, 203 input tokens, 450 max output tokens:

| Metric           | Value          |
|------------------|----------------|
| Decode TPS       | 19.34 tok/s    |
| Prefill TPS      | 3,663.31 tok/s |
| TTFT             | 0.18s          |
| Success Rate     | 100%           |

## Configuration

Key vLLM flags:
- `--quantization modelopt` — required for NVFP4
- `--dtype auto` — resolves to BF16
- `--enforce-eager` — disables CUDA graphs (needed for stable operation)
- `--enable-auto-tool-choice --tool-call-parser hermes` — tool calling support
- NO `--kv-cache-dtype fp8_e5m2` — incompatible with NVFP4

## Outcome

Model successfully deployed and benchmarked. All memory profiles measured.
Decode speed of 19.34 tok/s is reasonable for a 70B NVFP4 model on a single GPU.
