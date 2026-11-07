# VRAM Binary Search Measurement — Cohere Transcribe

**Date**: 2026-04-28
**Hardware**: NVIDIA RTX PRO 6000 Blackwell (96 GiB)
**Container**: vllm/vllm-openai:v0.19.0
**Method**: Binary search for minimum working gpu_memory_utilization

## Model Characteristics

- **Weights**: ~4.13 GiB safetensors
- **Architecture**: CohereAsrForConditionalGeneration (custom, requires --trust-remote-code)
- **Audio support**: Requires `pip install "vllm[audio]" librosa` at startup
- **Load time**: ~5 seconds (+ ~10s pip install)

## Empirical Results

| GPU Util | Reserved (GiB) | nvidia-smi (MiB) | Status | Notes |
|----------|----------------|-------------------|--------|-------|
| 0.04     | 3.8            | -                 | FAIL   | No available memory for cache blocks |
| 0.05     | 4.8            | -                 | FAIL   | No available memory for cache blocks |
| 0.06     | 5.8            | 6,450             | PASS   | Minimum working |

### At gpu_util=0.06 (detailed)

From vLLM logs:
- Available KV cache memory: 0.41 GiB
- GPU KV cache size: 6,768 tokens
- Maximum concurrency for 1,024 tokens per request: 1.47x
- CUDA graph memory: 0.13 GiB (actual), 0.06 GiB captured
- Init engine took 46.26 seconds (includes pip install + torch.compile warmup)

## Recommended Catalog Values

```
min_working_util = 0.06
recommended_util = 0.06 × 1.15 = 0.069 → 0.07
nvidia-smi at 0.06 = 6,450 MiB ≈ 6.3 GiB → min_gi = 7
```

```yaml
resources:
  weights_gi: 4
  memory:
    - name: "default"
      params: ""
      min_gi: 7  # nvidia-smi reading 6,450 MiB at min util + 15% margin
```

Previous `min_gi: 6` was too tight — at 6/96 = 0.0625, barely above the 0.06 minimum. The 15% margin brings it to 7.

## Notes

- KV cache concurrency is low (1.47x at 1024 tokens) at minimum utilization. For production ASR workloads with multiple concurrent transcriptions, consider `min_gi: 8` for more headroom.
- The ~10s pip install at startup adds to readiness time. Image with pre-installed audio deps would eliminate this.
- First inference triggers torch.compile warmup (~15-30s); envoy timeout of 120s covers this.
