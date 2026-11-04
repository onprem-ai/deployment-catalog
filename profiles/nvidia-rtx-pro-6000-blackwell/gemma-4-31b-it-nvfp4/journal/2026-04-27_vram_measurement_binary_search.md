# VRAM Measurement: Binary Search (2026-04-27)

## Hardware
- GPU: NVIDIA RTX PRO 6000 Blackwell (98 GiB VRAM)
- vLLM image: `vllm/vllm-openai:gemma4-cu130` (vLLM 0.19.1.dev6)
- Model weights: 31.04 GiB (NVFP4, 4 safetensors shards)

## Methodology
Per `skills/vram-measurement.md`:
1. Binary search for minimum `min_gi` that boots + serves inference
2. Compute `min_util = min_min_gi / 98`
3. Apply 15% safety margin to utilization: `recommended_util = min_util × 1.15`
4. Deploy at `ceil(recommended_util × 98)` and record nvidia-smi reading
5. nvidia-smi reading becomes the `min_gi` in the catalog

## Key Finding: float16 KV cache NOT supported
This model only supports FP8 KV cache. Attempting `--kv-cache-dtype float16` results in:
```
AssertionError: unsupported kv_cache_dtype (str), got float16.
```
This comes from flashinfer during CUDA graph capture. The `--kv-cache-dtype auto` also maps to `fp8_e4m3` for this model. Removed float16 profiles from inference.yaml.

## Binary Search Results (FP8 KV cache)

### 32K context (`--max-model-len 32768 --kv-cache-dtype fp8_e4m3`)
| min_gi | gpu_util | Result | nvidia-smi |
|----------|----------|--------|------------|
| 38       | 0.39     | FAIL   | -          |
| 40       | 0.41     | FAIL   | -          |
| 41       | 0.42     | PASS   | 43.5 GiB   |
| 42       | 0.43     | PASS   | 44.2 GiB   |

min_util = 41/98 = 0.418, recommended = 0.481, deploy at ceil(0.481×98) = 48

**nvidia-smi at recommended: 49,901 MiB (48.7 GiB) → min_gi = 49**

### 64K context (`--max-model-len 65536 --kv-cache-dtype fp8_e4m3`)
| min_gi | gpu_util | Result | nvidia-smi |
|----------|----------|--------|------------|
| 41       | 0.42     | FAIL   | -          |
| 42       | 0.43     | PASS   | 44.1 GiB   |
| 43       | 0.44     | PASS   | 45.0 GiB   |

min_util = 42/98 = 0.429, recommended = 0.493, deploy at ceil(0.493×98) = 49

**nvidia-smi at recommended: 51,259 MiB (50.1 GiB) → min_gi = 50**

### 128K context (`--max-model-len 131072 --kv-cache-dtype fp8_e4m3`)
| min_gi | gpu_util | Result | nvidia-smi |
|----------|----------|--------|------------|
| 44       | 0.45     | FAIL   | -          |
| 45       | 0.46     | PASS   | 47.1 GiB   |
| 46       | 0.47     | PASS   | 49.5 GiB   |

min_util = 45/98 = 0.459, recommended = 0.528, deploy at ceil(0.528×98) = 52

**nvidia-smi at recommended: 55,281 MiB (54.0 GiB) → min_gi = 54**

## Final Catalog Values

| Profile | min_gi | Source |
|---------|----------|--------|
| 32K context | 49 | nvidia-smi reading at recommended util |
| 64K context | 50 | nvidia-smi reading at recommended util |
| 128K context | 54 | nvidia-smi reading at recommended util |
