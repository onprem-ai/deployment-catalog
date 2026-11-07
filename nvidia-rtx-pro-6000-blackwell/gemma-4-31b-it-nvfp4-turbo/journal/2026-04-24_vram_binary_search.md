# 2026-04-24 — VRAM Binary Search for All Context Sizes

**Goal:** Measure minimum VRAM for 128k, 64k, and 16k context profiles via binary search.

## Method

Deploy with decreasing `min_gi`, wait for pod readiness. If pod crashes (3+ restarts), mark as failed. GPU has chandra-ocr-2 co-deployed (~19.6 GiB).

## Correction

The 2026-04-22 "measurement" of 69.7 GiB for 128k context was misleading. It was just the VRAM used when deployed at `min_gi: 70`. vLLM expands KV cache to fill the allocated VRAM (via `--gpu-memory-utilization`), so the "used" number reflects the allocation, not the minimum needed.

## Results

### 128k context (`--max-model-len 131072 --kv-cache-dtype fp8_e4m3 --max-num-seqs 128`)

| min_gi | gpu_util | Result |
|----------|----------|--------|
| 30 | 0.31 | FAIL — 3 restarts |
| 31 | 0.32 | FAIL — 3 restarts |
| 32 | 0.33 | **WORKS** — inference confirmed |
| 33 | 0.34 | WORKS — ~32.6 GiB actual |
| 38 | 0.39 | WORKS — ~36.8 GiB actual |
| 45 | 0.46 | WORKS — ~45 GiB actual |
| 55 | 0.56 | WORKS — ~55.5 GiB actual |
| 60 | 0.61 | WORKS — ~60.2 GiB actual |
| 65 | 0.66 | WORKS — ~65 GiB actual |
| 70 | 0.72 | WORKS — ~69.7 GiB actual (2026-04-22 measurement) |

### 64k context (`--max-model-len 65536 --kv-cache-dtype fp8_e4m3 --max-num-seqs 128`)

| min_gi | gpu_util | Result |
|----------|----------|--------|
| 28 | 0.29 | FAIL — 3 restarts |
| 30 | 0.31 | **WORKS** — inference confirmed |
| 33 | 0.34 | WORKS — ~32.5 GiB actual |
| 35 | 0.36 | WORKS — ~34.2 GiB actual |
| 40 | 0.41 | WORKS — ~38.5 GiB actual |
| 45 | 0.46 | WORKS — ~45 GiB actual |

### 16k context (`--max-model-len 16384 --kv-cache-dtype fp8_e4m3 --max-num-seqs 128`)

| min_gi | gpu_util | Result |
|----------|----------|--------|
| 25 | 0.26 | FAIL — OOM during engine core init |
| 26 | 0.27 | FAIL — OOM, 3 restarts |
| 27 | 0.28 | **WORKS** — inference confirmed |

## Summary

| Profile | min_gi | Fails at | Original estimate | Change |
|---------|----------|----------|-------------------|--------|
| 128k context (fp8 kv) | 32 GiB | 31 | 70 GiB | **-54%** |
| 64k context (fp8 kv) | 30 GiB | 28 | 45 GiB | **-33%** |
| 16k context (fp8 kv) | 27 GiB | 26 | 25 GiB | +8% |

## VRAM Breakdown

Weights ~18 GiB + CUDA overhead ~9 GiB = ~27 GiB minimum. KV cache adds only ~3 GiB for 64k, ~5 GiB for 128k with fp8_e4m3. The NVFP4 turbo variant is extremely memory-efficient — 128k context only needs 5 GiB more than 16k.

## Outcome

All three memory profiles updated in inference.yaml. Maturity promoted to stable.
