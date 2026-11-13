# 2026-11-12 - Full Benchmark and GPU Memory

**Goal:** Measure the tuned settings (journal 2026-11-12, tuning, round 3) with every workload of
the replica benchmark, and write the measured GPU memory.

## The run

Job `16c5a9e6-bf8a-4574-bddd-07413a3465f3`, all three workloads, 23:11:36 to 23:42:07Z, on inference
node `nebius-e00pn5kkfnayh8wn6j`. It succeeded with no failed request. The file is committed as the
control plane wrote it: `benchmarks/2026-11-12-2311Z-llmperfbench.yaml`, whose `compose_sha256`
(`8d0abe98...`) is the committed `compose.yaml`. A probe right before it reproduced the tuning
(147 / 3,827 / 9,395 / 11,170 decode tokens/s at 1 / 32 / 128 / 256 requests).

| Workload | Stop | Figures |
|---|---|---|
| `max-prefill/1.0.0` | plateau at 20 | 15,404 prefill tokens/s at 1 (TTFT p50 0.13 s for 8,192 tokens); 61,450 at 32 |
| `max-decode/1.0.0` | 128, still growing | 157 decode tokens/s at 1 (ITL p50 5.8 ms); 4,290 at 32; 8,656 at 128 |
| `mixed-load/1.0.0` | 128, still growing | 134 decode tokens/s at 1; at 128: 29,103 prefill and 5,311 decode tokens/s, TTFT p50 0.16 s |

`max-decode` and `mixed-load` still grew at 128, where the load generator stops; the probe measured
11,170 decode tokens/s at 256. The card therefore shows no plateau concurrency.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s: 64,588,087,296 bytes (60.15 GiB),
within the 62 GiB hardware booking.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 27,700,629,842 | 25.80 | `Model loading took 26.58 GiB`, minus the drafter |
| `drafters` | 839,427,840 | 0.78 | size of the drafter's safetensors |
| `cuda_graphs` | 214,748,365 | 0.20 | `CUDA graph pool memory: 0.2 GiB (actual)` |
| `kv_cache` | 26,661,009,490 | 24.83 | `Available KV cache memory: 24.83 GiB` |
| `runtime_overhead` | 9,172,271,759 | 8.54 | the peak minus every other part |

`GPU KV cache size: 726,486 tokens` (5.5 requests of 131,072). Engine budget 56 GiB
(`--gpu-memory-utilization 0.4006`).
