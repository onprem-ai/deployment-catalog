# 2026-11-12 - Full Benchmark and GPU Memory

**Goal:** Measure the tuned settings (journal 2026-11-12, tuning, round 1: draft length by load) with
every workload of the replica benchmark, and write the measured GPU memory.

## The run

Job `84f370a2-32aa-47f6-8e5d-f4c89e681cc2`, all three workloads, 23:53:14Z to 00:41:04Z, on inference
node `nebius-e00pn5kkfnayh8wn6j`. It succeeded with no failed request; every workload reached a plateau.
The file is committed as the control plane wrote it: `benchmarks/2026-11-12-2353Z-llmperfbench.yaml`,
whose `compose_sha256` (`ad972dde...`) is the committed `compose.yaml`. A probe right before it
reproduced the tuning (133 / 891 / 2,725 / 3,814 decode tokens/s at 1 / 8 / 32 / 128 requests).

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 16 | 6,668 prefill tokens/s at 1 (TTFT p50 0.61 s for 8,192 tokens); 12,711 at 32 |
| `max-decode/1.0.0` | 80 | 143 decode tokens/s at 1 (ITL p50 7.0 ms); 3,036 at 32; 3,882 at 64 |
| `mixed-load/1.0.0` | 80 | 110 decode tokens/s at 1; at 64: 7,975 prefill and 1,500 decode tokens/s, TTFT p50 0.33 s |

Above 64 to 80 requests the KV cache (389,156 tokens) fills with the benchmark's longer requests: the
time to first token in `mixed-load` jumps to 4.8 s at 96. The tuning showed a larger engine booking
does not add tokens for this model (sliding-window and full-attention layers share the pool); more
requests at once need a second replica.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s: 77,108,084,736 bytes (71.81 GiB), within
the 74 GiB hardware booking.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 34,054,203,484 | 31.72 | `Model loading took 32.59 GiB`, minus the drafter |
| `drafters` | 939,042,560 | 0.87 | size of the drafter's safetensors |
| `cuda_graphs` | 236,223,201 | 0.22 | `CUDA graph pool memory: 0.22 GiB (actual)` |
| `kv_cache` | 37,194,416,783 | 34.64 | `Available KV cache memory: 34.64 GiB` |
| `runtime_overhead` | 4,684,198,708 | 4.36 | the peak minus every other part |

`GPU KV cache size: 389,156 tokens` (2.97 requests of 131,072). Engine budget 72 GiB
(`--gpu-memory-utilization 0.515`).
