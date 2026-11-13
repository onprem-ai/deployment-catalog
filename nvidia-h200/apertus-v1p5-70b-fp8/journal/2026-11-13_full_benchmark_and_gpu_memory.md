# 2026-11-13 - Full Benchmark and GPU Memory

**Goal:** Measure the first deployment's settings (journal 2026-11-13, tuning) with every workload of
the replica benchmark, and write the measured GPU memory.

## The run

Job `c3863263-8253-4bba-ba3d-ebca478c0b08`, all three workloads, 09:34:04 to 11:34:42Z (2 hours: one
user decodes at 44 tokens/s), on inference node `nebius-e00pn5kkfnayh8wn6j`. It succeeded with no failed
request. The file is committed as the control plane wrote it: `benchmarks/2026-11-13-0934Z-llmperfbench.yaml`;
its `compose_sha256` (`a4db3631...`) is the compose.yaml it measured, unchanged except for the hardware
booking in `ai-deploy.yaml`.

| Workload | Stop | Figures |
|---|---|---|
| `max-prefill/1.0.0` | plateau at 16 | 2,816 prefill tokens/s at 1 (TTFT p50 1.24 s for 8,192 tokens); 6,128 at 32 |
| `max-decode/1.0.0` | 128, still growing | 44 decode tokens/s at 1 (ITL p50 22.7 ms); 1,024 at 32; 3,059 at 128 |
| `mixed-load/1.0.0` | 128, still growing | 42 decode tokens/s at 1; at 128: 4,541 prefill and 873 decode tokens/s, TTFT p50 0.71 s |

## GPU memory

The replica's highest `vram_used_bytes`: 138,326,048,768 bytes (128.83 GiB), during `max-prefill`, 0.8 GiB
above the 128 GiB hardware booking the card started with (the probe and the answer checks had read at
most 124.0 GiB). The hardware booking is now 131 GiB; the engine booking stays 120 GiB.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 75,967,234,048 | 70.75 | `Model loading took 70.75 GiB` |
| `kv_cache` | 47,158,740,910 | 43.92 | `Available KV cache memory: 43.92 GiB` |
| `runtime_overhead` | 15,200,073,810 | 14.16 | the peak minus every other part (CUDA graphs, activations, the encoders' buffers) |

`GPU KV cache size: 287,840 tokens` (4.4 requests of 65,536).
