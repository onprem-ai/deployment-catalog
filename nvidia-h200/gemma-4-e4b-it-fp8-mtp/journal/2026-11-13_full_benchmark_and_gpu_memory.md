# 2026-11-13 - Full Benchmark and GPU Memory

**Goal:** Measure the tuned settings (journal 2026-11-13, tuning, round 1) with every workload of the
replica benchmark, and write the measured GPU memory.

## The run

Job `b016b7b4-3460-4973-b729-db88b505cdaf`, all three workloads, 01:44:40 to 02:02:15Z, on inference node
`nebius-e00pn5kkfnayh8wn6j`. It succeeded with no failed request. The file is committed as the control
plane wrote it: `benchmarks/2026-11-13-0144Z-llmperfbench.yaml`, whose `compose_sha256` (`0c367ecb...`)
is the committed `compose.yaml`. A probe right before it reproduced the tuning (290 / 5,775 / 10,987
decode tokens/s at 1 / 32 / 128 requests).

| Workload | Stop | Figures |
|---|---|---|
| `max-prefill/1.0.0` | plateau at 20 | 23,732 prefill tokens/s at 1 (TTFT p50 0.11 s for 8,192 tokens); 75,857 at 32 |
| `max-decode/1.0.0` | 128, still growing | 324 decode tokens/s at 1 (ITL p50 3.1 ms); 6,732 at 32; 12,391 at 128 |
| `mixed-load/1.0.0` | 128, still growing | 291 decode tokens/s at 1; at 128: 38,281 prefill and 7,111 decode tokens/s, TTFT p50 0.20 s |

`max-decode` and `mixed-load` still grew at 128, where the load generator stops: the card shows no
plateau concurrency.

## Answers (through the API gateway, during the run)

The prime count and the train question right with thinking off and on; two parallel tool calls, then a
correct answer from both results; an image (the Q2 total of a rendered table: 255, right);
`/v1/messages` answers.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s: 31,817,990,144 bytes (29.63 GiB), within
the 39 GiB hardware booking (the tuning probe reached 37.1 GiB with 256 requests at once).

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 12,188,931,719 | 11.35 | `Model loading took 11.53 GiB`, minus the drafter |
| `drafters` | 191,311,512 | 0.18 | size of the drafter's safetensors |
| `cuda_graphs` | 1,513,975,972 | 1.41 | `CUDA graph pool memory: 1.41 GiB (actual)` |
| `kv_cache` | 12,863,427,052 | 11.98 | `Available KV cache memory: 11.98 GiB` |
| `runtime_overhead` | 5,060,343,889 | 4.71 | the peak minus every other part |

`GPU KV cache size: 958,940 tokens` (7.3 requests of 131,072). Engine budget 28 GiB
(`--gpu-memory-utilization 0.2003`).
