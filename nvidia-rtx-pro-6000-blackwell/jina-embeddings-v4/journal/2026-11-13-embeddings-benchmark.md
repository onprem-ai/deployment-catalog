# 2026-11-13 - Embeddings Benchmark on One RTX PRO 6000

**Goal:** Measure the profile once with the replica embeddings benchmark, check its
answers, and write the measured GPU memory. The settings are unchanged.

## Setup

- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, SM120), inference node
  `ain-xtg6o`, driver 595.112.07, CUDA 13.2; GPU 0, beside qwen3-embedding-8b (the benchmarks of models sharing a
  GPU ran one after the other).
- The profile as published: vLLM share 0.15, 8,192 tokens of context.

## Answers

Through the API gateway, one request with a query, its answer in English, the answer in
German and an unrelated sentence: cosine similarity 0.696 (answer), 0.736
(German) and 0.239 (unrelated). Both answers rank above the unrelated sentence;
2048 dimensions.

## The benchmark

Job `c746407f-e9f8-4f9d-ac77-a8b73e9c4b4c`, 23:36:33 to 23:40:03Z, both
embeddings workloads. It succeeded and no request failed.

| Workload | Stopped | 1 request | Best |
|---|---|---|---|
| `embed-text-query/1.0.0` | plateau at 96 | 101 inputs/s, 10 ms p50 | 1,185 inputs/s (25,554 tokens/s) at 96, 76 ms p50 |
| `embed-text-indexing/1.0.0` | plateau at 2 | 93 inputs/s, 339 ms p50 | 100 inputs/s (53,265 tokens/s) at 2, 612 ms p50 |

`embed-text-query` sends short queries; `embed-text-indexing` sends long documents, so it
saturates at two requests in flight.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
13.75 GiB. It stays within the profile's 14 GiB hardware booking; the bookings do not change.

| Part | GiB | Source (vLLM's start-up lines) |
|---|---|---|
| `model_weights` | 7.16 | `Model loading took 7.16 GiB` |
| `kv_cache` | 4.5 | `Available KV cache memory: 4.5 GiB` |
| `runtime_overhead` | 2.09 | the peak minus the other parts |

CUDA graphs took no memory of their own at start (vLLM reported 0 GiB or less), so the
profile states no `cuda_graphs` part. `GPU KV cache size: 131,120 tokens`.
