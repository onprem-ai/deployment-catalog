# 2026-11-13 - Embeddings Benchmark on One RTX PRO 6000

**Goal:** Measure the profile once with the replica embeddings benchmark, check its
answers, and write the measured GPU memory. The settings are unchanged.

## Setup

- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, SM120), inference node
  `ain-xtg6o`, driver 595.112.07, CUDA 13.2; GPU 1, beside jina-v5-text-small and nomic-embed-v2 (the benchmarks of models sharing a
  GPU ran one after the other).
- The profile as published: vLLM share 0.08, 8,192 tokens of context.

## Answers

Through the API gateway, one request with a query, its answer in English, the answer in
German and an unrelated sentence: cosine similarity 0.887 (answer), 0.895
(German) and 0.106 (unrelated). Both answers rank above the unrelated sentence;
2048 dimensions.

## The benchmark

Job `60ea0839-43da-47d4-a543-5d070eeb0edf`, 23:29:29 to 23:31:23Z, both
embeddings workloads. It succeeded and no request failed.

| Workload | Stopped | 1 request | Best |
|---|---|---|---|
| `embed-text-query/1.0.0` | plateau at 24 | 135 inputs/s, 7 ms p50 | 812 inputs/s (24,199 tokens/s) at 32, 39 ms p50 |
| `embed-text-indexing/1.0.0` | plateau at 2 | 160 inputs/s, 196 ms p50 | 185 inputs/s (100,354 tokens/s) at 2, 324 ms p50 |

`embed-text-query` sends short queries; `embed-text-indexing` sends long documents, so it
saturates at two requests in flight.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
7.81 GiB. It stays within the profile's 8 GiB hardware booking; the bookings do not change.

| Part | GiB | Source (vLLM's start-up lines) |
|---|---|---|
| `model_weights` | 4.26 | `Model loading took 4.26 GiB` |
| `kv_cache` | 2.18 | `Available KV cache memory: 2.18 GiB` |
| `runtime_overhead` | 1.37 | the peak minus the other parts |

CUDA graphs took no memory of their own at start (vLLM reported 0 GiB or less), so the
profile states no `cuda_graphs` part. `GPU KV cache size: 20,400 tokens`.
