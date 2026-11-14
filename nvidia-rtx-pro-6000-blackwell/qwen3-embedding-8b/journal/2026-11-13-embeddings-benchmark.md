# 2026-11-13 - Embeddings Benchmark on One RTX PRO 6000

**Goal:** Measure the profile once with the replica embeddings benchmark, check its
answers, and write the measured GPU memory. The settings are unchanged.

## Setup

- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, SM120), inference node
  `ain-xtg6o`, driver 595.112.07, CUDA 13.2; GPU 0, beside jina-embeddings-v4 (the benchmarks of models sharing a
  GPU ran one after the other).
- The profile as published: vLLM share 0.29, 32,768 tokens of context.

## Answers

Through the API gateway, one request with a query, its answer in English, the answer in
German and an unrelated sentence: cosine similarity 0.780 (answer), 0.741
(German) and 0.163 (unrelated). Both answers rank above the unrelated sentence;
4096 dimensions.

## The benchmark

Job `44add895-002b-4cc2-a377-f842fbf420b6`, 23:28:28 to 23:36:07Z, both
embeddings workloads. It succeeded and no request failed.

| Workload | Stopped | 1 request | Best |
|---|---|---|---|
| `embed-text-query/1.0.0` | plateau at 80 | 60 inputs/s, 16 ms p50 | 423 inputs/s (11,145 tokens/s) at 112, 262 ms p50 |
| `embed-text-indexing/1.0.0` | plateau at 2 | 43 inputs/s, 741 ms p50 | 45 inputs/s (22,770 tokens/s) at 4, 2,871 ms p50 |

`embed-text-query` sends short queries; `embed-text-indexing` sends long documents, so it
saturates at two requests in flight.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
28.67 GiB. It was above the profile's 28 GiB hardware booking, so the hardware booking is 30 GiB. The engine booking is 28 GiB, the former hardware booking: from it the deploy writes vLLM's share as before (0.29), so the engine is not told the larger booking.

| Part | GiB | Source (vLLM's start-up lines) |
|---|---|---|
| `model_weights` | 14.11 | `Model loading took 14.11 GiB` |
| `kv_cache` | 12.54 | `Available KV cache memory: 12.54 GiB` |
| `runtime_overhead` | 2.02 | the peak minus the other parts |

CUDA graphs took no memory of their own at start (vLLM reported 0 GiB or less), so the
profile states no `cuda_graphs` part. `GPU KV cache size: 91,296 tokens`.
