# 2026-11-13 - Embeddings Benchmark on One RTX PRO 6000

**Goal:** Measure the profile once with the replica embeddings benchmark, check its
answers, and write the measured GPU memory. The settings are unchanged.

## Setup

- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, SM120), inference node
  `ain-xtg6o`, driver 595.112.07, CUDA 13.2; GPU 1, beside qwen3-vl-embedding-2b and nomic-embed-v2 (the benchmarks of models sharing a
  GPU ran one after the other).
- The profile as published: vLLM share 0.08, 32,768 tokens of context.

## Answers

Through the API gateway, one request with a query, its answer in English, the answer in
German and an unrelated sentence: cosine similarity 0.878 (answer), 0.831
(German) and -0.003 (unrelated). Both answers rank above the unrelated sentence;
1024 dimensions.

## The benchmark

Job `7485171d-a1a8-48e4-adc1-a4ec1437485c`, 23:28:28 to 23:29:12Z, both
embeddings workloads. It succeeded and no request failed.

| Workload | Stopped | 1 request | Best |
|---|---|---|---|
| `embed-text-query/1.0.0` | still rising at the highest level | 154 inputs/s, 6 ms p50 | 1,544 inputs/s (50,662 tokens/s) at 112, 81 ms p50 |
| `embed-text-indexing/1.0.0` | plateau at 2 | 374 inputs/s, 83 ms p50 | 523 inputs/s (256,512 tokens/s) at 2, 124 ms p50 |

`embed-text-query` sends short queries; `embed-text-indexing` sends long documents, so it
saturates at two requests in flight.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
8.87 GiB. It was above the profile's 8 GiB hardware booking, so the hardware booking is 10 GiB. The engine booking is 8 GiB, the former hardware booking: from it the deploy writes vLLM's share as before (0.08), so the engine is not told the larger booking.

| Part | GiB | Source (vLLM's start-up lines) |
|---|---|---|
| `model_weights` | 1.12 | `Model loading took 1.12 GiB` |
| `kv_cache` | 6.44 | `Available KV cache memory: 6.44 GiB` |
| `runtime_overhead` | 1.31 | the peak minus the other parts |

CUDA graphs took no memory of their own at start (vLLM reported 0 GiB or less), so the
profile states no `cuda_graphs` part. `GPU KV cache size: 60,288 tokens`.
