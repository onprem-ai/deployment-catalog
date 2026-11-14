# 2026-11-14 - Full Benchmark on RTX PRO 6000

**Goal:** Measure the profile once with the full replica benchmark, on its settings as
published, and write the measured GPU memory.

## Setup

- Inference node `ain-xtg6o` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB, SM120),
  1 GPU; driver 595.112.07, CUDA 13.2.
- The profile as published; served as `gemma-4-26b-a4b-it-bench` during the run.

## Answers

Through the API gateway, "What is 17 * 23?" with thinking off and with the model's default:
HTTP 200, answer '391', 0 reasoning characters; HTTP 200, answer '391', 0 reasoning characters.

## The benchmark

Job `2d1ecf63-c852-406d-bb99-e674859f4dfc` on replica `da6fa223-4bcf-4547-a17b-7a6699e836cb`, all three workloads,
10:47:41 to 11:37:10Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 20 | 12,349 at 1 (TTFT p50 0.24 s, ITL p50 6.2 ms) | 33,323 at 32 |
| `max-decode/1.1.0` | 128 | 179 at 1 (TTFT p50 0.03 s, ITL p50 5.6 ms) | 5,786 at 160 |
| `mixed-load/1.1.0` | 128 | 165 at 1 (TTFT p50 0.05 s, ITL p50 5.6 ms) | 3,269 at 128 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
59,905,146,880 bytes (55.79 GiB) per GPU. hardware booking raised from 55.00 to 56 GiB; engine booking 55 GiB (the former hardware booking) so the engine keeps its share.

Measured parts from the engine's start-up lines: model weights 15.87 GiB, KV cache 37.49 GiB, CUDA graphs 0.32 GiB; runtime overhead is the peak minus these. KV cache size 3,287,168 tokens.
