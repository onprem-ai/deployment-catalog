# 2026-11-14 - Full Benchmark on RTX PRO 6000

**Goal:** Measure the profile once with the full replica benchmark, on its settings as
published, and write the measured GPU memory.

## Setup

- Inference node `ain-xtg6o` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB, SM120),
  1 GPU; driver 595.112.07, CUDA 13.2.
- The profile as published; served as `gemma-4-31b-it-bench` during the run.

## Answers

Through the API gateway, "What is 17 * 23?" with thinking off and with the model's default:
HTTP 200, answer '391', 0 reasoning characters; HTTP 200, answer '391', 0 reasoning characters.

## The benchmark

Job `71db0f18-45e8-419b-9ac9-b54970e04af0` on replica `b8ea0ea5-3dbb-417c-81ff-33ae9e4e9715`, all three workloads,
11:41:07 to 12:54:07Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 8 | 2,583 at 1 (TTFT p50 1.70 s, ITL p50 22.0 ms) | 4,188 at 12 |
| `max-decode/1.1.0` | 40 | 84 at 1 (TTFT p50 0.06 s, ITL p50 11.5 ms) | 1,413 at 40 |
| `mixed-load/1.1.0` | 24 | 57 at 1 (TTFT p50 0.11 s, ITL p50 14.3 ms) | 491 at 32 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
91,159,003,136 bytes (84.90 GiB) per GPU. hardware booking raised from 79.00 to 85 GiB; engine booking 79 GiB (the former hardware booking) so the engine keeps its share.

Measured parts from SGLang's start-up lines: model weights 33.15 GiB, drafter 0.93 GiB, KV cache 44.21 GiB (free memory before the memory pools minus after them), CUDA graphs 1.3 GiB; runtime overhead is the peak minus these. KV cache size 128,497 tokens (`max_total_num_tokens`).
