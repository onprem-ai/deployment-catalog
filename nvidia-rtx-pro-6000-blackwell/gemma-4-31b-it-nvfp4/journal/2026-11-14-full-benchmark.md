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

Job `a590a88a-6d8c-41b5-bfdd-75951593d019` on replica `a626a902-b462-480e-ba5a-d4c32bb222a5`, all three workloads,
11:08:25 to 12:56:47Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 16 | 2,830 at 1 (TTFT p50 1.02 s, ITL p50 29.0 ms) | 7,581 at 32 |
| `max-decode/1.1.0` | 64 | 36 at 1 (TTFT p50 0.05 s, ITL p50 27.8 ms) | 1,369 at 96 |
| `mixed-load/1.1.0` | 48 | 34 at 1 (TTFT p50 0.09 s, ITL p50 27.8 ms) | 662 at 64 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
61,576,577,024 bytes (57.35 GiB) per GPU. hardware booking raised from 54.00 to 58 GiB; engine booking 54 GiB (the former hardware booking) so the engine keeps its share.

Measured parts from the engine's start-up lines: model weights 31.04 GiB, KV cache 21.87 GiB, CUDA graphs 0.41 GiB; runtime overhead is the peak minus these. KV cache size 47,760 tokens.
