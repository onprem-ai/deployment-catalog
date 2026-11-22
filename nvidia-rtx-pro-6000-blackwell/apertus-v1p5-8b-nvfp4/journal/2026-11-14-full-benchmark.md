# 2026-11-14 - Full Benchmark on RTX PRO 6000

**Goal:** Measure the profile once with the full replica benchmark, on its settings as
published, and write the measured GPU memory.

## Setup

- Inference node `ain-xtg6o` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB, SM120),
  1 GPU; driver 595.112.07, CUDA 13.2.
- The profile as published; served as `apertus-v1.5-8b-bench` during the run.

## Answers

Through the API gateway, "What is 17 * 23?" with thinking off and with the model's default:
HTTP 200, answer '391', 0 reasoning characters; HTTP 200, answer '391', 0 reasoning characters.

## The benchmark

Job `bceedc9b-eccf-4ecc-8e0d-7cebbbb0fc89` on replica `ae0ed867-a281-4073-9d56-f5c02df9ad7f`, all three workloads,
10:12:38 to 10:47:43Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 12 | 10,645 at 1 (TTFT p50 0.40 s, ITL p50 5.6 ms) | 18,288 at 32 |
| `max-decode/1.1.0` | 128 | 186 at 1 (TTFT p50 0.02 s, ITL p50 5.3 ms) | 8,255 at 160 |
| `mixed-load/1.1.0` | 80 | 173 at 1 (TTFT p50 0.04 s, ITL p50 5.3 ms) | 2,186 at 96 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
28,944,891,904 bytes (26.96 GiB) per GPU. hardware booking raised from 25.00 to 27 GiB; engine booking 25 GiB (the former hardware booking) so the engine keeps its share.

Measured parts from the engine's start-up lines: model weights 8.15 GiB, KV cache 16.3 GiB, CUDA graphs 0.45 GiB; runtime overhead is the peak minus these. KV cache size 267,024 tokens.
