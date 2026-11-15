# 2026-11-14 - Full Benchmark on RTX PRO 6000

**Goal:** Measure the profile once with the full replica benchmark, on its settings as
published, and write the measured GPU memory.

## Setup

- Inference node `ain-xtg6o` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB, SM120),
  1 GPU; driver 595.112.07, CUDA 13.2.
- The profile as published; served as `jev-9b-bench` during the run.

## Answers

Through the API gateway, "What is 17 * 23?" with thinking off and with the model's default:
HTTP 200, answer '391', 0 reasoning characters; HTTP 200, answer '391', 1199 reasoning characters.

## The benchmark

Job `78849955-e17d-4dc9-aced-b726cf361976` on replica `62daabdb-122e-4015-aaaa-d9aa049b4f64`, all three workloads,
07:53:23 to 08:41:04Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 12 | 7,006 at 1 (TTFT p50 0.40 s, ITL p50 11.9 ms) | 18,291 at 32 |
| `max-decode/1.1.0` | 64 | 85 at 1 (TTFT p50 0.03 s, ITL p50 11.8 ms) | 3,306 at 96 |
| `mixed-load/1.1.0` | 64 | 81 at 1 (TTFT p50 0.07 s, ITL p50 11.8 ms) | 1,711 at 96 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
33,770,438,656 bytes (31.45 GiB) per GPU. hardware booking raised from 30.00 to 32 GiB; engine booking 30 GiB (the former hardware booking) so the engine keeps its share.

Measured parts from the engine's start-up lines: model weights 16.93 GiB, KV cache 10.34 GiB, CUDA graphs 0.12 GiB; runtime overhead is the peak minus these. KV cache size 304,409 tokens.
