# 2026-11-14 - Full Benchmark on RTX PRO 6000

**Goal:** Measure the profile once with the full replica benchmark, on its settings as
published, and write the measured GPU memory.

## Setup

- Inference node `ain-xtg6o` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB, SM120),
  1 GPU; driver 595.112.07, CUDA 13.2.
- The profile as published; served as `gemma-4-e4b-it-bench` during the run.

## Answers

Through the API gateway, "What is 17 * 23?" with thinking off and with the model's default:
HTTP 200, answer '391', 485 reasoning characters; HTTP 200, answer '391', 397 reasoning characters.

## The benchmark

Job `b4695d65-a32e-4dba-9d54-fb37c3779726` on replica `efb06fa6-5424-41f7-a1dd-bf48db93f9c9`, all three workloads,
13:23:37 to 14:01:20Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 24 | 11,167 at 1 (TTFT p50 0.20 s, ITL p50 8.4 ms) | 48,597 at 32 |
| `max-decode/1.1.0` | 128 | 140 at 1 (TTFT p50 0.03 s, ITL p50 7.1 ms) | 8,830 at 192 |
| `mixed-load/1.1.0` | 128 | 129 at 1 (TTFT p50 0.03 s, ITL p50 7.1 ms) | 4,643 at 128 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
25,497,174,016 bytes (23.75 GiB) per GPU. hardware booking raised from 22.00 to 24 GiB; engine booking 22 GiB (the former hardware booking) so the engine keeps its share.

Measured parts from the engine's start-up lines: model weights 9.82 GiB, KV cache 11.3 GiB, CUDA graphs 0.32 GiB; runtime overhead is the peak minus these. KV cache size 123,376 tokens.
