# 2026-11-14 - Full Benchmark on RTX PRO 6000

**Goal:** Measure the profile once with the full replica benchmark, on its settings as
published, and write the measured GPU memory.

## Setup

- Inference node `ain-xtg6o` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB, SM120),
  1 GPU; driver 595.112.07, CUDA 13.2.
- The profile as published; served as `qwen36-35b-a3b-bench` during the run.

## Answers

Through the API gateway, "What is 17 * 23?" with thinking off and with the model's default:
HTTP 200, answer '391', 0 reasoning characters; HTTP 200, answer '391', 461 reasoning characters.

## The benchmark

Job `7f09a5c9-4fe5-4f4d-ac51-b133c845f751` on replica `e08ac0d0-d7cf-4785-9a10-f0df577fbe69`, all three workloads,
12:58:21 to 13:17:25Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 20 | 16,007 at 1 (TTFT p50 0.26 s, ITL p50 2.9 ms) | 36,821 at 32 |
| `max-decode/1.1.0` | 40 | 300 at 1 (TTFT p50 0.03 s, ITL p50 3.3 ms) | 3,847 at 48 |
| `mixed-load/1.1.0` | 40 | 279 at 1 (TTFT p50 0.04 s, ITL p50 3.0 ms) | 2,519 at 48 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
62,516,101,120 bytes (58.22 GiB) per GPU. hardware booking raised from 53.00 to 59 GiB; engine booking 53 GiB (the former hardware booking) so the engine keeps its share.

The engine's start-up lines did not give all memory parts, so the profile states no measured parts.
