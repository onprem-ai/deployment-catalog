# 2026-11-14 - Full Benchmark on RTX PRO 6000

**Goal:** Measure the profile once with the full replica benchmark, on its settings as
published, and write the measured GPU memory.

## Setup

- Inference node `ain-xtg6o` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB, SM120),
  1 GPU; driver 595.112.07, CUDA 13.2.
- The profile as published; served as `jev-27b-vl-bench` during the run.

## Answers

Through the API gateway, "What is 17 * 23?" with thinking off and with the model's default:
HTTP 200, answer '391', 0 reasoning characters; HTTP 200, answer '391', 116 reasoning characters.

## The benchmark

Job `61d46685-230b-426e-8b37-5c980d2285a2` on replica `be7bb1b3-8de4-4b76-b92f-dd79813931ab`, all three workloads,
08:08:28 to 10:13:22Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 8 | 2,116 at 1 (TTFT p50 1.31 s, ITL p50 37.7 ms) | 4,996 at 12 |
| `max-decode/1.1.0` | 8 | 27 at 1 (TTFT p50 0.08 s, ITL p50 37.4 ms) | 201 at 24 |
| `mixed-load/1.1.0` | 12 | 26 at 1 (TTFT p50 0.14 s, ITL p50 37.4 ms) | 178 at 16 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
84,093,698,048 bytes (78.32 GiB) per GPU. hardware booking raised from 75.00 to 79 GiB; engine booking 75 GiB (the former hardware booking) so the engine keeps its share.

Measured parts from the engine's start-up lines: model weights 51.63 GiB, KV cache 20.96 GiB, CUDA graphs 0.08 GiB; runtime overhead is the peak minus these. KV cache size 299,008 tokens.
