# 2026-11-14 - Full Benchmark on RTX PRO 6000

**Goal:** Measure the profile once with the full replica benchmark, on its settings as
published, and write the measured GPU memory.

## Setup

- Inference node `ain-xtg6o` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB, SM120),
  2 GPUs; driver 595.112.07, CUDA 13.2.
- The profile as published; served as `deepseek-v4-flash-bench` during the run.
- One change to start it: engine booking 93 GiB (vLLM refused 0.9897 of the GPU).

## Answers

Through the API gateway, "What is 17 * 23?" with thinking off and with the model's default:
HTTP 200, answer '391', 0 reasoning characters; HTTP 200, answer '391', 0 reasoning characters.

## The benchmark

Job `1af94dd9-2f49-4b0f-a84b-0762a5d60d8c` on replica `8c532c89-e333-4986-92a5-fb11625f94f8`, all three workloads,
14:04:58 to 14:46:15Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 8 | 6,780 at 1 (TTFT p50 0.82 s, ITL p50 5.8 ms) | 9,568 at 32 |
| `max-decode/1.1.0` | 8 | 181 at 1 (TTFT p50 0.08 s, ITL p50 5.3 ms) | 639 at 8 |
| `mixed-load/1.1.0` | 12 | 79 at 1 (TTFT p50 0.11 s, ITL p50 5.9 ms) | 462 at 20 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
101,393,104,896 bytes (94.43 GiB) per GPU. hardware booking raised from 94.00 to 95 GiB.

Measured parts from the engine's start-up lines: model weights 79.36 GiB, KV cache 10.55 GiB, CUDA graphs 0.14 GiB; runtime overhead is the peak minus these. KV cache size 616,712 tokens.
