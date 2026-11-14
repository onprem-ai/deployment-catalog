# 2026-11-14 - Full Benchmark on RTX PRO 6000

**Goal:** Measure the profile once with the full replica benchmark, on its settings as
published, and write the measured GPU memory.

## Setup

- Inference node `ain-xtg6o` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GiB, SM120),
  1 GPU; driver 595.112.07, CUDA 13.2.
- The profile as published; served as `qwen36-35b-a3b-bench` during the run.

## Answers

Through the API gateway, "What is 17 * 23?" with thinking off and with the model's default:
HTTP 200, answer '391', 0 reasoning characters; HTTP 200, answer '391', 625 reasoning characters.

## The benchmark

Job `52ec0d05-3593-4fcd-a2f1-13f6a36f7a68` on replica `9e2065a4-25a0-4916-aa1d-cc3b0d50a612`, all three workloads,
13:22:57 to 13:44:30Z. It succeeded; 0 requests failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 4 | 16,239 at 1 (TTFT p50 0.26 s, ITL p50 3.0 ms) | 29,463 at 16 |
| `max-decode/1.1.0` | 4 | 297 at 1 (TTFT p50 0.04 s, ITL p50 3.2 ms) | 905 at 4 |
| `mixed-load/1.1.0` | 20 | 259 at 1 (TTFT p50 0.04 s, ITL p50 3.1 ms) | 806 at 4 |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
50,390,368,256 bytes (46.93 GiB) per GPU. hardware booking raised from 43.00 to 47 GiB; engine booking 43 GiB (the former hardware booking) so the engine keeps its share.

Measured parts from SGLang's start-up lines: model weights 21.85 GiB, drafter 3.79 GiB, KV cache 16.79 GiB (free memory before the memory pools minus after them), CUDA graphs 3.01 GiB; runtime overhead is the peak minus these. KV cache size 1,155,712 tokens (`max_total_num_tokens`).
