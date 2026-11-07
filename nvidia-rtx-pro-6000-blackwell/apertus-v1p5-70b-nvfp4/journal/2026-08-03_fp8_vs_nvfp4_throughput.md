# 2026-08-03 - FP8 vs NVFP4 side-by-side throughput benchmarks (official)

## Single-user canonical values (for catalog)

Measured via llmapibenchmark per the benchmarking skill:
- **Decode TPS**: num_words=50, max_tokens=500 (minimal prefill, long generation)
- **Prefill TPS**: num_words=1500, max_tokens=100 (large input, short output)

| Metric | NVFP4 | FP8 |
|--------|-------|-----|
| Decode TPS | **30.7** | **19.8** |
| Prefill TPS | **2385** | **2696** |

NVFP4 single-user decode (30.7 TPS) is **55% faster** than FP8 (19.8 TPS).
FP8 single-user prefill (2696 TPS) is **13% faster** than NVFP4 (2385 TPS)
due to lower compute cost per token for FP8 attention.

## Setup

| | NVFP4 | FP8 |
|--|-------|-----|
| Weight size | 48 GiB | 71 GiB |
| VRAM used | ~49 GiB | ~86 GiB |
| GPU util | 0.64 | 0.95 |
| Context | 16384 (16k) | 16384 (16k) |
| CUDA graphs | Yes | No (--enforce-eager) |
| KV cache | fp8_e4m3 | fp8_e4m3 |
| Image | onpremai/vllm-apertus-1p5:p20 | same |
| GPU | RTX PRO 6000 Blackwell (96 GiB) | same |
| Profile | 16k context, min_gi 60, +1 GiB | 16k context, min_gi 83, +8 GiB |

Both benchmarks used `llmapibenchmark` (onpremai/llmapibenchmark:latest) via
the benchmark API (POST /api/inference-services/benchmark). Each concurrency
level sends 10 requests with random prompts (200 words, ~270 tokens input)
and generates 200 tokens output. Metrics come from the model's own
`usage.prompt_tokens` and `usage.completion_tokens` fields.

## Generation TPS (the higher the better)

| Concurrency | NVFP4 gen_tps | FP8 gen_tps | NVFP4 vs FP8 |
|-------------|--------------|-------------|-------------|
| 1 | 29.7 | 19.5 | **1.53x** |
| 2 | 56.5 | 37.8 | 1.50x |
| 4 | 102.9 | 70.9 | 1.45x |
| 8 | 174.5 | 127.2 | 1.37x |
| 16 | 265.5 | 211.5 | 1.26x |
| 24 | 316.9 | 270.0 | 1.17x |
| 32 | 343.7 | 313.1 | 1.10x |
| 40 | 373.0 | 347.1 | 1.07x |
| 48 | 385.3 | 370.1 | 1.04x |
| 56 | 389.5 | 389.9 | 1.00x |
| 64 | 397.7 | 393.1 | 1.01x |

## Prefill TPS (the higher the better)

| Concurrency | NVFP4 prefill_tps | FP8 prefill_tps |
|-------------|------------------|-----------------|
| 1 | 2605 | 2855 |
| 2 | 2572 | 2878 |
| 4 | 2613 | 2901 |
| 8 | 2590 | 2948 |
| 16 | 2584 | 2959 |
| 24 | 2536 | 2930 |
| 32 | 2507 | 2909 |
| 40 | 2465 | 2878 |
| 48 | 2427 | 2834 |
| 56 | 2406 | 2812 |
| 64 | 2381 | 2790 |

## TTFT (Time To First Token, the lower the better)

| Concurrency | NVFP4 max_ttft | FP8 max_ttft |
|-------------|---------------|-------------|
| 1 | 0.33s | 0.29s |
| 2 | 0.66s | 0.58s |
| 4 | 1.31s | 1.16s |
| 8 | 2.61s | 2.32s |
| 16 | 5.27s | 4.60s |
| 24 | 8.08s | 6.98s |
| 32 | 10.89s | 9.39s |
| 40 | 13.74s | 11.74s |
| 48 | 16.83s | 14.35s |
| 56 | 19.69s | 16.94s |
| 64 | 22.87s | 19.54s |

## Key takeaways

- NVFP4 is **53% faster** at single-user generation (29.7 vs 19.5 TPS)
- The gap narrows at higher concurrency as both hit CPU/network bottlenecks
- At concurrency 56+, both models deliver the same generation throughput (~390 TPS)
- FP8 has slightly higher prefill throughput (2855 vs 2605 at conc 1) because
  it has more VRAM pressure forcing larger batch processing
- FP8 has lower TTFT across all concurrency levels (smaller model in terms of
  compute per token, even though more VRAM used)
- All concurrency levels achieved 100% success rate for both models

The earlier ad-hoc measurement (35.6 vs 19.9 TPS) was higher for NVFP4 because
it used a shorter prompt (9 tokens) and included only decode time. The official
benchmark uses a realistic 270-token prompt + TTFT, which is why the numbers
are slightly lower but more representative of real workloads.
