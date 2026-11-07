# 2026-08-15 - Official Performance Benchmarks

**Goal:** Run the standard benchmark suite against
the 1M deployment after validating the final `qwen3.8-27b` served model ID.

## Method

All tests ran through `/api/inference-services/benchmark` as Kubernetes Jobs
against the KServe Service URL. Inputs used `num_words`, which generates a new
random prompt per request and avoids static-prefix cache inflation. Hardware
was one RTX PRO 6000 Blackwell; runtime was vLLM 0.27.1 with 1,010,000-token
context, FP8 KV cache, two-token MTP, and an 84 GiB allocation.

## Standard Single-Concurrency Suite

| Test | Random words | Max output | Input tokens | Generation TPS | Prompt TPS | TTFT | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Decode | 50 | 500 | 105 | 129.21 | 119.49 | 1.90 s | 100% |
| Prefill, retry | 1500 | 100 | 1462 | 63.27 | invalid (-28,030,000) | invalid (0 s) | 100% |
| Balanced | 500 | 500 | 518 | 118.16 | 1348.27 | 1.42 s | 100% |
| Long context | 3000 | 100 | 2785 | 50.50 | invalid (-54,820,000) | invalid (0 s) | 100% |

The first prefill run also completed inference but the benchmark serializer
failed on positive infinity:

```text
Error formatting benchmark result: error marshalling JSON:
json: unsupported value: +Inf
```

Its immediate retry serialized but produced impossible negative prompt
throughput and zero TTFT. The long-context test showed the same arithmetic
failure. These values are benchmark-client defects and are not copied into the
catalog. vLLM's coarse ten-second server log observed about 707 prompt tok/s
for the prefill retry and 1,375 prompt tok/s for the 3000-word request, but
those windows are too coarse to serve as official single-request measurements.

The decode result is valid and repeatable enough to use as the catalog's
single-user decode measurement: 129.21 output tok/s.

## Scaling Suite

Parameters: 200 random words, 200 output tokens, concurrency 1,2,4,8,16.

| Concurrency | Generation TPS | Prompt TPS | Max TTFT | Success |
|---:|---:|---:|---:|---:|
| 1 | 121.83 | 1473.80 | 0.55 s | 100% |
| 2 | 137.81 | invalid (-3,832,500) | invalid (0 s) | 100% |
| 4 | 145.81 | 607.41 | 5.16 s | 100% |
| 8 | 629.23 | 5190.23 | 1.22 s | 100% |
| 16 | 950.73 | 7715.39 | 1.63 s | 100% |

All requests succeeded. The non-monotonic concurrency 4 result and invalid
concurrency 2 prompt/TTFT fields show that this very short 200-token run is
useful as an operational scaling probe, not a latency SLO baseline.

## Outcome

The official suite confirms 129.21 tok/s single-user decode and successful
operation through concurrency 16. Prefill and long-context requests completed,
but the benchmark client's TTFT/prompt-throughput arithmetic is unreliable for
those short jobs; no `tps_prefill` is recorded until the API produces a valid
measurement. The benchmark Job was deleted and the model remains Ready.
