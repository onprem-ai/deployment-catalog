# 2026-11-11 - Full Benchmark and GPU Memory

**Goal:** Measure the chosen settings (journal 2026-11-11, first deployment and tuning) with
every workload of the replica benchmark, and write the measured GPU memory.

## The run

Job `24328f88-d382-4497-9404-b6552a6ec33e` on replica `8e293406-b3d8-4552-9dd3-f91e05f8c0a8`,
all three workloads, 07:42:10 to 08:24:23Z, load generator 5.2.0-rc.8. It succeeded: every
workload ended at its plateau and no request failed (the connection resets of the Kolibri runs,
issue onprem-ai/platform-alpha#330, did not occur). The spot VM was not preempted. The control
plane runs 5.2.0 and wrote the file in format 1.0.0; it is committed as written:
`benchmarks/2026-11-11-0742Z-llmperfbench.yaml`, whose `compose_sha256` (`810cbed1...`) is the
committed `compose.yaml`.

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 16 | 4,807 prefill tokens/s at 1 (TTFT p50 1.15 s for 8,192 tokens); 7,217 at 24 |
| `max-decode/1.0.0` | 48 | 192 decode tokens/s at 1 (ITL p50 4.9 ms); 1,041 at 8; 2,086 at 32; 2,430 at 48; 2,526 at 80 |
| `mixed-load/1.0.0` | 20 | 121 decode tokens/s at 1; at 32: 4,130 prefill and 822 decode tokens/s, TTFT p50 0.71 s |

The probe of the same settings gave 218 and 205 decode tokens/s for one request; the
benchmark's 192 is lower because its prompts and answers differ (256-token prompts,
1,024-token answers), and the drafter accepts fewer tokens late in a long answer.

Prefill is this model's weak side on one H200: 65B dense parameters per prompt token, with
FP8 weights, give about 4,800 to 7,200 prefill tokens/s (Kolibri, 3.5B active: 14,444 to
49,857). `mixed-load` plateaus at 20 for the same reason.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the run: 130,904,227,840
bytes (121.91 GiB). It was at that value from the first reading and did not grow.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 66,453,881,487 | 61.89 | `Model loading took 61.89 GiB` in round 6, the same weights without the drafter |
| `drafters` | 8,300,024,300 | 7.73 | `Model loading took 69.62 GiB` with the drafter, minus the weights |
| `cuda_graphs` | 4,337,916,969 | 4.04 | `Actual usage is ... 4.04 GiB for CUDAGraph memory` |
| `kv_cache` | 50,337,016,709 | 46.88 | `Available KV cache memory: 46.88 GiB` |
| `runtime_overhead` | 1,475,388,375 | 1.37 | the peak minus every other part |

`GPU KV cache size: 453,109 tokens` (1.73 requests of 262,144 tokens). The engine's budget is
119.99 GiB (`--gpu-memory-utilization 0.8583`). Headroom to the 132 GiB hardware booking: 10.09
GiB; overshoot over the engine booking: 1.91 GiB.

## Next

- The 1.73 requests of full context are the card's limit for long work. The FP8 KV cache
  doubles it (916,550 tokens in round 3, answers unchanged in the check); a
  `compose.long-context.yaml` alternative with it and a larger `--max-model-len` (the model
  supports 2M positions) would serve long contexts.
- A higher engine booking: the peak leaves 10.09 GiB to the hardware booking.
- Rerun on a vLLM release that holds Ripple support, to leave the nightly image.
