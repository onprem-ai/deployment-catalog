# 2026-11-10 - Full Benchmark and GPU Memory

**Goal:** Measure the FP8 card with DFlash 7 with every workload of the replica benchmark,
and write its measured GPU memory.

## The run

`inference_services_change replicas_benchmark_start` on replica
`cd8575d6-3a43-43f9-a5b9-82760c3b4586` (job `994847be-aa74-45cb-a18e-e200152a4ffa`), all
three workloads, started 17:41:43Z, succeeded 18:17:55Z. Load generator 5.2.0-rc.8. The
control plane runs 5.2.0, which writes the benchmark file in format 1.0.0 (date only); it is
committed as written, named after the run's start:
`benchmarks/2026-11-10-1741Z-llmperfbench.yaml`. Its `compose_sha256`
(`1f6b6fb1...`) is the sha256 of `compose.yaml` as committed. No request failed. The spot VM
was not preempted during the run.

| Workload | Stop | Plateau | Figures |
|---|---|---|---|
| `max-prefill/1.0.0` | plateau | 12 | 7,651 prefill tokens/s at 1 (TTFT p50 0.50 s for 8,192 tokens); 15,693 at 32 |
| `max-decode/1.0.0` | plateau | 64 | 323 decode tokens/s at 1 (ITL p50 2.9 ms); 4,779 at 32; 5,438 at 64; 5,598 at 80 |
| `mixed-load/1.0.0` | plateau | 64 | 172 decode tokens/s at 1; at 64: 9,769 prefill and 1,821 decode tokens/s, TTFT p50 0.54 s |

The catalog card shows 341 decode tokens/s (1 / ITL p50 at concurrency 1 of `max-decode`,
2.93 ms; the window's throughput at that level is 323) and 7,651 prefill tokens/s.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s while the benchmark ran
(`inference_services_read get`): 122,907,787,264 bytes (114.47 GiB). It was already at that
value when the benchmark started (after the probes before it) and did not grow.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 34,917,118,591 | 32.52 | `Model loading took 37.28 GiB` minus the drafter |
| `drafters` | 5,111,976,608 | 4.76 | size of the drafter's `model.safetensors` (BF16, loaded as is) |
| `cuda_graphs` | 1,739,461,755 | 1.62 | `CUDA graph pool memory: 1.62 GiB (actual)` |
| `kv_cache` | 73,787,538,145 | 68.72 | `Available KV cache memory: 68.72 GiB` |
| `runtime_overhead` | 7,351,692,165 | 6.85 | the peak minus every other part |

`GPU KV cache size: 3,071,550 tokens`. The engine's budget is 111.99 GiB
(`Desired GPU memory utilization is (0.8011, 111.99 GiB)`, the 112 GiB engine booking).
Headroom to the 120 GiB hardware booking: 5.53 GiB. Overshoot over the engine booking:
2.47 GiB. vLLM reports 5.41 GiB of peak activation inside its budget.

The deploy profile checker reports no findings (30 rules passed).

## Start-up time

The replica is ready about 7 minutes after the deploy (BF16 without DFlash: 2.5 minutes).
From the deployment log: weights load in 20 s; torch.compile 50 s for the target and 13 s
for the drafter (cache under `/root/.cache/vllm/torch_compile_cache`); graph capture of the
first pass 118 s, mostly DeepGEMM warm-up of the FP8 block kernels (1,447 shapes); JIT kernel
warm-up 13 s; about 95 s more before the FlashInfer autotune (cache under
`/root/.cache/flashinfer`); the second graph capture 12 s. `init engine` took 338 s in
total, 64 s of it compilation.

Every one of these caches lives in the container's own filesystem, so each new replica starts
cold. A profile cannot mount a writable cache today: the compose subset allows only the
weights volumes and `container-files/`, both read-only. The ways to shorten it are a follow-up:
an image built with the caches for this model and GPU, a writable cache volume the inference
node keeps per profile (a format and node change, so a design first), or fewer DeepGEMM
shapes by vLLM's own options.

## Outcome

Benchmark file and measured GPU memory committed. The card is ready to publish.
