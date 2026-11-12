# 2026-11-11 - Full Benchmark and GPU Memory

**Goal:** Measure the card's settings (NVIDIA Dynamo's H200 recipe, journal 2026-11-11, initial
catalog entry) with every workload of the replica benchmark, and write the measured GPU memory.

## The run

Job `a0f222e7-3f2e-412f-90b1-42f6091b32a9` on replica `201a4f38-b793-4122-b37f-6f62001bfecf`
(inference node `nebius-e00pn5kkfnayh8wn6j`), all three workloads, 23:01:01 to 23:36:32Z. It
succeeded with no failed request. The file is committed as the control plane wrote it:
`benchmarks/2026-11-11-2301Z-llmperfbench.yaml`, whose `compose_sha256` (`b57ea65d...`) is the
committed `compose.yaml`.

| Workload | Stop | Figures |
|---|---|---|
| `max-prefill/1.0.0` | plateau at 20 | 15,956 prefill tokens/s at 1 (TTFT p50 0.26 s for 8,192 tokens); 30,773 at 32 |
| `max-decode/1.0.0` | 128, still growing | 285 decode tokens/s at 1 (ITL p50 3.5 ms); 2,885 at 32; 6,555 at 128 (ITL p50 19.6 ms) |
| `mixed-load/1.0.0` | 128, still growing | 248 decode tokens/s at 1; at 128: 17,799 prefill and 3,416 decode tokens/s, TTFT p50 0.40 s |

`max-decode` and `mixed-load` were still growing at 128, where the load generator stops: the
replica serves more than 128 requests at once (the probe measured 8,457 decode tokens/s at 256,
with the KV cache 7 percent used). The card therefore shows no plateau concurrency.

## GPU memory

The highest `vram_used_bytes`, read every 30 s: 137,688,514,560 bytes (128.23 GiB), the same value
as after the probe before the run; it did not grow during the benchmark.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 68,316,089,428 | 63.62 | `Model loading took 65.09 GiB`, minus the drafter |
| `drafters` | 1,573,765,896 | 1.47 | size of the drafter's `model.safetensors` |
| `cuda_graphs` | 2,050,846,884 | 1.91 | `CUDA graph pool memory: 1.91 GiB (actual)` |
| `kv_cache` | 49,295,487,140 | 45.91 | `Available KV cache memory: 45.91 GiB` |
| `runtime_overhead` | 16,452,325,212 | 15.32 | the peak minus every other part |

`GPU KV cache size: 1,687,841 tokens`. Engine budget 119.99 GiB. Headroom to the 132 GiB hardware
booking: 3.77 GiB; overshoot over the engine booking: 8.23 GiB, the largest of the H200 cards:
FlashInfer's autotuning and the 32,768-token batch with 256 sequences take memory outside vLLM's
budget.

## Next

- A load generator ladder above 128 would find the plateau.
- A smaller engine booking (KV cache is mostly unused at these lengths) would leave room for the
  overshoot; measure the KV cache use at long contexts first.
