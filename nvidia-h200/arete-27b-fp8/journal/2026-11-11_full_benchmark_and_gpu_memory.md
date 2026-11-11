# 2026-11-11 - Full Benchmark and GPU Memory

**Goal:** Measure the chosen settings (journal 2026-11-11, tuning, round 4) with every workload
of the replica benchmark, and write the measured GPU memory.

## The run

Job `876ea5f1-1026-4f17-9dce-596553878362` on replica `a93eeb0d-d59d-4aaf-92f5-bf76eaa48b1e`, all
three workloads, 16:26:58 to 17:04:22Z. It succeeded: every workload ended at its plateau and no
request failed. The control plane now runs 5.3.0 and wrote the file with the run's start time;
it is committed as written: `benchmarks/2026-11-11-1626Z-llmperfbench.yaml`, whose
`compose_sha256` (`6edd5863...`) is the committed `compose.yaml`.

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 12 | 9,002 prefill tokens/s at 1 (TTFT p50 0.46 s for 8,192 tokens); 15,945 at 32 |
| `max-decode/1.0.0` | 96 | 231 decode tokens/s at 1 (ITL p50 4.1 ms); 3,367 at 32; 4,428 at 96; TTFT p50 5.33 s at 128 |
| `mixed-load/1.0.0` | 40 | 177 decode tokens/s at 1; at 48: 9,026 prefill and 1,666 decode tokens/s |

At 128 requests the time to first token jumps to 5.3 s: the KV cache fills, as found in the
tuning, and requests wait. The plateau of 96 is the practical limit of one replica.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s: 142,738,456,576 bytes
(132.94 GiB), reached during `max-decode` at 128 requests. It was **above the 132 GiB hardware
booking** the tuning had set, so the hardware booking is raised to 136 GiB.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 30,693,456,582 | 28.59 | `Model loading took 32.17 GiB`, minus the drafter |
| `drafters` | 3,848,817,896 | 3.58 | size of the drafter's `model.safetensors` (BF16, loaded as is) |
| `cuda_graphs` | 1,471,026,299 | 1.37 | `CUDA graph pool memory: 1.37 GiB (actual)` |
| `kv_cache` | 96,099,893,248 | 89.50 | `Available KV cache memory: 89.5 GiB` |
| `runtime_overhead` | 10,625,262,551 | 9.90 | the peak minus every other part |

`GPU KV cache size: 2,041,754 tokens` (7.79 requests of 262,144 tokens). Engine budget 127.99
GiB (`--gpu-memory-utilization 0.9155`). Headroom to the new 136 GiB hardware booking: 3.06 GiB;
overshoot over the engine booking: 4.94 GiB. The H200 shows 139.8 GiB to CUDA, so 136 GiB
leaves no room for a second replica on the GPU.

## Next

- A smaller engine booking (for example 124 GiB) would keep the peak further below the
  hardware booking, at the cost of KV cache and so of requests at once; measure it if the GPU
  is to be shared.
- `--gdn-prefill-backend triton` would skip FlashInfer's JIT and shorten the 9-minute start;
  measure whether prefill stays as fast.
- A drafter trained on Arete itself, if Nous publishes one, would raise the acceptance above
  2.7 tokens.
