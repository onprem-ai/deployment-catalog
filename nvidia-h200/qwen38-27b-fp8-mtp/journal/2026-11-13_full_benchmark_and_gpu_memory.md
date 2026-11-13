# 2026-11-13 - Full Benchmark and GPU Memory

**Goal:** Measure the chosen settings (journal 2026-11-13, tuning: round 0) with every workload of the
replica benchmark, check the answers, and write the measured GPU memory.

## The run

Job `f0d0da69-8096-4bfc-a873-b0d165d7b63d`, all three workloads, 07:49:03 to 08:32:55Z, on inference node
`nebius-e00pn5kkfnayh8wn6j`. It succeeded with no failed request; every workload reached a plateau. The
file is committed as the control plane wrote it: `benchmarks/2026-11-13-0749Z-llmperfbench.yaml`, whose
`compose_sha256` (`cb12b08a...`) is the committed `compose.yaml`. A probe right before it reproduced the
tuning (208 / 3,253 / 4,558 decode tokens/s at 1 / 32 / 128 requests).

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 20 | 9,613 prefill tokens/s at 1 (TTFT p50 0.46 s for 8,192 tokens); 16,573 at 40 |
| `max-decode/1.0.0` | 96 | 219 decode tokens/s at 1 (ITL p50 4.4 ms); 3,572 at 32; 5,001 at 128 |
| `mixed-load/1.0.0` | 80 | 219 decode tokens/s at 1; at 96: 10,400 prefill and 2,038 decode tokens/s, TTFT p50 0.53 s |

Against the Arete 27B H200 card (same architecture, DFlash2 drafter, 177 decode tokens/s at 1): Qwen's
own MTP head gives about 24 % more for one user.

## Answers (through the API gateway, after the run)

The prime count (15) and the train question (205 minutes) right with thinking off and on; two parallel
tool calls (Zurich, Oslo), then a correct answer from both results; an image (the Q2 total of a
rendered table: 255, right); `/v1/messages` answers.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s: 141,536,788,480 bytes (131.82 GiB),
within the 136 GiB hardware booking.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 30,682,785,508 | 28.58 | `Model loading took 29.02 GiB`, minus the MTP head |
| `drafters` | 477,202,224 | 0.44 | size of `mtp.safetensors` (the built-in MTP head) |
| `cuda_graphs` | 2,126,008,812 | 1.98 | `CUDA graph pool memory: 1.98 GiB (actual)` |
| `kv_cache` | 99,793,565,123 | 92.94 | `Available KV cache memory: 92.94 GiB` |
| `runtime_overhead` | 8,457,226,813 | 7.88 | the peak minus every other part |

`GPU KV cache size: 2,579,669 tokens` (9.8 requests of 262,144). Engine budget 128 GiB
(`--gpu-memory-utilization 0.9156`).
