# 2026-11-20 GPU memory and benchmark: no PLE offload (TP2)

Measured on a copy of this profile in a catalog without an upstream on the hupac control plane
(`drafts`, `qwen38-flash-next-nvidia-fp4-1m-tp2-no-ple-offload`): this profile with one change,
`VLLM_PLE_CPU_OFFLOAD: "0"` (and its display name). This commit applies the same change here.

Why: with the offload (official value `"1"`) each tensor-parallel worker keeps the PLE n-gram table in host RAM
(~20 GB RSS per worker). On ain-2wx1le (60 GiB RAM, no swap, control plane on the same host) the container sat at
its `limits.memory: 48G`, the host thrashed for ~10 h (load ~140, SSH dead) and the model never became ready
(2026-11-19). With `"0"` the table is in GPU memory: host RAM 25 GiB used, 34 GiB available, during this whole run.

## The run

- Replica `f2d7cbdf-dcca-4f6b-bf76-6d9925b9878c` of service `96ba0231-3ca5-40f7-9e83-d1cc48ae9981` on inference node
  ain-2wx1le (`dedfd729-2945-427b-b274-5c506cccda79`), 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, node 5.6.0.
- Engine: `ghcr.io/local-inference-lab/vllm:release-10152026` (pinned by digest in compose.yaml), vLLM
  `v0.1.dev21830+g8244d1c44`.
- Loading: engine start 2026-11-19 18:06:59, weights loaded 18:09:27 (96.6 s), KV cache sized 18:14:43, graph
  capture finished 18:18:35: about 12 minutes from engine start to the end of graph capture (kernels compile at every start).
- Benchmark job `65bccc21-3b52-4d8e-9061-1f3096cf847b`, load generator 5.6.0-rc.5, 2026-11-20T03:12:35.490995Z to 2026-11-20T04:03:34.409807Z, state `succeeded`, no failed request.
  Started with `measure_while_serving_clients: true`: this replica is the only service of `qwen3.8-flash-next`
  on the hupac control plane (no second GPU pair for a test service).
- `vram_used_bytes` of the replica's GPU allocations read every 30 s (102 samples, 03:13:05 to 04:04:02 UTC).

| Workload | Stop | Plateau | req/s | prefill tok/s | decode tok/s | TTFT p50 s | ITL p50 s |
|---|---|---|---|---|---|---|---|
| `max-prefill/1.0.0` | plateau | 16 | 1.952 | 14464 | 119 | 2.39 | 0.100 |
| `max-decode/1.1.0` | plateau | 64 | 2.019 | 638 | 2175 | 0.52 | 0.029 |
| `mixed-load/1.1.0` | plateau | 64 | 3.250 | 6857 | 1143 | 0.77 | 0.058 |

(The figures at the plateau level; every level is in `benchmarks/2026-11-20-0312Z-llmperfbench.yaml`.)

## GPU memory, per GPU (GPU-0695d9c8-…-8d0d, the higher of the two)

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 54932631716 | 51.16 GiB | `Model loading took 51.16 GiB memory and 96.6 seconds` (both workers) |
| `cuda_graphs` | 783831532 | 0.73 GiB | `CUDA graph pool memory: 0.73 GiB (actual), 0.64 GiB (estimated)` |
| `kv_cache` | 36646808453 | 34.13 GiB | `Available KV cache memory: 34.13 GiB` |
| `runtime_overhead` | 7607964139 | 7.09 GiB | the peak minus the three above |
| **peak** | **99971235840** | **93.11 GiB** | highest `vram_used_bytes`, first at 03:13:35 UTC; it did not grow after the first load level |

- Tokens: `GPU KV cache size: 4,467,722 tokens, Maximum concurrency for 1,048,576 tokens per request: 4.26x`
  (official profile with the offload: 6,220,472).
- `drafters` left out: MTP (`num_speculative_tokens: 3`) comes from the checkpoint itself (`model='/models/model'` for
  both), and `model.weights` lists: main:
      provider: huggingface
      repository: nvidia/Qwen3.8-Flash-Next-FP; so vLLM counts it in `Model loading took`.
- Engine budget: `Desired GPU memory utilization is (0.9581, 90.99 GiB)` = the engine booking 97,710,505,984 (91.00 GiB).
- Hardware booking 102,005,473,280 (95.00 GiB): headroom 1.89 GiB (booking minus peak); overshoot 2.11 GiB (peak minus engine
  booking). The peak fits; the bookings stay as they are.
- Idle before the run the replica held 99,486,793,728 per GPU; the load added 0.45 GiB.
- Compared with the official profile: weights +13.4 GiB per GPU (the PLE table, 51.16 vs 37.74 GiB), KV cache
  -13.4 GiB (34.13 vs 47.52 GiB).
- Log warning seen: `Speculative decoding (method=mtp) is enabled but no KV cache
  group could be identified as the draft model's`.
