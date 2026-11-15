# 2026-11-15 - Full Benchmark at 1M Context on One RTX PRO 6000

**Goal:** Measure this profile with the full replica benchmark, on its settings as committed
(journal `2026-11-15-as-a-deploy-profile.md`).

## Run

Job `1cff4550-ecf1-4d44-a283-03f89ded266d` on replica `01c613d1-c954-4159-b88f-adc172bc3055`
(inference node `ain-4euk3`, one RTX PRO 6000 Blackwell), all three workloads,
11:38:26 to 12:07:03Z on 2026-11-15. It succeeded; no request failed.

| Workload | Plateau | 1 request | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 16 | 9,031 prefill tokens/s (TTFT p50 0.54 s) | 14,082 at 16 |
| `max-decode/1.1.0` | 16 | 214 decode tokens/s (ITL p50 4.3 ms) | 1,031 at 16 |
| `mixed-load/1.1.0` | 20 | 180 decode tokens/s | 716 decode at 32, 4,251 prefill at 16 |

Decode stops growing at 16 requests: the profile serves 16 sequences at once
(`--max-num-seqs 16`, NVIDIA's value); requests beyond it queue (TTFT p50 16 s at 32).

## Against the current two-GPU card

`qwen38-flash-next-nvfp4` (community NVFP4 checkpoint, SGLang, two GPUs, 262K context),
benchmark of 2026-11-13:

| | This profile, 1 GPU | `qwen38-flash-next-nvfp4`, 2 GPUs |
|---|---|---|
| Context per request | 1,048,576 | 262,144 |
| KV cache tokens | 1,114,309 | 592,576 |
| Decode, 1 request | 214 tokens/s | 208 tokens/s |
| Decode, best | 1,031 at 16 | 2,109 at 64 |
| Prefill, best | 14,082 at 16 | 12,682 at 32 |
| Mixed load, best decode | 716 at 32 | 1,055 at 32 |

Per GPU, this profile gives about the same decode for one request, more prefill, twice the
KV cache tokens of the two-GPU card on half the GPUs, and four times the context. The
two-GPU card serves more requests at once; a two-GPU version of this profile is the next run.

## GPU memory

Highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
100,195,631,104 bytes (93.31 GiB), within the 95 GiB hardware booking (1.7 GiB headroom);
engine booking 93 GiB.

| Part | GiB | Source (start-up lines) |
|---|---|---|
| `model_weights` | 71.71 | `Model loading took 71.71 GiB` (MTP drafter included) |
| `cuda_graphs` | 0.65 | `Graph capturing finished ... took 0.39 GiB` and `0.26 GiB` |
| `kv_cache` | 16.12 | `Available KV cache memory: 16.12 GiB` |
| `runtime_overhead` | 4.83 | the peak minus every other part |

`GPU KV cache size: 1,114,309 tokens` (1.06 requests of 1,048,576 tokens).

## Published name

Measured as a test name; published as `qwen3.8-flash-next`, the name of the profile this one
replaces (`qwen38-flash-next-nvfp4`, community checkpoint), so clients keep working. Only the
routing name and `--served-model-name` changed; the benchmark file's `compose_sha256` is that of
the measured compose file.
