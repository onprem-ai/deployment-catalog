# 2026-11-13 - Full Benchmark and GPU Memory

**Goal:** Measure the first deployment's settings (journal 2026-11-13, tuning) with every workload of
the replica benchmark, and write the measured GPU memory.

## The run

Job `6bda1d1c-54f7-41cc-9fe1-008a979d5dbd`, all three workloads, 09:01:00 to 09:19:44Z, on inference node
`nebius-e00pn5kkfnayh8wn6j`. It succeeded with no failed request. The file is committed as the control
plane wrote it: `benchmarks/2026-11-13-0901Z-llmperfbench.yaml`; its `compose_sha256` (`5d5376e3...`) is
the compose.yaml it measured, which is unchanged except for the hardware booking in `ai-deploy.yaml`.

| Workload | Stop | Figures |
|---|---|---|
| `max-prefill/1.0.0` | plateau at 16 | 18,015 prefill tokens/s at 1 (TTFT p50 0.18 s for 8,192 tokens); 43,369 at 32 |
| `max-decode/1.0.0` | 128, still growing | 250 decode tokens/s at 1 (ITL p50 4.0 ms); 5,601 at 32; 14,502 at 128 |
| `mixed-load/1.0.0` | plateau at 64 | 236 decode tokens/s at 1; at 64: 24,032 prefill and 4,126 decode tokens/s, TTFT p50 0.07 s |

## GPU memory

The replica's `vram_used_bytes` read 37,534,826,496 bytes (34.96 GiB) from the benchmark's first sample
to its last; it had reached that level during the probe and the answer checks before the run (the probe
read 26.2 GiB in its window; the image request followed). That is 7 GiB above the 28 GiB hardware booking
the card started with: vLLM's start-up profiling does not cover the vision and audio tokenizers' first
real requests. The hardware booking is now 36 GiB; the engine booking stays 24 GiB.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 11,220,602,061 | 10.45 | `Model loading took 10.45 GiB` |
| `kv_cache` | 11,564,199,444 | 10.77 | `Available KV cache memory: 10.77 GiB` |
| `runtime_overhead` | 14,750,024,991 | 13.74 | the peak minus every other part (CUDA graphs, activations, the encoders' buffers) |

`GPU KV cache size: 176,480 tokens` (1.3 requests of 131,072).
