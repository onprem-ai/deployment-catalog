# 2026-11-10 - Full Benchmark and GPU Memory

**Goal:** Measure Kolibri 1 FP8 with every workload of the replica benchmark, with the
settings chosen in the tuning (256 sequences, 16,384 batched tokens, 1M context, FP8 KV
cache, 120 GiB engine booking, 132 GiB hardware booking), and write its measured GPU memory.

## The runs

Both runs on replica `4282086b-c456-4ece-b808-ec3bf92fdc79`, all three workloads, load
generator 5.2.0-rc.8. The spot VM was not preempted.

1. Job `27292355-d377-41db-9762-5f1f73434ced`, 20:40-21:02Z. `max-prefill` reached its plateau,
   `max-decode` ran to 128. Then the run failed before `mixed-load`: "the calibration request
   failed: ... read: connection reset by peer". It wrote no benchmark file.
2. Job `3a3d95d8-74c7-48d4-b542-4ffbc44fe9e0`, started 21:02:54Z (by the benchmark watch, which
   then restarted a run after any failure; it now restarts only after a preemption), ended
   21:35:01Z as failed: "Workload max-decode/1.0.0: a request to the replica failed at
   concurrency 96: ... read: connection reset by peer". `max-decode` stopped at 96
   (`request-failed`); `mixed-load` ran to 128. Its benchmark file is committed as written:
   `benchmarks/2026-11-10-2102Z-llmperfbench.yaml` (`compose_sha256` `67d8ad5e...`, the
   committed `compose.yaml`).

In both failures the engine kept running, did not restart and logged no error; every other
request of both runs succeeded. In the replica's network namespace the kernel counted 2
`TCPAbortOnClose` (a connection closed with unread data) and no listen or backlog drops. A
race between the load generator reusing a kept-alive connection and the engine closing it is
the likely cause; not confirmed.

## Figures

Run 2, as committed:

| Workload | Stop | Plateau | Figures |
|---|---|---|---|
| `max-prefill/1.0.0` | plateau | 24 | 14,444 prefill tokens/s at 1 (TTFT p50 0.18 s for 8,192 tokens); 47,729 at 24; 49,857 at 32 |
| `max-decode/1.0.0` | request-failed at 96 | - | 162 decode tokens/s at 1 (ITL p50 6.1 ms); 2,795 at 32; 4,382 at 64; 4,327 at 80 |
| `mixed-load/1.0.0` | max-concurrency | - | 154 decode tokens/s at 1; at 128: 3,445 decode and 18,935 prefill tokens/s |

Run 1 adds `max-decode` above 80, from the run's answer (not a committed file): 4,977 at 96,
5,402 at 112, 6,008 at 128 (ITL p50 20.2 ms). Run 1 and run 2 agree within 1 percent at every
level both measured.

`max-decode` and `mixed-load` still grew at 128: the replica serves more than 128 requests at
once (the tuning probe measured 6,687 decode tokens/s at 256).

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during both runs: 134,630,866,944
bytes (125.39 GiB). It was at that value from the first reading of run 1, after the tuning
probes, and did not grow.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 78,866,336,973 | 73.45 | `Model loading took 73.45 GiB` |
| `cuda_graphs` | 923,417,969 | 0.86 | `CUDA graph pool memory: 0.86 GiB (actual)` |
| `kv_cache` | 46,342,697,124 | 43.16 | `Available KV cache memory: 43.16 GiB` |
| `runtime_overhead` | 8,498,414,878 | 7.91 | the peak minus every other part |

`GPU KV cache size: 4,015,963 tokens` (3.83 requests of 1,048,576 tokens). The engine's budget
is 119.99 GiB. Headroom to the 132 GiB hardware booking: 6.62 GiB. Overshoot over the engine
booking: 5.38 GiB, larger than Muse Glimmer's 2.47 GiB; vLLM reports 2.21 GiB of peak
activation inside its budget.

## Next

- Find the cause of the connection resets and run the benchmark once more for a file whose
  workloads all end without a failure.
- Tune the Triton MoE configuration for this expert shape with the GPU alone.
