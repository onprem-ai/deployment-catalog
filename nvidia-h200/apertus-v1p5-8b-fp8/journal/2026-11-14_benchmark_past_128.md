# 2026-11-14 - Full Benchmark Past 128 Requests

**Goal:** The full benchmark with the 1.1.0 text workloads (release 5.4.0-rc.1), whose ladder continues past
128 requests in flight (onprem-ai/platform-alpha#340): the 2026-11-13 benchmark still grew at 128.

Same settings, unchanged `compose.yaml` (`compose_sha256` 5d5376e3...). Job on inference node
`nebius-e00pn5kkfnayh8wn6j`, 00:41 to 01:02Z, succeeded, no failed request; file
`benchmarks/2026-11-14-0041Z-llmperfbench.yaml`.

| Workload | Plateau | Best decode tokens/s | Time to first token there (p50) |
|---|---|---|---|
| `max-prefill/1.0.0` | 16 | 364 at 16 | 1.33 s |
| `max-decode/1.1.0` | 128 | 14,572 at 128 | 0.55 s |
| `mixed-load/1.1.0` | 80 | 4,381 at 96 | 0.82 s |

`max-decode` stops at 128 to 192: the engine runs 256 sequences, and the 176,480-token KV cache fills there.
The GPU held at most 30.50 GiB (32,747,028,480 bytes), within the 36 GiB hardware booking.
