# 2026-11-14 - Full Benchmark Past 128 Requests

**Goal:** The full benchmark with the 1.1.0 text workloads (release 5.4.0-rc.1), whose ladder continues past
128 requests in flight (onprem-ai/platform-alpha#340): the 2026-11-12 benchmark still grew at 128.

Same settings, unchanged `compose.yaml` (`compose_sha256` 8d0abe98...). Job on inference node
`nebius-e00pn5kkfnayh8wn6j`, 22:54 to 23:41Z, succeeded, no failed request; file
`benchmarks/2026-11-13-2254Z-llmperfbench.yaml`.

| Workload | Plateau | Best decode tokens/s | Time to first token there (p50) |
|---|---|---|---|
| `max-prefill/1.0.0` | 20 | 505 at 32 | 0.71 s |
| `max-decode/1.1.0` | 112 | 9,333 at 160 | 0.16 s |
| `mixed-load/1.1.0` | 256 | 6,031 at 256 | 2.62 s |

`max-decode` levels off at 112 to 160 requests (9,333 against 8,656 at 128 in the 1.0.0 run). `mixed-load`
grows to 256 (6,031 against 5,311 at 128), with the time to first token rising to 2.6 s there. The GPU held
at most 57.84 GiB (62,105,059,328 bytes), within the 62 GiB hardware booking.
