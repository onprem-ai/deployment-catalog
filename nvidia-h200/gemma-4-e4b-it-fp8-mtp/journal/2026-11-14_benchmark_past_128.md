# 2026-11-14 - Full Benchmark Past 128 Requests

**Goal:** The full benchmark with the 1.1.0 text workloads (release 5.4.0-rc.1), whose ladder continues past
128 requests in flight (onprem-ai/platform-alpha#340): the 2026-11-13 benchmark still grew at 128.

Same settings, unchanged `compose.yaml` (`compose_sha256` 0c367ecb...). Job on inference node
`nebius-e00pn5kkfnayh8wn6j`, 01:16 to 01:44Z, succeeded, no failed request; file
`benchmarks/2026-11-14-0116Z-llmperfbench.yaml`.

| Workload | Plateau | Best decode tokens/s | Time to first token there (p50) |
|---|---|---|---|
| `max-prefill/1.0.0` | 24 | 628 at 32 | 0.57 s |
| `max-decode/1.1.0` | 112 | 12,480 at 128 | 0.17 s |
| `mixed-load/1.1.0` | 256 | 8,049 at 256 | 0.31 s |

`max-decode` levels off at 112 to 160 requests; `mixed-load` grows to 256 (8,049 against 7,111 at 128 in
the 1.0.0 run) with the time to first token still at 0.31 s. The GPU held at most 29.84 GiB
(32,042,385,408 bytes), within the 39 GiB hardware booking.

A first run at 00:13Z failed at `mixed-load/1.1.0` 256: one request of the level's sample answered
"connection reset by peer" with the replica still ready (onprem-ai/platform-alpha#330). The repeat had no
failed request.
