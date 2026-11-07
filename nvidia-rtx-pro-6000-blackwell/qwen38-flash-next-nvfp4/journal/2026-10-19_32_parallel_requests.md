# 2026-10-19 - 32 parallel requests

## Problem

The profile ran with `--max-running-requests 1`, `--max-mamba-cache-size 16` and
`--cuda-graph-max-bs-decode 1`: the engine served one request at a time. Under
client traffic, every other request waited, and the control plane's test request
timed out after 30 seconds, so the model showed "Not serving" while it served.

## Change

- `--max-running-requests 32`: the target concurrency.
- `--max-mamba-cache-size 192`: SGLang reserves 5 recurrent-state slots per
  running request with `--mamba-radix-cache-strategy extra_buffer` and the overlap
  schedule (3 base, 2 for the extra buffer), so 32 requests need 160; the rest
  keeps prefixes in the cache. A smaller pool caps the running requests and logs
  "max_running_requests is capped ... by the mamba state cache".
- `--cuda-graph-max-bs-decode 32`: CUDA graphs for every decode batch up to 32,
  including the MTP target-verify, draft-decode and draft-extend graphs.
- `--mem-fraction-static 0.83`: room for the larger state pool and the MTP buffer.
- `memory_bytes_per_device`: 88 GiB (94489280512).
- Unchanged: `--max-total-tokens 262144` (the 32 requests share one 262k-token
  KV pool; one request can still use the whole context) and MTP (NEXTN, 3 steps,
  4 draft tokens).

## Measured on the deployment

Startup log on two RTX PRO 6000 Blackwell GPUs:
`max_running_requests=32`, no cap warning, CUDA graphs for batch sizes 1 to 32,
14.60 GB free per GPU after graph capture.

32 requests sent at once to the engine (`max_tokens` 300, `reasoning_effort`
low): `num_running_reqs` 32, `num_queue_reqs` 0, all 32 answered HTTP 200 in 3.4
to 5.8 seconds.

## Still to measure

Throughput and time per output token at 1, 16 and 32 parallel requests through
the API gateway, and the peak memory under long prompts at 32 parallel requests.
