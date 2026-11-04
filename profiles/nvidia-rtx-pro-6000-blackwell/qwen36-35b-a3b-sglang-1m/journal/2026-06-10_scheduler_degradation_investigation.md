# 2026-06-10 — SGLang Scheduler Degradation (0.3 tok/s)

**Symptom**: Inference not working — status-probe reports 2/3 probes successful,
pod-level probe times out every cycle (30s). Health endpoint takes 13-60s to respond.

**Pod**: Running 1/1, 0 restarts, 30h uptime. Kubernetes thinks it's healthy.

## Root Cause

The sglang scheduler process entered a degraded state where decode throughput
collapsed from ~510 tok/s to **0.3 tok/s** and never recovered. The process is
alive but effectively frozen — GPU utilization at 0% despite 77 GiB VRAM allocated.

## Timeline (UTC)

| Time | Throughput | Event |
|------|-----------|-------|
| 04:05 | ~510 tok/s | Healthy, generating with 160K+ context tokens |
| 05:05 | ~250-530 tok/s | Still healthy, processing multiple conversations |
| 05:11:09 | — | Prefill: 2240 new + 8128 cached tokens |
| 05:11:12 | **10 tok/s** | First throughput drop |
| 05:11:47 | — | Prefill: 192 new + 48704 cached tokens (follow-up to ~49K conversation) |
| 05:11:51 | **6 tok/s** | Stays degraded, 59K tokens across 2 running requests |
| 05:12-05:18 | 3-15 tok/s | Continues degrading |
| 05:30+ | **0.3 tok/s** | Essentially frozen, never recovers |

## Observations

- **KV cache nearly empty**: At 07:33, `full token: 0, mamba num: 0` — no active
  context in cache, yet throughput still 0.35 tok/s. The scheduler is stuck in
  internal bookkeeping, not memory-bound.
- **Detokenizer heartbeat failures**: Continuous "Server couldn't get a response
  from detokenizer for last 20 seconds" — 407 failures in last 5000 log lines.
  The scheduler process is too slow to send heartbeats.
- **Input throughput also collapsed**: Prefill dropped from 85K tok/s to 1-5 tok/s,
  confirming the entire GPU pipeline is throttled, not just decode.
- **No errors in logs**: No OOM, CUDA errors, NCCL failures, or crashes. The
  process degraded silently.

## Cascade Effects

1. Scheduler stuck → health endpoint blocks 13-60s → K8s readiness probe (5s timeout)
   fails intermittently (696+ Unhealthy events in ~2.5h)
2. Status-probe pod-direct probe times out at 30s every cycle → reports 2/3 successful
   (gateway and public URL probes still succeed via API gateway routing)
3. The API gateway intermittently fails to fetch its config

## Configuration at Time of Failure

```
--model-path /model
--context-length 1048576
--max-running-requests 4
--quantization compressed-tensors
--kv-cache-dtype fp8_e5m2
--mem-fraction-static 0.71
--mamba-scheduler-strategy extra_buffer
--page-size 64
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

Image: `onpremai/sglang-blackwell:v0.5.11`

## Resolution

Pod restart required — the scheduler state is unrecoverable without process restart.

## Open Questions

- Is the `extra_buffer` mamba scheduler strategy prone to this degradation under
  concurrent requests with EAGLE speculative decoding?
- Could the 1M context-length configuration with page-size 64 cause memory
  fragmentation that triggers this?
- Should a watchdog mechanism detect throughput collapse and trigger automatic restart?
