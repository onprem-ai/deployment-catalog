# 2026-08-16 - Follow-up TODO

**Goal:** Record the remaining resource and performance validation work for Muse Glimmer 30B NVFP4 plus DFlash.

## TODO

- Run a VRAM usage bisection to determine the minimum working `min_gi` for the 131,072-token FP8 KV cache profile. The current 75 GiB value is only the first validated working allocation, not the measured floor.
- Run full speed measurements through the application benchmark API, including:
  - Single-user decode throughput.
  - Valid prefill throughput measurement.
  - Balanced workload latency and throughput.
  - Long-context prefill behavior.
  - Concurrency scaling across multiple request levels.
  - DFlash acceptance length and its effect on throughput.
- Update `inference.yaml` only with measured results. Do not retain invalid metrics such as the negative prefill throughput returned by the initial benchmark run.

## Outcome

The remaining VRAM bisection and comprehensive performance measurements are explicitly tracked for a future validation session.
