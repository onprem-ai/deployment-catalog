# 2026-10-21 - allocation set from the engine's memory share

## Problem

The profile allocated 92 GiB on each GPU (`memory_bytes_per_device: 98784247808`)
but starts vLLM with `GPU_MEMORY_UTILIZATION: "0.95"`, which is 90.8 GiB of an RTX PRO
6000 Blackwell (95.6 GiB). The deploy profile checker's rounding tolerance is one
hundredth of the GPU (0.96 GiB); the 1.18 GiB difference is over it, so the checker
reported it (`replica.engine-gpu-memory`) and the shipped-catalog sweep failed on it.
1.2 GiB booked on each of the eight GPUs stayed unused by the engine and unavailable
to any other replica.

## Change

- `memory_bytes_per_device: 97710505984` (91 GiB): 0.95 of 95.6 GiB is 90.8 GiB,
  0.2 GiB off, within the tolerance.
- Unchanged: `GPU_MEMORY_UTILIZATION` (0.95) and every other engine option, so the
  engine takes exactly what it took before; only the booking follows it.

## Still to check

Nothing to measure on the GPUs: the engine's behavior is unchanged. A separate finding
of the checker stays open: the model's port marker names port 8080 of `inference`,
which that service's command never mentions.
