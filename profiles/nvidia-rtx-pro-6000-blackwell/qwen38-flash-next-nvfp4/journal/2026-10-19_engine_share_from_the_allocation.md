# 2026-10-19 - engine memory share set from the allocation

## Problem

The profile allocated 88 GiB on each GPU (`memory_bytes_per_device: 94489280512`)
but started SGLang with `--mem-fraction-static 0.83`, which is 79.3 GiB of an empty
RTX PRO 6000 Blackwell (95.6 GiB). The deploy profile checker reported the
disagreement (`replica.engine-gpu-memory`), and the shipped-catalog sweep failed on it.

`memory_bytes_per_device` is the single source of truth for a replica's memory on each
GPU (`docs/architecture/gpu-memory-allocation.md`). The engine's share is a fraction of
the memory free when SGLang starts, so it is fitted to the allocation, never the other
way round.

## Change

- `--mem-fraction-static 0.92`: 88 GiB / 95.6 GiB is 0.9206; 0.92 is 87.9 GiB of an
  empty GPU, within the checker's rounding tolerance of one hundredth of the GPU
  (0.96 GiB).
- Unchanged: `memory_bytes_per_device` (88 GiB) and every other engine option.

## Still to measure

Not yet run on the GPUs. The share outside SGLang's static pool falls from 16.3 GiB to
7.6 GiB per GPU, at the low end of the 5 to 8 GB SGLang's tuning guide recommends for
activations and CUDA graphs. The previous entry measured 14.60 GB free per GPU after
graph capture at 0.83, so about 5.6 GB is expected at 0.92. Measure:

- the startup log: no cap warning, CUDA graphs for batch sizes 1 to 32, and the free
  memory after graph capture;
- 32 parallel requests with long prompts: no out-of-memory error, and the peak memory.

If it runs out of memory, lower `memory_bytes_per_device` and the share together, from
the measured peak.
