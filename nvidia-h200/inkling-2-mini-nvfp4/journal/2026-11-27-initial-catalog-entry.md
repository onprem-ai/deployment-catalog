# 2026-11-27 Initial catalog entry (written from the RTX PRO 6000 profile, nothing run on an H200)

**Hardware:** none used. The profile targets one H200 (141 GB).

## Source

`nvidia-rtx-pro-6000-blackwell/inkling-2-mini-nvfp4` as measured on 2026-11-27 (its journal
`2026-11-27-first-start-tuning-and-benchmark.md`): the same image, weights revision and engine
flags. vLLM's recipe (`recipes.vllm.ai/thinkingmachines/Inkling-2-Mini`) lists the H200 as
verified; its H200 page was not read again for this entry.

## Decisions

- **The whole card:** hardware booking 146,028,888,064 bytes (136 GiB) and engine booking
  137,438,953,472 bytes (128 GiB), the values of `nvidia-h200/qwen38-27b-fp8-mtp`, the largest
  in this catalog's H200 profiles. `--gpu-memory-utilization 0.9156` as there; the deploy
  replaces it with the engine booking's share.
- **`--max-model-len 1048576`,** from an estimate: on the RTX PRO 6000 the engine's 84 GiB held
  21.87 GiB of weights, 1.20 GiB of CUDA graphs, a 55.45 GiB KV cache of 861,838 tokens (FP8)
  and 5.48 GiB of other memory. With 128 GiB the KV cache would be about 99.4 GiB, about
  1,545,000 tokens: 1.47 requests of the full context. Not a measurement.
- **CPU limit 6** (8 in the source profile), as asked for this profile. The source profile's
  CPU use was not measured, so whether 6 limits throughput at high concurrency is not known.
- **Host memory limit 16G** kept: the source replica held 11.5 GiB at its benchmark's peak. A
  KV pool almost twice as large may need more host memory for long prompts; not known.
- **FP8 KV cache and 7 DSpark draft tokens** kept from the source profile, where they were
  measured. Not repeated on an H200.
- **No `measured_gpu_memory_bytes_per_device` and no benchmark file:** nothing was measured.

## Not known

- The speed. The H200 does not compute NVFP4 or MXFP8 natively.
- Whether the engine's memory above its booking stays below the 8 GiB left for it (3.9 GiB at
  160 requests in flight on the RTX PRO 6000).
- The driver the nightly's CUDA 13.2 build needs on an H200 host.

## Not done

No start, no answer checks, no benchmark, no measured GPU memory.
