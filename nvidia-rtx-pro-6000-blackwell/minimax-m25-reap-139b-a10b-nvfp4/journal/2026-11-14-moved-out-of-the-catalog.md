# 2026-11-14 - Moved Out of the Catalog: It Answers Only Zero Bytes

**Goal:** Benchmark the profile once on RTX PRO 6000 Blackwell (inference node `ain-xtg6o`
on platform-a4ec), as part of benchmarking every RTX PRO 6000 chat profile without one.

## What happened

- The first start ran out of GPU memory in FlashInfer's MoE autotuning, right after vLLM
  took 0.98 of the 94 GiB booking ("CUDA out of memory. Tried to allocate 398.00 MiB ...
  73.56 MiB is free"). With an engine booking of 92 GiB it started: `Model loading took
  74.79 GiB`, KV cache 18.5 GiB (78,384 tokens).
- Every answer is the byte 0x00 repeated up to the token limit, on chat completions and
  plain completions, at temperature 0 and 0.7 ("17 * 23", "Name the capital of France in one
  word.", "The capital of France is").

The benchmark was cancelled after its prefill workload; the replica was removed.

## Outcome

The profile cannot be deployed as published, so it moved from `main` to the branch
`drafts/undeployable`. It returns to `main` by pull request once a deployment answers correctly (a newer vLLM image, or the quantization and MoE backend checked against the model's `config.json`), checked before any benchmark.
