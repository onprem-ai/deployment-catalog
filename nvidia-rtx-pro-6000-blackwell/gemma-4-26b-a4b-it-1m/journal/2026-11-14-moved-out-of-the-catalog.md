# 2026-11-14 - Moved Out of the Catalog: llama-server Closes the Benchmark's Connection

**Goal:** Benchmark the profile once on RTX PRO 6000 Blackwell (inference node `ain-xtg6o`
on platform-a4ec), as part of benchmarking every RTX PRO 6000 chat profile without one.

## What happened

- It started and answered ("17 * 23" gave 391, with and without thinking).
- The full replica benchmark (job `ccb0b66a-ef24-4fdb-9c92-2b011df94afd`) failed every
  workload on its first request ("write: connection reset by peer"); a second `max-prefill`
  run failed the same way ("write: broken pipe").
- The replica stayed ready and did not restart. Through the API gateway it answered prompts
  of 19,939 and 39,652 tokens, two at once, with `ignore_eos`, streamed and not.
- The engine's output at that moment was not kept. A likely cause: the load generator reuses
  the connection of its calibration request, and llama-server has closed it as idle by the
  time the first measured request is written.

The investigation is tracked in an issue of this repository.

## Outcome

The profile cannot be deployed as published, so it moved from `main` to the branch
`drafts/undeployable`. It returns to `main` by pull request once it completes the full replica benchmark.
