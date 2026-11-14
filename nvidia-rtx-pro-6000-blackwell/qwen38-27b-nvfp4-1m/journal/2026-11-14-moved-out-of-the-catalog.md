# 2026-11-14 - Moved Out of the Catalog: The Engine Resets the Connection Under Load

**Goal:** Benchmark the profile once on RTX PRO 6000 Blackwell (inference node `ain-xtg6o`
on platform-a4ec), as part of benchmarking every RTX PRO 6000 chat profile without one.

## What happened

- It started and answered ("17 * 23" gave 391, with and without thinking).
- The full replica benchmark (job `a22bd43f-6bae-42cc-8a66-23725c14ba4c`) completed
  `max-prefill` and `max-decode` (decode plateau 64, 2,511 tokens/s; 2,637 at 96).
  `mixed-load` ran to 48 requests (1,197 decode tokens/s) and failed on its first request at
  64: "read: connection reset by peer".
- Peak GPU memory during the run: 86.55 GiB, above the 84 GiB hardware booking.
- The engine's output at the reset was not kept, so the cause is not known. Candidates: CUDA
  out of memory under 64 long mixed prefills, or an engine crash in vLLM v0.27.1.

The investigation is tracked in an issue of this repository.

## Outcome

The profile cannot be deployed as published, so it moved from `main` to the branch
`drafts/undeployable`. It returns to `main` by pull request once it completes the full replica benchmark.
