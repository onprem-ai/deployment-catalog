# 1M profile KV pool expanded via --kv-cache-memory -- +206K tokens free

**Date:** 2026-07-23
**Basis:** operator request to reclaim idle VRAM at the 1M profile after the
DCP work showed ~3 GiB/GPU unused at util 0.95. Applied to the production
(replicated, non-DCP) 1M variant; deployed via DevSpace + app API and
stress-validated on this node.

## Measured numbers (2026-07-23, this node)

- --kv-cache-memory 9.75 GiB (10468982784): KV pool 1,358,228 tokens,
  1.32x concurrency at max len, +205,852 tokens over the util-0.95
  baseline (1,152,376). Decode unchanged (~219 tok/s, DSpark active).
- Validated: two back-to-back 978K-token needle retrievals (4 m 23 s and
  4 m 02 s prefill, correct answers, zero worker restarts), plus short
  battery.

## The OOM bisection that set the value (do not retune blindly)

vLLM's boot log offers a "fully utilize gpu memory" kv-cache-memory hint.
It is a CEILING, not a recommendation:

| kv-cache-memory | pool (tokens) | real ~1M prefill result |
|---|---|---|
| 11.21 GiB (vLLM hint) | 1,561,144 | OOM at ~12% depth (124 MiB alloc, 97 MiB free) |
| 10.5 GiB | 1,462,761 | OOM at ~40% depth (394 MiB alloc, 373 MiB free) |
| 9.75 GiB | 1,358,228 | PASSES at full depth, ~1 GiB free at peak |

Root cause: prefill activations GROW with accumulated context -- the sparse
indexer's per-chunk intermediates scale with how much context the chunk
attends over (measured allocation demand 124 MiB -> 394 MiB -> ~1 GiB
extrapolated at full 1M depth). The engine's memory profiler uses a short
dummy run and never sees this growth, so its "fully utilize" figure leaves
zero headroom for exactly the workload the 1M profile exists for.

## Rule of thumb for this model

Leave >= 1 GiB per GPU free at peak prefill depth when setting
--kv-cache-memory. Validate any retune with a full-depth (~978K token)
needle request -- short prompts pass at values that crash on real 1M
prefills (both failed values answered the short battery correctly first).

## Known constraints

- Value is specific to this image/config (weights 79.3 GiB/GPU, max 32
  seqs, capture sizes to 48, fp8 kv). Rebake or config changes shift it.
- The engine self-recovers from a prefill OOM (worker restart, ~2 min
  outage) -- bad values degrade, they do not brick.
- Remaining theoretical headroom (~190K tokens) IS the 1M prefill
  activation budget; it is not reclaimable without fixing activation
  growth upstream (e.g. chunked indexer logits).
