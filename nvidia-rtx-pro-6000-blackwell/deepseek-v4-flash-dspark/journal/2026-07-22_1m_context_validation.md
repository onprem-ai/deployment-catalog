# 1M-context memory variant -- validated and enabled

**Date:** 2026-07-22
**Basis:** operator request for the full 1,032,192-token window; deployed via
DevSpace + app API (memory_profile=1), validated end-to-end on this node.
Committed as 9eb6a523.

## Measured numbers (2026-07-22, this node)

- KV pool 1,152,376 tokens at util 0.95 (1.12x concurrency at max len);
  weights 79.3 GiB/GPU, KV 8.27 GiB/GPU
- Needle retrieval correct at 272,641 tok (52 s prefill) and 978,408 tok
  (3 m 48 s prefill)
- KV headroom exploited 2026-07-23: --kv-cache-memory 9.75 GiB -> pool
  1,358,228 tokens (1.32x at 1M, +206K over util-0.95), dual 978K needles
  validated. CAUTION: vLLM's "fully utilize" hint (11.21 GiB) and 10.5 GiB
  both OOM on real ~1M prefills -- prefill activations (indexer
  intermediates) GROW with context depth (124 MiB at 12% -> 394 MiB at 40%
  -> ~1 GiB at full depth) and the profiler's short dummy run does not see
  this. Leave >=1 GiB free at peak.

## Why 1M fits in the same 94 GiB budget as 256k

The model is hybrid: sliding-window compressed layer groups
(compress_ratios 4/128, window 128) need max_model_len-INDEPENDENT memory,
so at the 1M config the hybrid allocator gives nearly the whole KV budget
to the full-attention group -- the pool roughly doubles in tokens for the
same GiB vs the 256k config.

## Corrections to earlier journal claims

- "1M ctx boots (KV 1,072,998 tokens)" from the 07-18 bring-up was measured
  on the OLD (pre-fix) image and that boot never served a request (its
  first inference 500'd on the empty-prefill bug). The number happened to
  be roughly right for the hybrid-allocator reason above, not for the
  reason we believed at the time.
- KV cache under TP=2 is REPLICATED per GPU (MLA latent has no head dim to
  shard) -- adding the second GPU doubles compute and weight capacity, not
  KV capacity. Token-sharding it is DCP; see the 2026-07-23 entry.

## Known constraints

- A full 1M request occupies ~90% of the pool: max-length concurrency is
  effectively 1 (shorter requests still batch)
- 256k profile remains memory_profile=0; 1M is memory_profile=1
