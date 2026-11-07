# 2026-08-14 -- Scheduler Memory Leak: Root Cause + LRU Reference-Cycle Fix

**Symptom**: SGLang predictor evicted by kubelet every 1-3 days under load.
`sglang::scheduler` anonymous-memory usage grows linearly with load, holding
flat through idle days, reset only by a pod restart.

**Pod**: qwen36-35b-a3b SGLang (1m + 262k
variants). Working set exceeds the 8Gi memory request under sustained
traffic, making it the node's preferred eviction victim.

## Root Cause (upstream PR #25770)

Matched to a known, fixed SGLang defect:
[sgl-project/sglang PR #25770](https://github.com/sgl-project/sglang/pull/25770)
(merged 2026-05-20):

> `LRUList._remove_node` leaks scheduler RSS linearly under sustained load on
> hybrid Mamba models (e.g. Qwen3-Next). When multiple nodes are evicted in
> the same radix-cache batch, the dangling `prev`/`next` pointers between
> evicted nodes form 2-element reference cycles that Python refcounting cannot
> reclaim. They wait for gen-2 GC, too infrequent under load. `gc.collect()`
> released ~1.9 GB instantly (reference cycles confirmed).

Maps exactly to our symptoms: load-proportional RSS growth in
`sglang::scheduler`, flat through idle (gen-2 GC too rare), reset only by
restart. Our model is the same hybrid-Mamba class (Qwen3.6: 30 Gated DeltaNet
+ 10 Full Attention) with `--mamba-scheduler-strategy extra_buffer` and the
radix cache enabled.

### Correction to earlier investigation

The upstream link cited in the earlier investigation (sgl-project/sglang#6914) is closed
(2025-08-21, inactive) -- it is NOT the live leak source. The live candidate
is the mamba radix-cache fix family (PR #25770 primary).

## The Fix (3 lines)

`LRUList._remove_node` had to clear the removed node's own pointers:

```python
setattr(node, self.prv, None)
setattr(node, self.nxt, None)
```

This breaks the 2-element reference cycles between nodes evicted in the same
batch so refcount can reclaim them immediately instead of waiting for gen-2 GC.

## Implementation: Pure-Python Overlay Image

- `patch_mamba_radix_cache.py` -- idempotent, fails-loudly patch adding the two
  `setattr(...)` lines to `LRUList._remove_node`. Fails if the expected pre-fix
  block is absent, so a future base bump surfaces at build time.
- `Dockerfile` -- starts from the already-GPU-built `onpremai/sglang-blackwell:v0.5.11`
  (NVFP4 CUTLASS JIT kernels for SM120a baked in) and overlays the 3-line Python
  fix. **No GPU rebuild required.**
- `build.sh` -- builds `onpremai/sglang-blackwell:v0.5.11-memleak-fix` (+ `--push`).
- `README.md` -- why/what/deploy targets.

## Build + Push (2026-08-14)

- Pulled the base image `onpremai/sglang-blackwell:v0.5.11` (digest
  `sha256:210865f8...`, matches Docker Hub).
- Built locally: patch applied cleanly (`OK: applied LRU leak fix to
  .../srt/mem_cache/mamba_radix_cache.py`), ~3s, no GPU required.
- Verified at runtime (CPU-only test container): `sglang.srt` imports OK and
  `inspect.getsource(LRUList._remove_node)` contains both clearing lines.
- Pushed to Docker Hub.

**Pushed image**:

```
onpremai/sglang-blackwell:v0.5.11-memleak-fix
digest: sha256:88f67f7a2d97500a3c3634702a014824803c96e237f200b64fb8049de55b784b
```

## Catalog Config Update

Both SGLang hybrid-Mamba Qwen3.6 variants pin the new digest:

- `qwen36-35b-a3b-sglang` (262k, `extra_buffer`)
- `qwen36-35b-a3b-sglang-1m` (1M, the variant that exhibited the leak)

`image:` updated from unpinned `onpremai/sglang-blackwell:v0.5.11` to the
pinned memleak-fix digest.

## Validation Plan (before/after)

1. Deploy the new image to the affected node.
2. Confirm the eviction cycle stops -- the scheduler no longer RSS-grows under
   sustained load.
3. Watch for load-proportional growth returning; hold flat through idle days.

## Open Questions

- Separate decision: full base bump to a newer SGLang (v0.5.13 minimum contains
  the fix; latest v0.5.17) as the proper long-term fix. Requires the 90-120 min
  GPU kernel rebuild -- distinct from the fast targeted patch above.
- `qwen36-35b-a3b-nvfp4` is vLLM, not SGLang -- not affected by this leak.
