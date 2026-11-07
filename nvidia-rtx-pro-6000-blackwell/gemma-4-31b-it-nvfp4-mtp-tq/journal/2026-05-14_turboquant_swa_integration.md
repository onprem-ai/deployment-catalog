# TurboQuant SWA Integration Patches — Gemma 4 31B NVFP4 SGLang MTP

**Date**: 2026-05-14
**Goal**: Fix TurboQuant KV cache compression for SWA (Sliding Window Attention) models
**Status**: PAUSED — Compression confirmed (3.06 GB vs 12.05 GB). Blocked on Crash #4 (shared memory 131072 > 101376). Tile tuning exhausted (5 attempts, all 131072). Next: eager dequant fallback for head_dim=512 layers.
**Image**: `onpremai/sglang-blackwell:gemma4-mtp-turboquant-swa`
**Base**: `onpremai/sglang-blackwell:gemma4-mtp-turboquant` (cherry-pick of PR #23135)

## Background

TurboQuant (PR #23135) was tested on Gemma 4 and found NOT to compress KV cache on
SWA models (see `2026-05-14_turboquant_test.md`). Root cause: `SWAKVPool` hardcodes
`MHATokenToKVPool` as its backing pool class, ignoring `MHATokenToKVPoolTurboQuant`.

However, the architecture supports pluggable pool classes — `SWAKVPool.__init__` accepts
a `token_to_kv_pool_class` parameter (defaults to `MHATokenToKVPool`). The plumbing
exists, it's just not wired up for TurboQuant.

## Root Cause Analysis

Three integration gaps were identified:

### Gap 1: Pool class routing (model_runner_kv_cache_mixin.py)

The `is_hybrid_swa` branch in `_init_kv_cache()` (~line 483) constructs `SWAKVPool`
but never passes `token_to_kv_pool_class`. The TurboQuant check
(`if hasattr(self, "turboquant_bits")`) only exists in the `else` branch (non-SWA models).

### Gap 2: tq_config propagation (swa_memory_pool.py)

Attention backends detect TurboQuant via `getattr(pool, "tq_config", None)`.
`forward_batch.token_to_kv_pool` is the `SWAKVPool` wrapper, which has no `tq_config`
even when its backing pools are TurboQuant instances. Without this:
- Decode: TQ fused kernel not selected → falls back to standard (calls `get_key_buffer` → NotImplementedError crash)
- Extend/Prefill: Q not rotated into WHT domain → garbage attention against WHT-rotated cached KV

### Gap 3: Layer-aware buffer access (triton_backend.py)

The TQ fused decode kernel accesses `pool.k_buffer[idx]` directly where
`idx = layer.layer_id - pool.start_layer`. But `SWAKVPool` doesn't have `k_buffer` —
it has two sub-pools (`swa_kv_pool.k_buffer` and `full_kv_pool.k_buffer`) with separate
index spaces. Needs a routing method that maps `layer_id` → correct backing pool + index.

## Patches Applied

All patches applied to `onpremai/sglang-blackwell:gemma4-mtp-turboquant-swa` via
`patch_all_swa_turboquant.py` (copy at `_docker/sglang-blackwell-turboquant/`).

### Patch 1: model_runner_kv_cache_mixin.py — Pool class routing

Inserts TurboQuant pool class selection before `SWAKVPool()` construction in both
SWA branches (NPU and CUDA):

```python
if hasattr(self, "turboquant_bits"):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolTurboQuant
    kwargs["token_to_kv_pool_class"] = MHATokenToKVPoolTurboQuant
    kwargs["turboquant_bits"] = self.turboquant_bits
    kwargs["turboquant_k_bits"] = getattr(self, "turboquant_k_bits", 0)
    kwargs["turboquant_v_bits"] = getattr(self, "turboquant_v_bits", 0)
    kwargs["turboquant_uniform"] = getattr(self, "turboquant_uniform", False)
```

The TQ-specific kwargs flow through `SWAKVPool.__init__` → `**kwargs` → backing pool constructors.
`token_to_kv_pool_class` is consumed as a named parameter by `SWAKVPool`, NOT forwarded.

### Patch 2: swa_memory_pool.py — tq_config propagation + get_tq_layer_pool()

After backing pool creation:
```python
self.tq_config = getattr(self.full_kv_pool, "tq_config", None)
```

New method for layer-aware TQ buffer access:
```python
def get_tq_layer_pool(self, layer_id):
    pool_idx, is_swa = self.layers_mapping[layer_id]
    return (self.swa_kv_pool if is_swa else self.full_kv_pool), pool_idx
```

### Patch 3: memory_pool.py — SWA param acceptance + get_tq_layer_pool()

Added `swa_head_num`, `swa_head_dim`, `swa_v_head_dim` params to
`MHATokenToKVPoolTurboQuant.__init__` and forwarded to `super().__init__()`.
Gemma 4 is `is_hybrid_swa_compress=True`, so these params are present in kwargs.

Added matching method:
```python
def get_tq_layer_pool(self, layer_id):
    return self, layer_id - self.start_layer
```

### Patch 4: triton_backend.py — Layer-aware TQ decode

Changed TQ fused decode kernel to use `get_tq_layer_pool()`:
```python
if hasattr(pool, "get_tq_layer_pool"):
    tq_pool, idx = pool.get_tq_layer_pool(layer.layer_id)
else:
    tq_pool, idx = pool, layer.layer_id - pool.start_layer
# Then: tq_pool.k_buffer[idx], tq_pool.v_buffer[idx], etc.
```

## Known Risks

### Risk 1: get_key_buffer NotImplementedError

`MHATokenToKVPoolTurboQuant._get_key_buffer()` raises `NotImplementedError`.
The extend path calls `pool.get_kv_buffer(layer_id)` when `k is None` (reading
cached KV for continuation). During normal prefill, K/V come from model forward
so this path is skipped. But MTP verification or cache continuation may trigger it.

**If this crashes**: The fix would be adding a dequant path to
`MHATokenToKVPoolTurboQuant._get_key_buffer()` that unpacks uint8 → bf16 on the fly.
Not hard, but increases scope.

### Risk 2: set_kv_buffer through SWAKVPool

`SWAKVPool.set_kv_buffer()` delegates to backing pool with `layer=None` and
`layer_id_override=pool_idx`. TurboQuant's `set_kv_buffer` accepts `layer_id_override`
and uses `idx = layer_id - self.start_layer`. For SWA backing pools `start_layer=0`,
so `idx = pool_idx`. Should work — but needs verification.

### Risk 3: Flashattention extend backend

TurboQuant forces triton for decode only. Extend/prefill may use flashattention
or flashinfer backend, which also checks `tq_config` for Q rotation. Our tq_config
propagation (Patch 2) should make this work, but the extend code paths in
flashattention_backend.py were NOT patched for layer-aware buffer access.
If extend reads from pool (k is None case), it would crash.

## How to Continue

### Step 1: Deploy and test

```bash
# Catalog entry already updated to use the -swa image tag
# Reload catalog (DevSpace must be running):
curl -X POST -H "X-API-Key: $API_KEY" http://localhost:8437/api/catalog/reload

# Deploy:
curl -X POST -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"catalog_id": "rtx6000bw--gemma-4-31b-it-nvfp4-mtp-tq", "memory_profile_index": 0}' \
  http://localhost:8437/api/catalog/deploy
```

### Step 2: Check server logs

Look for:
- `MHATokenToKVPoolTurboQuant` being used (not `MHATokenToKVPool`)
- `TurboQuant K=4bit V=4bit KV cache enabled`
- `SWAKVPool mem usage: X.XX GB` — compare with FP8 baseline (13 GB at min_gi=46)
- `max_total_num_tokens` — should be HIGHER than FP8's 35,103 if compression works

### Step 3: If it starts, run capability tests

```bash
# Text generation
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-31b-it","messages":[{"role":"user","content":"What is 17*23?"}]}'

# Tool calling, reasoning, etc. — same as 2026-05-14_turboquant_test.md
```

### Step 4: If it crashes

Read the error. Most likely failure modes:
- `NotImplementedError` in `_get_key_buffer` → extend path hitting pool read
- `TypeError` unexpected kwarg → signature mismatch in pool constructor
- `KeyError` in `layers_mapping` → layer ID routing bug
- `RuntimeError` CUDA error → tensor shape mismatch in fused kernel

Each tells us exactly which patch needs iteration.

### Step 5: If compression works

- Re-benchmark decode/prefill (expect ~63 tok/s decode, ~6800 tok/s prefill from earlier TQ test)
- VRAM binary search at 32k and 128k contexts
- Evaluate whether the -30% decode penalty is acceptable given the compression benefit
- If 128k fits: this variant becomes the long-context option for Gemma 4

---

## Test Results — Deployment Attempt 1

### Crash 1: TypeError in get_kv_size_bytes (FIXED)

**Error:** `TypeError: cannot unpack non-iterable int object` at `swa_memory_pool.py:108`

```python
k_size, v_size = self.full_kv_pool.get_kv_size_bytes()
```

**Root cause:** `MHATokenToKVPoolTurboQuant.get_kv_size_bytes()` returned a single `int`,
but `SWAKVPool` (and the parent `MHATokenToKVPool`) returns `(k_bytes, v_bytes)` tuple.
API contract violation.

**Fix:** Patch 5 — split return into `(k_total, v_total)` tuple:
```python
def get_kv_size_bytes(self):
    k_total = 0
    v_total = 0
    for i in range(self.layer_num):
        k_total += self.k_buffer[i].nbytes + self.k_dequant_scale_buffer[i].nbytes
        v_total += self.v_buffer[i].nbytes + self.v_dequant_scale_buffer[i].nbytes
    return k_total, v_total
```

Applied via `docker exec` + `docker commit` to running container. Image re-tagged as
`gemma4-mtp-turboquant-swa`, imported into containerd.

### Compression CONFIRMED

After Patch 5 fix, server progressed past pool initialization. Key log output:

```
SWAKVPool mem usage: 5.74 GB, swa size: 74988, full size: 74988
```

Compare with unpatched (FP8 KV cache): **12.05 GB**. That's a **2.1x reduction**,
confirming TurboQuant compression IS working through SWAKVPool.

### Crash 2: RuntimeError signs dim mismatch (FIXED — two-part root cause)

**Error:** `RuntimeError: signs dim must match input dim` from `hadamard_jit.cuh:656`
during CUDA graph capture warmup, in `set_kv_buffer` → `fused_turboquant_quantize_and_store_kv`.

**Root cause:** Gemma 4 has **different head dimensions** for different layer types:
- Full-attention layers: `head_dim=512`, 4 KV heads
- SWA layers: `head_dim=256`, 16 KV heads

Two fixes were needed:

**Fix A — Patch 3b:** `MHATokenToKVPoolTurboQuant.__init__` created `TurboQuantConfig` with
`head_dim=head_dim` (the constructor parameter, always 512). But when constructed as SWA
backing pool, the parent `MHATokenToKVPool.__init__` overrides `self.head_dim = swa_head_dim`
(256) for buffer allocation. Result: signs=512 but tensors=256. Fix:
```python
tq_head_dim = swa_head_dim if swa_head_dim is not None else head_dim
self.tq_config = TurboQuantConfig(head_dim=tq_head_dim, ...)
```

**Fix B — Patch 6:** Both extend and decode paths got `tq_config` from the SWAKVPool wrapper
(`getattr(pool, "tq_config", None)`) which was the full pool's config (signs=512). Even with
Patch 3b fixing the backing pools, the wrapper exposed the wrong one for SWA layers. Fix:
per-layer `get_tq_config(layer_id)` method routes to the correct backing pool's config.

**After fix:** SWAKVPool mem usage dropped to **3.06 GB** (from 5.74 GB in attempt 2, and
12.05 GB baseline). The further reduction is because the SWA pool now correctly uses
head_dim=256 for packed buffer sizing (256/2 = 128 packed dim vs 512/2 = 256 before).

### Crash 3: TurboQuant extend fallback not supported (FIXED)

**Error:** `RuntimeError: TurboQuant extend fallback not supported` during CUDA graph capture.

**Root cause:** The fused TQ extend kernel had a `sliding_window_size == -1` guard. SWA layers
have `sliding_window_size > 0`, causing them to fall to the standard extend path, which
raises RuntimeError when `tq_config is not None`.

**Analysis:** Read the fused TQ extend Triton kernel (`turboquant_extend_attention.py`).
The kernel uses `kv_indptr`/`kv_indices` for indexing into cached KV. The scheduler already
restricts these indices to the sliding window. No window-specific logic exists inside the
kernel — it reads from whatever indices it's given. The `sliding_window_size == -1` guard was
defensive (untested on SWA models), not functional.

**Fix — Patch 7:** Remove the `sliding_window_size == -1` guard and route buffer access
through `get_tq_layer_pool()` (same as Patch 4 did for decode):
```python
if (tq_config is not None
    and tq_config.k_bit_width in (2, 4)
    and tq_config.v_bit_width in (2, 4)
    and not self.enable_deterministic
    and kv_indptr is not None):  # removed: sliding_window_size == -1
    if hasattr(pool, "get_tq_layer_pool"):
        tq_pool, idx = pool.get_tq_layer_pool(layer.layer_id)
    else:
        tq_pool, idx = pool, layer.layer_id - pool.start_layer
```

**Status:** FIXED.

### Crash 4: OutOfResources shared memory (IN PROGRESS — iterating)

**Error:** `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376`

Stack trace points to `turboquant_extend_attention.py:556` → `_fwd_tq_extend_kernel[grid](...)`.
Crash happens during CUDA graph capture warmup when first full-attention layer (head_dim=512)
is compiled.

**Root cause:** The fused TQ extend kernel uses N-way split dot products. Each dot operates
on `[BLOCK_*, K_BLOCK_PACKED_DIM]` tiles where `K_BLOCK_PACKED_DIM = next_power_of_2(head_dim / K_VALS_PER_BYTE)`.
For head_dim=512, 4-bit: K_BLOCK_PACKED_DIM = 256. The Triton compiler allocates shared memory
for dot product operands (loaded via LDMATRIX for MMA instructions). With large packed dims,
the total exceeds Blackwell SM limit (101376 bytes).

**Iteration history:**

| Attempt | BLOCK_M | BLOCK_N | num_stages | Shared mem | Outcome |
|---------|---------|---------|------------|------------|---------|
| v1 (orig) | 64 | 64 | 2 | 131072 | FAIL |
| v5 | 32 | 64 | 2 | 131072 | FAIL — num_stages still doubled tiles |
| v6 | 32 | 32 | 1 | 131072 | FAIL — packed dim (256) dominates, not BLOCK_N |
| v7 | 32 | 16 | 1 | 131072 | FAIL — identical shmem despite 4x smaller tiles |
| v8 | 32 | 16 | 1 | 131072 | FAIL — added debug logging, only Lq=256 printed |

All 5 attempts produced the **exact same 131072 bytes**. Tile size changes have zero effect.
This rules out tile tuning as a solution — the shared memory requirement is not coming from
tile dimensions, or the patches are not reaching the failing compilation path.

**Analysis of why BLOCK_N=32 still needed 131072:** The kernel has TWO dot product phases
per loop iteration:
1. QK: `tl.dot(q_0, k_0) + tl.dot(q_1, k_1)` — operands [32,256]×[256,32] bf16
2. PV: `tl.dot(p_scaled, v_0) + tl.dot(p_scaled, v_1)` — operands [32,32]×[32,256] bf16

The Triton compiler appears to allocate shared memory for all dot operands simultaneously.
With K_BLOCK_PACKED_DIM=256, reducing BLOCK_N from 64 to 32 only halves k/v tiles (256×32→256×16
would be BLOCK_N=16) but q tiles at [32,256] remain large. The compiler's aggregate allocation
still exceeded 101376.

**Fix — Patch 8 v3 (current):** BLOCK_N=16 for head_dim >= 512:
```python
_large_head = Lq >= 512
BLOCK_M = 32 if _large_head else 64
BLOCK_N = 16 if _large_head else 64
num_stages = 1 if _large_head else 2
```

With BLOCK_N=16, K/V tiles shrink from [256,32] to [256,16] bf16 = 8192 bytes each (vs 16384).
Expected shared memory: QK: 2×(32×256×2 + 256×16×2) = 2×(16384+8192) = 49152 bytes.
PV: 2×(32×16×2 + 16×256×2) = 2×(1024+8192) = 18432 bytes. Total ~67584 — well under 101376.

SWA layers (head_dim=256) keep original BLOCK_M=64, BLOCK_N=64, num_stages=2.

**v7 result — STILL FAILS (131072 bytes):**

v7 (BLOCK_M=32, BLOCK_N=16, num_stages=1) was built and deployed. Same error: 131072 bytes required.
This is the third consecutive tile size reduction that produced the *exact same* shared memory number.

**v8 — added unconditional debug logging:**

Added `logging.getLogger("sglang").info(...)` to the tile selection code to print Lq, 
K_BLOCK_PACKED_DIM, BLOCK_M, BLOCK_N, and `_large_head` for EVERY kernel compilation 
(not just head_dim >= 512). Result:

- Logs showed `Lq=256` (SWA layers) with correct BLOCK_M=64, BLOCK_N=64
- **Lq=512 (full-attention layers) debug log was NEVER printed** — the crash likely happens 
  before the log can flush, or the Triton compiler computes shared memory requirements during 
  JIT and aborts before the wrapper code runs
- Image digest verified matching between `docker images` and pod spec — pod IS using the 
  correct new image
- Triton cache cleared in each build (`rm -rf /root/.triton/cache/`)

**Key mystery:** The value 131072 (128 KiB) is suspiciously constant across ALL tile sizes.
Changing BLOCK_M from 64→32, BLOCK_N from 64→32→16, num_stages from 2→1 — none of these
changed the shared memory AT ALL. This strongly suggests either:

1. **The patch isn't affecting the actual kernel launch for head_dim=512 layers** — the 
   Triton compiler may be computing shared memory requirements at JIT compilation time from 
   the kernel's AST/IR, and our changes to the launch-site wrapper code are either not reaching 
   the right call site, or the Triton JIT cache from a prior compilation is being reused despite 
   cache clearing.

2. **K_BLOCK_PACKED_DIM=256 is the bottleneck, not BLOCK_M/BLOCK_N** — the kernel loads 
   Q tiles of shape [BLOCK_M, K_BLOCK_PACKED_DIM] into shared memory. With K_BLOCK_PACKED_DIM=256, 
   even BLOCK_M=32 gives [32, 256] bf16 = 16384 bytes per Q half-tile. The split dot products 
   have 2 Q halves, 2 K halves, 2 V halves per loop iteration — the compiler may need all 
   operands resident simultaneously. But theoretical calculations with BLOCK_N=16 give ~67584 bytes 
   which should fit. So this hypothesis alone doesn't explain the constant 131072.

3. **Triton has a minimum shared memory allocation** or the kernel has additional shared memory 
   usage we're not accounting for (codebook arrays, WHT rotation tables, etc.).

4. **There's a different kernel or compilation path** that also needs 131072 bytes — not the 
   extend kernel at all. The stack trace says `turboquant_extend_attention.py:556` but the line 
   numbers may shift after patching.

**Status:** BLOCKED. Tile size tuning is a dead end. Need a fundamentally different approach.

### Planned Approach: Eager Dequant Fallback (NOT YET IMPLEMENTED)

The most promising path forward is to **skip the fused TQ extend kernel entirely for 
full-attention layers** (head_dim >= 512) and use an eager dequant + standard extend path:

1. In the TQ extend condition (Patch 7 area), add a guard: `layer.qk_head_dim <= 256`
2. In the `else` branch, instead of raising RuntimeError when `tq_config is not None`, 
   implement an eager dequant path:
   - Get the TQ pool's packed uint8 buffers via `get_tq_layer_pool()`
   - Call `batched_dequantize_rotspace()` (from `kv_turboquant.py`) to dequant 
     packed uint8 → bf16 K and V
   - Pass dequantized bf16 KV to `self.extend_attention_fwd()` (standard extend kernel)
3. This preserves the 4-bit KV *storage* benefit (3.06 GB vs 12 GB) while avoiding the 
   fused kernel's shared memory problem on full-attention layers
4. Performance impact: extra dequant step per extend call on full-attention layers only 
   (18 of 48 layers). Decode still uses the fused TQ kernel (which works with smaller tiles 
   for head_dim=512, or uses a separate kernel).

**Implementation steps:**
- Read `kv_turboquant.py` in the container to find `batched_dequantize_rotspace()` signature
- Understand how to call it: inputs = packed uint8 buffers + dequant scales + centroids + 
  tq_config, output = bf16 K and V tensors matching standard extend kernel expectations
- Modify Patch 7 to add the head_dim guard
- Add a new Patch 9 to the `else` branch in triton_backend.py for eager dequant fallback
- Build and test

**Alternative approach to investigate:**
- Check if there's a way to tile over K_BLOCK_PACKED_DIM in the Triton kernel itself 
  (would require modifying the kernel's inner loop, not just launch parameters)
- Check if Triton 3.6.0 has a `num_stages` or `num_warps` option that affects shared memory 
  differently than expected

**Status:** PAUSED — awaiting next session to implement eager dequant fallback.

## Architecture Discovery: Gemma 4 Head Dimensions

Critical finding for any future SWA+TurboQuant work:

| Property | Full-attention layers | SWA layers |
|----------|----------------------|------------|
| head_dim | 512 | 256 |
| num_kv_heads | 4 | 16 |
| kv_bytes per token | 4×512×2 = 4096 | 16×256×2 = 8192 |
| WHT signs size | 512 | 256 |
| sliding_window_size | -1 (infinite) | >0 (windowed) |

Source: `configs/model_config.py` `_derive_model_shapes` reads `head_dim` and `swa_head_dim`
from HuggingFace config. `gemma4_causal.py` selects per-layer head_dim at model construction.

`SWAKVPool` creates two backing pools with `is_hybrid_swa_compress=True`:
- `swa_kv_pool`: `head_num=swa_head_num`, `head_dim=swa_head_dim`
- `full_kv_pool`: `head_num=head_num`, `head_dim=head_dim`

Each TQ backing pool creates its own `TurboQuantConfig` with signs matching its `head_dim`.

## Files

| File | Location |
|------|----------|
| Patch script | `_docker/sglang-blackwell-turboquant/patch_swa_turboquant.py` |
| Build script | `_docker/sglang-blackwell-turboquant/build.sh` |
| Warmup script | `_docker/sglang-blackwell-turboquant/warmup_triton_kernels.py` |
| Catalog entry | `gemma-4-31b-it-nvfp4-mtp-tq/inference.yaml` |
| Image (base TQ) | `onpremai/sglang-blackwell:gemma4-mtp-turboquant` |
| Image (SWA patch) | `onpremai/sglang-blackwell:gemma4-mtp-turboquant-swa` |
| Prior test results | `journal/2026-05-14_turboquant_test.md` |
