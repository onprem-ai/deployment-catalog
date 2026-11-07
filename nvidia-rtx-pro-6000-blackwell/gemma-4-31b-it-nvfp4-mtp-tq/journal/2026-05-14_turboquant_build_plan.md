# Build Plan: SGLang + TurboQuant KV Cache Image

**Date**: 2026-05-14
**Goal**: Build a custom SGLang image with TurboQuant KV cache (PR #23135) to test
3.88x KV compression on Gemma 4 31B NVFP4 + MTP
**Estimated build time**: ~3-5 minutes

## Context

- TurboQuant is not available in our current image (`lmsysorg/sglang:gemma4-mtp`, v0.5.12.dev93)
- FP4 KV cache (`fp4_e2m1`) crashes on Gemma 4 — `torch.zeros` not implemented for `Float4_e2m1fn_x2` in `SWAKVPool`
- TurboQuant uses its own pool class (`MHATokenToKVPoolTurboQuant`), which may bypass the SWA issue
- PR #23135 cherry-picks cleanly onto our image (tested — zero conflicts, single commit)

## Source: PR #23135

- **Repo**: `sgl-project/sglang`
- **Branch**: `liuhuijiayou:feat/turboquant-kvcache`
- **Commit**: `7a2dc8a65044` — "feat(turboquant): TurboQuant KV cache compression"
- **Status**: Open (not merged), mergeable=CONFLICTING with main but CLEAN on our base
- **Size**: 20 files, +4,395 / -50 lines (mostly new files, minimal changes to existing code)
- **PR state as of check**: single squashed commit, all additive

## Cherry-pick verification

Tested inside the running image container:
```
git fetch origin pull/23135/head:pr-23135
git cherry-pick --no-commit pr-23135
# Result: 0 conflicts, clean apply on 9f64d51c7 (our HEAD)
```

Files modified in existing code (merge-critical):
- `server_args.py` — adds `turboquant_*` to `--kv-cache-dtype` choices
- `model_runner.py` — TQ dtype parsing, WHT rotation fusion
- `model_runner_kv_cache_mixin.py` — TQ cell size calculation
- `pool_configurator.py` — TQ pool construction
- `memory_pool.py` — adds `MHATokenToKVPoolTurboQuant`
- `triton_backend.py` — routes to TQ fused kernels
- `flashinfer_backend.py` — adds TQ hooks (raises NotImplementedError for extend)
- `flashattention_backend.py` — adds TQ hooks

New files (no conflict risk):
- `kv_turboquant.py` — config, codebook, memory layout
- `turboquant_decode_attention.py` — fused Triton decode kernel
- `turboquant_extend_attention.py` — fused Triton extend/prefill kernel
- `turboquant_quantize.py` — WHT rotation + norm + quantize + pack kernels
- `hadamard.py`, `hadamard_jit.cuh`, etc. — Walsh-Hadamard transform
- `tq4_decode_stage1.cu` — experimental CUDA JIT decode kernel
- `tq_decode_attention.py` — CUDA kernel Python wrapper

## Build steps

### 1. Build image (~3-5 minutes)

```bash
# Create a temporary container from the base image
docker run -d --name sglang-tq lmsysorg/sglang:gemma4-mtp sleep infinity

# Cherry-pick TurboQuant PR
docker exec sglang-tq bash -c '
  cd /sgl-workspace/sglang
  git fetch origin pull/23135/head:pr-23135
  git cherry-pick pr-23135
'

# Re-install sglang in editable mode (picks up new Python files)
docker exec sglang-tq bash -c '
  cd /sgl-workspace/sglang
  pip install -e "python[all]" --no-deps 2>&1 | tail -5
'

# Verify turboquant is importable
docker exec sglang-tq python3 -c '
  from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig
  print("TurboQuant: available")
'

# Commit as new image
docker commit sglang-tq onpremai/sglang-blackwell:gemma4-mtp-turboquant

# Cleanup
docker rm -f sglang-tq
```

### 2. Verify image

```bash
# Quick smoke test — check kv-cache-dtype choices include turboquant
docker run --rm onpremai/sglang-blackwell:gemma4-mtp-turboquant python3 -c "
from sglang.srt.server_args import ServerArgs
import argparse, re, inspect
src = inspect.getsource(ServerArgs.add_cli_args)
m = re.search(r'kv.cache.dtype.*?choices=\[([^\]]+)\]', src, re.DOTALL)
print('kv-cache-dtype choices:', m.group(1) if m else 'NOT FOUND')
"
# Should include turboquant_4bit, turboquant_4bit_uniform, etc.
```

### 3. Create catalog entry

New entry: `gemma-4-31b-it-nvfp4-sglang-mtp-tq`

Key differences from `gemma-4-31b-it-nvfp4-sglang-mtp`:
- Image: `onpremai/sglang-blackwell:gemma4-mtp-turboquant`
- Launch args: `--kv-cache-dtype turboquant_4bit_uniform` (replaces `fp8_e4m3`)
- Memory profiles: needs full VRAM binary search (expect lower min_gi due to ~1.94x better KV compression vs FP8)

### 4. Deploy and test

1. Deploy with 32k context + generous headroom
2. Verify server starts (watch for SWA pool errors)
3. Run capability tests: text, vision, tool calling, reasoning
4. Measure VRAM (`nvidia-smi`) and KV cache tokens (`max_total_num_tokens`)
5. Run prefill and decode benchmarks
6. Compare against FP8 baseline

### 5. VRAM binary search (if step 4 passes)

- 32k context with TurboQuant: expect min_gi ~35-40 (vs 46 with FP8)
- 128k context with TurboQuant: expect min_gi ~55-65 (vs 79 with FP8)
- 256k context with TurboQuant: try it — might fit in 96 GiB (~85-90 GiB estimated)

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| SWA pool incompatibility | High | TQ uses own pool class; if SWA routing doesn't dispatch to it, server crashes on startup. Detectable immediately. |
| NEXTN MTP interaction | Medium | Drafter model uses its own KV cache. TQ may not apply to drafter (should fall back to default), but untested. |
| Prefill regression | Medium | TQ locks prefill to Triton backend (no FA3/FlashInfer). PR reports 64-94% prefill throughput vs baseline. |
| Triton kernel JIT on first start | Low | One-time ~30-60s delay on first pod start (Triton compiles TQ kernels). Subsequent starts use cache. |
| Quality degradation | Low | PR benchmarks show 79.3% MMLU with TQ4 uniform vs 79.2% bf16. Near-lossless. |
| pip install pulls unexpected deps | Low | Using `--no-deps` flag. If it fails, try without flag but pin versions. |

## Existing image preservation

The current working image (`lmsysorg/sglang:gemma4-mtp`) and catalog entry
(`gemma-4-31b-it-nvfp4-sglang-mtp`) remain completely untouched. The TurboQuant
variant is a separate image and separate catalog entry.

## Expected outcomes

**Best case**: TurboQuant works with Gemma 4 SWA + MTP. We get:
- ~1.94x less KV VRAM vs FP8 at all context lengths
- 256K context feasible on single GPU (first for this model)
- 93-105% decode throughput (minimal penalty)
- Prefill slower but acceptable for long-context use cases

**Worst case**: SWA pool routing doesn't work with TurboQuant pool class.
Server crashes on startup. We learn exactly what's missing and can file
an upstream issue or attempt a fix.
