# TODO — Gemma 4 31B IT NVFP4 SGLang MTP

## Completed

- [x] Catalog entry created with measured VRAM (2026-05-14)
- [x] 32k context: min_gi=46 (max_total_num_tokens=35103)
- [x] 128k context: min_gi=79 (max_total_num_tokens=131239)
- [x] Benchmarks: decode 90 tok/s, prefill ~4719 tok/s
- [x] All capability tests passed (text, vision, tool calling, reasoning)
- [x] vLLM MTP entry deprecated

## Remaining

### Upgrade to stable SGLang release
- Currently using `lmsysorg/sglang:gemma4-mtp` (PR #24436 branch, v0.5.12.dev93)
- When SGLang v0.5.12+ stable releases with Gemma 4 MTP support, rebuild using `_docker/sglang-blackwell/build.sh`
- Re-verify all capabilities and re-measure VRAM (CUDA graph sizes may change)

### Track A — Fast Decode (on new SGLang build)
All three PRs are merged into SGLang main, just not in our current image.

- [ ] **trtllm_mha attention backend** (PR #25006, merged)
  - Add `--attention-backend trtllm_mha` to launch args
  - Expected: +13-22% throughput (decode 90 → ~110 tok/s)
  - Requires flashinfer v0.6.10.post1+
  - Verify MTP compatibility (drafter uses own attention path)
- [ ] **FlashInfer Blackwell prefill** (PR #22921, merged)
  - Add `--linear-attn-prefill-backend flashinfer`
  - Expected: +5-19% prefill (especially GDN layers)
  - Stacks with trtllm_mha (different parts of forward pass)
- [ ] **Piecewise CUDA graphs + spec decode** (PR #22128, merged)
  - Remove `--disable-piecewise-cuda-graph` (currently set by image default)
  - Expected: reduced kernel launch overhead
  - Risk: newer code path, test all capabilities
- [x] ~~**Chunked prefill size tuning**~~ — NO IMPROVEMENT (2026-05-14)
  - Tested 16384 and 32768 (default: 8192). No gain at 6k tokens; 32768+max_prefill=32768 was 28% slower at 20k tokens.
  - See `journal/2026-05-14_chunked_prefill_test.md`
- [ ] Re-run full benchmark suite after changes
- [ ] Re-run VRAM binary search (32k and 128k profiles)

### ~~Track B — Long Context via TurboQuant~~ ❌ NOT VIABLE (2026-05-14)
TurboQuant (PR #23135) tested — does NOT compress KV for SWA models.

- [x] ~~**TurboQuant 4-bit KV cache**~~ — TESTED, NOT VIABLE
  - Built custom image, cherry-picked PR #23135, deployed and benchmarked
  - **Root cause**: `MHATokenToKVPoolTurboQuant` does NOT integrate with `SWAKVPool`
  - KV pool allocates in bf16 (2 bytes), not packed 4-bit (0.5 bytes)
  - At same VRAM (min_gi=46): max_total_num_tokens=17,551 vs FP8's 35,103 (**half!**)
  - Decode -30% slower with zero memory benefit
  - Same class of issue as FP4 E2M1 — SWA pool incompatibility
  - See `journal/2026-05-14_turboquant_test.md` for full results
- [x] ~~**256K context profile**~~ — BLOCKED indefinitely
  - TurboQuant doesn't compress on SWA models → no KV savings → still infeasible
  - FP8 estimate: ~138 GiB (exceeds 96 GiB)
  - Needs upstream fix: SWA pool support in TurboQuant or FP4

### ~~Prefill TPS for catalog~~ ✓
- Added `tps_prefill: 4719` to catalog (2026-05-14)

### ~~Maturity upgrade~~ ✓
- Promoted to `stable` (2026-05-14)
