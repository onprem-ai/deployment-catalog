# Session: Initial Catalog Entry — SGLang with `--mamba-scheduler-strategy extra_buffer`

**Date:** 2026-05-09
**Model:** Qwen3.6-35B-A3B NVFP4 (RedHatAI), served by SGLang
**Hardware:** NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date - is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

---

## Why This Catalog Item Exists

The sibling entry `qwen36-35b-a3b-nvfp4` runs the same model on vLLM. vLLM's
hybrid KV-cache management for Qwen3.6's Gated DeltaNet + Full Attention
backbone is solid but exposes few tuning knobs. SGLang has a **purpose-built**
prefix-caching path for hybrid Mamba/GDN models that vLLM does not match
feature-for-feature today. This catalog item is the SGLang variant of the same
weights so we can A/B benchmark on identical hardware.

The decisive flag is `--mamba-scheduler-strategy extra_buffer`. This entry
also enables EAGLE speculative decoding, which finally puts
`model_mtp.safetensors` (~1.57 GiB, downloaded by both entries) to work.

---

## What `extra_buffer` Actually Does

`--mamba-scheduler-strategy extra_buffer` is **not** speculative decoding. It
is a memory-allocation strategy for SSM (state-space model) cache slots in the
hybrid attention path. SGLang has two strategies:

| Strategy | Radix cache | Overlap scheduler | page_size > 1 | Spec dec + radix | Memory cost |
|----------|-------------|-------------------|---------------|------------------|-------------|
| `no_buffer` (default) | yes | no | no | no (need `--disable-radix-cache`) | 1× SSM slot/req |
| `extra_buffer` | yes | yes | yes | yes | 2× for non-spec, ~1.16× with EAGLE/4 draft tokens |

Concretely, with `extra_buffer` we get four wins simultaneously:

1. **Radix prefix cache snapshots SSM state**, not just KV cache. Repeated
   prompts (system prompts, tool definitions, codebase context) get full
   cache hits including the GDN/SSM layers. With `no_buffer`, only the
   attention-layer prefix is cached and SSM state is recomputed every time.

2. **Overlap scheduler** pipelines CPU scheduling with GPU forward passes
   (CPU prepares batch N+1 while GPU runs batch N). Reduces GPU idle time;
   typically 10–20% throughput gain at moderate concurrency, more at high
   concurrency. PR #22542 reports up to **+37% throughput** for non-MTP
   workloads at high concurrency.

3. **Branching-point caching** stores SSM state at radix-tree split points,
   so prefix splits no longer destroy SSM reuse.

4. **Speculative decoding compatibility**. SGLang's `_handle_mamba_radix_cache()`
   raises a hard `ValueError` if you try to combine speculative decoding with
   the radix cache under `no_buffer`:
   > "Speculative decoding for Qwen3_5MoeForConditionalGeneration is not
   > compatible with radix cache when using `--mamba-scheduler-strategy no_buffer`.
   > To use radix cache with speculative decoding, please use
   > `--mamba-scheduler-strategy extra_buffer` and set `SGLANG_ENABLE_SPEC_V2=1`."

That is why this catalog item sets both:
- `--mamba-scheduler-strategy extra_buffer`
- `SGLANG_ENABLE_SPEC_V2=1` (env var)

---

## Cost of `extra_buffer`

- **2× SSM state slots per request** (non-spec) or **~1.16×** with EAGLE at
  4 draft tokens. SSM state on hybrid GDN is small relative to KV cache, so
  on a 96 GiB RTX 6000 Blackwell this overhead is rarely the binding
  constraint — KV cache for long context dominates.
- **CUDA-only**. The FLA kernel backend is required; ROCm is not supported.
  Fine for our hardware.
- **Page-size constraint**: `--page-size` must satisfy
  `FLA_CHUNK_SIZE % page_size == 0` or `page_size % FLA_CHUNK_SIZE == 0`,
  with `FLA_CHUNK_SIZE=64`. We use `--page-size 64`.
- **Do not combine with `--enable-mixed-chunk`** — confirmed accuracy
  regression (PR #22876 raises a `ValueError`). On GSM8K, accuracy dropped
  93.8% → 87.6% under concurrency. We do not enable mixed-chunk in this
  config.
- **Known race condition with chunked prefill** in the overlap scheduler
  (open issue #24221, May 2026). Watch for accuracy drift at high
  concurrency until that lands.

---

## When `extra_buffer` Pays Off (and When It Doesn't)

Per SGLang's own docs:

> `extra_buffer` is strictly better for non-KV-cache-bound cases; for
> KV-cache-bound cases, the tradeoff depends on whether enabling overlap
> outweighs reduced max running requests.

For our use cases on this node:

| Workload | extra_buffer wins? |
|----------|---------------------|
| Agentic coding with repeated context (large system prompt + tools repeated across calls) | yes — radix cache hit rate is high, big TTFT speedup |
| Multi-user serving | yes — overlap scheduler compounds throughput uplift |
| Single-user 1M-token document analysis with unique prompts | marginal — 0% radix cache hit, only overlap scheduler benefit |
| Speculative decoding (EAGLE/MTP) requested | mandatory — no_buffer would force `--disable-radix-cache` |

---

## Why EAGLE Speculative Decoding (and not NEXTN)

The official Qwen 3.6 SGLang cookbook recipe uses
`--speculative-algorithm EAGLE` for `Qwen3.6-35B-A3B-FP8`. Note that this
differs from the **Qwen3-Next** family (used by `qwen3-coder-next-fp8-mtp`),
where SGLang exposes the algorithm as `--speculative-algo NEXTN`.

The EAGLE configuration is also identical to what the Qwen 3.6 cookbook
publishes:

```
--speculative-algorithm EAGLE
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

`model_mtp.safetensors` (already downloaded by both catalog entries) acts as
the draft head. With `extra_buffer` set, the radix cache stays active during
spec dec, so we keep prefix-cache hits across drafted tokens.

---

## Open Questions Before Marking Stable

The following items must be verified on first deployment. Maturity is
`experimental` until they are.

1. **NVFP4 quantization flag for SGLang.** vLLM uses
   `--quantization compressed-tensors` for the RedHatAI NVFP4 checkpoint.
   SGLang has historically used `--quantization modelopt_fp4` for NVFP4
   weights (see `nemotron-3-super-120b-a12b-nvfp4` journal). Our checkpoint
   is in compressed-tensors format, not modelopt format. The current YAML
   sets `modelopt_fp4` but this may need to flip to `compressed-tensors`
   if SGLang has added that loader. The Nemotron journal also reports
   `KeyError: 'model.layers.1.mixer.experts.w2_weight_scale'` on SGLang
   nightlies from March 2026 — verify SGLang has caught up with NVFP4
   weight key conventions for hybrid GDN MoE before relying on this.
2. **SGLang image tag.** `lmsysorg/sglang:latest` is set as a placeholder.
   Pin a specific tag once verified working. The tag must include:
   - PR #14792 (mamba radix cache for overlap scheduler) — merged Dec 2025
   - PR #22542 (ping-pong → direct radix slot, +37% throughput) — Apr 2026
   - PR #22908 (auto-fixup for spec dec + radix cache) — Apr 2026
   - Qwen3.6 model support
3. **`--page-size 64` interaction with NVFP4 weight pack alignment.**
   FLA requires `FLA_CHUNK_SIZE % page_size == 0`. NVFP4 packs weights
   per-block; verify no alignment conflict at load.
4. **`--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`** —
   confirmed in SGLang's Qwen3.6 cookbook (works for the FP8 variant).
   Verify they work identically on the NVFP4 variant.
5. **Vision encoder (`model_visual.safetensors`).** vLLM sibling has weight
   loading warnings ("Parameter visual.blocks.X not found in params_dict").
   SGLang's loader path may differ; test or disable vision via
   `--language-model-only` if it adds VRAM pressure without working.
6. **Memory profile re-measurement.** Numbers in `resources.memory` are
   extrapolated from the vLLM sibling plus an estimated extra_buffer +
   EAGLE overhead. Re-measure at 32k / 128k / 262k contexts.

---

## Validation Plan

1. Pull the SGLang image, deploy at 32k context first to confirm startup.
2. Run a single-shot inference smoke test (text-only).
3. Confirm reasoning mode emits thinking tokens.
4. Confirm tool calling parses correctly with `qwen3_coder`.
5. Run the standard decode TPS benchmark from the vLLM sibling
   (single-user, 5k token prompt). Compare against vLLM's 195 TPS.
6. Run a prefix-reuse benchmark: send the same 50K-token system prompt
   repeatedly with different tail queries. Measure TTFT — this is where
   `extra_buffer` should shine.
7. Re-measure VRAM, update `resources.memory[*].min_gi`.
8. Promote to `stable` once 1–7 pass; otherwise log specific failures here.

---

## Sources

- **Qwen 3.6 SGLang official cookbook** (EAGLE settings, `--reasoning-parser`,
  `--tool-call-parser`):
  https://cookbook.sglang.io/autoregressive/Qwen/Qwen3.6
- **SGLang server arguments — Mamba Cache** (extra_buffer description, FLA
  page-size constraint, memory cost):
  https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md
- **SGLang docs — Qwen3-Next Usage** (MambaRadixCache strategies):
  https://docs.sglang.io/basic_usage/qwen3.html
- **SGLang PR #14792** — mamba radix cache for overlap scheduler:
  https://github.com/sgl-project/sglang/pull/14792
- **SGLang PR #22542** — ping-pong dual buffer → direct radix slot
  (+37% throughput at high concurrency):
  https://github.com/sgl-project/sglang/pull/22542
- **SGLang PR #22908** — speculative decoding + radix cache resolution
  (auto-enables `extra_buffer` + `SGLANG_ENABLE_SPEC_V2=1` on CUDA):
  https://github.com/sgl-project/sglang/pull/22908
- **SGLang PR #22876** — `--enable-mixed-chunk` + `extra_buffer` accuracy
  regression on concurrency (GSM8K 93.8% → 87.6%):
  https://github.com/sgl-project/sglang/pull/22876
- **SGLang Issue #24221** — chunked prefill snapshot race in overlap
  scheduler (open as of May 2026):
  https://github.com/sgl-project/sglang/issues/24221
- **SGLang Issue #18590** — Qwen3.5/Qwen3-Next + Blackwell (SM120) tracking:
  https://github.com/sgl-project/sglang/issues/18590
- **Gated DeltaNet paper** (architectural background for what the SSM cache
  is actually caching): https://arxiv.org/abs/2412.06464
- **Sister catalog item** — `qwen36-35b-a3b-nvfp4/inference.yaml` (vLLM
  baseline, same weights, 195 TPS measured 2026-04-24).
- **Prior NVFP4-on-SGLang attempt** (cautionary tale, Nemotron):
  `nemotron-3-super-120b-a12b-nvfp4/journal/2026-03-16_initial_framework_testing.md`

---

## Status

**Experimental — not yet deployed or validated.** The configuration is
written based on documented SGLang behavior and the official Qwen 3.6
cookbook. All numbers in `resources.memory` are estimates. Do not promote
this entry to `stable` until the validation plan above completes
successfully.
