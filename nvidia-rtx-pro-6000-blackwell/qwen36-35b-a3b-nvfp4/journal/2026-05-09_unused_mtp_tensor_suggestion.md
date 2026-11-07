# Session: model_mtp.safetensors Downloaded but Speculative Decoding Not Enabled

**Date:** 2026-05-09
**Model:** Qwen3.6-35B-A3B NVFP4 (RedHatAI)
**Hardware:** NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date - is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

---

## Finding

The catalog `source.files` list pulls down `model_mtp.safetensors` (~1.57 GiB), but
the current `inference_service` args do **not** enable speculative decoding. The
MTP head sits on disk and consumes VRAM at load time without being used for
generation acceleration.

Current args (no speculative decoding):

```
python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_MOUNT_PATH} \
  --served-model-name ${MODEL_ID} \
  --host 0.0.0.0 --port 8080 \
  --trust-remote-code \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
  ${MEMORY_PARAMS} \
  --quantization compressed-tensors \
  --reasoning-parser qwen3 \
  --moe-backend marlin \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --chat-template ${MODEL_MOUNT_PATH}/chat_template.jinja \
  --max-cudagraph-capture-size 128
```

Compare with the sibling catalog item `qwen3-coder-next-fp8-mtp/inference.yaml`,
which does activate MTP via:

```
--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
```

That model has `speculative_decoding: true` in `capabilities` and is documented
as MTP-enabled. Our Qwen3.6 entry does not.

---

## Why This Matters

MTP (Multi-Token Prediction) speculative decoding typically delivers ~1.5-2× decode
TPS uplift on Qwen3-Next-family hybrid GDN models with `num_speculative_tokens=2`,
because most accepted drafts let the model emit multiple tokens per forward pass.
We are currently leaving that on the table while still paying the disk and
VRAM cost of the MTP head.

The Qwen3-Coder-Next-FP8-MTP catalog item reports 180 TPS @ concurrency 1; the
same model without MTP typically benches noticeably lower. Our current Qwen3.6
NVFP4 entry shows 195 TPS without MTP — there is reason to expect this could
clear 250+ TPS with MTP enabled, though that is a hypothesis, not a measurement.

---

## Suggested Change

Add to the vLLM args list:

```
--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'
```

And update `capabilities`:

```yaml
capabilities:
  ...
  speculative_decoding: true
```

### Open Questions Before Enabling

1. **Method name.** The `qwen3_next_mtp` registration name comes from the
   Qwen3-Next family. Qwen 3.6 uses the same `Qwen3_5MoeForConditionalGeneration`
   architecture (per `config.json`) and the same hybrid GDN backbone, so
   `qwen3_next_mtp` should apply — but verify against vLLM v0.19.1 source or
   nightly recipes before merging. If a `qwen3_5_mtp` or `qwen36_mtp` method has
   landed, prefer that.
2. **Memory headroom.** MTP adds intermediate state for draft tokens. The
   sibling `qwen3-coder-next-fp8-mtp` was marked `unsupported` because
   "FP8 model + MTP speculative decoding + CUDA graphs exceeds 96GB VRAM —
   OOM during sampler warmup." Our model is much smaller (NVFP4, ~22 GiB
   weights vs. ~80 GiB FP8), so that specific failure mode shouldn't apply,
   but re-measure VRAM at 32k / 128k / 262k contexts after enabling.
3. **CUDA graph interaction.** Same sibling catalog item uses `--enforce-eager`
   to dodge a CUDA graph + MTP OOM. Our config currently relies on
   `--max-cudagraph-capture-size 128`. Test whether MTP + CUDA graphs holds
   on NVFP4 weights; if not, fall back to `--enforce-eager` or reduce the
   capture size.
4. **`num_speculative_tokens` tuning.** Qwen recipes default to 2. Some
   workloads benefit from 3, but draft acceptance rate drops and the cost of
   rejected drafts grows. Start at 2; measure draft acceptance via
   `vllm.metrics`.

### Validation Plan

1. Deploy with `--speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}'`
   added, no other changes.
2. Confirm startup completes (watch for sampler warmup OOM seen in the FP8 sibling).
3. Re-run the existing decode TPS benchmark (single-user, 5k token prompt,
   long generation). Target: ≥195 TPS baseline; expect 250-350 TPS with MTP.
4. Re-run prefill TPS benchmark to confirm no regression.
5. Re-measure VRAM at 32k context — update `resources.memory[*].min_gi` if
   the floor moved.
6. Update this catalog item's `capabilities.speculative_decoding` and
   `display.performance` rows.

---

## Sources

- vLLM Qwen3-Next recipe (defines `qwen3_next_mtp` method):
  https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3-Next.md
- Sibling catalog item using MTP on Qwen3-Next family:
  `qwen3-coder-next-fp8-mtp/inference.yaml`
- Qwen3-Next-80B-A3B-Instruct model card (MTP description):
  https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- RedHatAI Qwen3.6-35B-A3B-NVFP4 model card (lists `model_mtp.safetensors`):
  https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4

---

## Status

**Not implemented** — this is a suggestion only. No code or config changes have
been made. The changes above should be made in a separate commit after the
open questions are resolved.
