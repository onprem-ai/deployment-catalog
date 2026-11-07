# 2026-07-07 -- Attempt 1 FAILED: oneshot + accelerate disk offload

**Goal:** Run the three-tier recipe via `oneshot` with the 146 GB model loaded
through accelerate disk offload (`device_map="auto"`, `max_memory={0: "12GiB",
"cpu": "22GiB"}`, `offload_folder=...`), since the node has only 64 GB RAM.

## What happened

Model loading itself worked (accelerate builds the offload map lazily, ~6 s).
`oneshot` then crashed in its pre-processing step:

```
File ".../llmcompressor/entrypoints/utils.py", line 93, in pre_process
    from_accelerate(model_args.model)
File ".../compressed_tensors/offload/convert/from_accelerate.py", line 155
    assert offload.device.type == "meta"
AssertionError
```

## Root cause

llm-compressor 0.12.0 converts accelerate-managed models to its own offload
representation (`from_accelerate`). That converter asserts every offloaded
module presents as `meta`-device. A mixed GPU + CPU + disk `device_map` from
`max_memory`/`offload_folder` violates this assumption. The documented,
supported pattern is `device_map=None` (whole model resident in CPU RAM) --
see examples/big_models_with_sequential_onloading -- which needs ~146 GB RAM
and is impossible on this node.

Note: the 8B-era `gate_proj` bug did NOT appear; the crash is earlier and
unrelated. The gate_proj fused-layer code is indeed gone in 0.12.0.

## Decision: two-pass model_free_ptq instead

`model_free_ptq` streams safetensors and already worked here (FP8 baseline in
1.8 min), but accepts only ONE scheme per invocation. Its config writer is
`create_or_update_quant_config` -- explicitly supports merging into a
pre-existing quantization_config. So:

- Pass 1: scheme NVFP4A16, ignore attention + lm_head + embed_tokens
  -> NVFP4 MLP weights, everything else passes through as BF16.
- Pass 2: input = pass 1 output, scheme FP8_DYNAMIC, ignore MLP modules +
  lm_head + embed_tokens -> FP8 attention weights added, config merged.

Skipping `llmcompressor.reindex_fused_weights` (required for microscale schemes
with fused modules): Apertus has no fused modules -- q/k/v are separate
projections and there is no gate_proj (xIELU MLP), so there are no global
scales to fuse across a shard boundary.

Operational note: model_free_ptq validation loads a full ~50 GB shard onto the
GPU per worker (learned during the FP8 run -> max_workers=1), so qwen36 must be
undeployed again for the duration; redeploy after the vLLM smoke tests.

## Outcome

Attempt 1 abandoned. Two-pass script next.
