# 2026-07-07 -- Attempt 2 FAILED: microscale fused-set matcher; runtime patch applied

## What happened

Two-pass model_free_ptq, pass 1 (NVFP4A16 on MLP) crashed during
`validate_safetensors_index`:

```
ValueError: Matched a re:.*(attn|attention)\.k_proj\.weight$ twice before
completing set (model.layers.10.self_attn.k_proj.weight,
model.layers.11.self_attn.k_proj.weight)
```

## Root cause -- the gate_proj bug reincarnated

llm-compressor 0.12.0's model_free microscale code
(`entrypoints/model_free/microscale.py`) hardcodes LLaMA-style fused module
sets: (q_proj, k_proj, v_proj) and (gate_proj, up_proj). For microscale
schemes (NVFP4) it eagerly matches these sets per shard file so fused modules
can share a global weight scale (vLLM merges them into qkv_proj /
gate_up_proj GEMMs at load time).

Two independent problems for this checkpoint:

1. The upstream Apertus shards pack tensors non-contiguously, so layer N's
   q/k/v land in different files. The eager matcher sees layer 11's k_proj
   while layer 10's set is still open -> ValueError. (This one is what
   `llmcompressor.reindex_fused_weights` exists for.)
2. Apertus has NO gate_proj (xIELU, non-gated MLP). up_proj opens a
   (gate, up) set that can never complete -> would crash even on perfectly
   reindexed shards. Reindexing cannot fix a module that does not exist.
   This is the same architecture assumption that broke the 8B oneshot run in
   June (then in `update_fused_layer_weight_global_scales`); the oneshot code
   was since refactored, but the assumption reappeared in the model_free path.

## Why clearing the mappings is correct (not a workaround hack)

Global-scale fusion only matters for module sets that vLLM merges into a
single GEMM. In this recipe:

- MLP up_proj/down_proj are the only microscale (NVFP4) targets. Apertus has
  no gate_proj, so vLLM cannot merge a gate_up_proj; up_proj stands alone,
  down_proj is never part of any fused set.
- Attention q/k/v/o are FP8_DYNAMIC (per-channel), not microscale -- no
  global scales to fuse.

So for Apertus the fused-mapping table is a semantic no-op, and clearing it at
runtime is exact:

```python
import llmcompressor.entrypoints.model_free.microscale as microscale
microscale.DEFAULT_FUSED_MAPPINGS.clear()
microscale._DEFAULT_FUSED_MAPPINGS_LIST.clear()
```

(Implemented as `disable_fused_global_scale_mappings()` in
`quantize_apertus_70b_three_tier.py`, called before the passes.)

## Prevention / upstream

- File an upstream issue on vllm-project/llm-compressor: model_free microscale
  validation crashes on (a) non-gated MLP architectures (no gate_proj) and
  (b) checkpoints whose fused partners span shard files. The June 8B journal
  already flagged the gate_proj assumption as "worth reporting upstream" --
  it has now bitten twice; report it this time.
- If a future llm-compressor version fixes set matching, drop the patch and
  retest with `reindex_fused_weights` for case (a).

## Outcome

Patch applied, attempt 3 running.
