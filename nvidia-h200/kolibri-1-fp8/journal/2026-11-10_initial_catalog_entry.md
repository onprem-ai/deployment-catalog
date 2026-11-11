# 2026-11-10 - Initial Kolibri 1 FP8 Catalog Entry for H200

**Goal:** A Kolibri 1 card for one H200 with Aleph Alpha's own FP8 weights and the full
1,048,576-token context, then tune it with short concurrency probes before the full benchmark.

## Weights

- `Aleph-Alpha/Kolibri-1` at `abfe85f673fd79a3e1cdb98289e7a1c635345ee1` (Apache-2.0, not
  gated). It is the official release and already FP8: `float8_e4m3fn` weights in 128x128
  blocks, dynamic activations; embeddings, LM head, norms and the MoE router stay BF16
  (`quant_method: fp8`). 32 shards, 73.4 GiB. A BF16 build exists
  (`Aleph-Alpha/Kolibri-1-BF16`, 145.5 GiB) and does not fit one H200 with a KV cache.
  GitHub has no official FP8 build beyond this one; only community recipes (DGX Spark, MLX).
- 78.1B parameters, 3.46B active; 50 MoE layers with 384 experts (6 routed, 1 shared);
  48 query and 4 KV heads, head dim 128; sliding window 513 in 40 layers and full attention
  in every fifth layer (10 layers).
- Aleph Alpha names 1x H200 as the minimum and recommends at most 262,144 tokens for serving
  efficiency; it validated quality up to 1,048,576 tokens.

## Engine

- vLLM `v0.33.0` (the image already pinned for Muse Glimmer). It serves `Kolibri1ForCausalLM`
  through the Transformers modeling backend (vllm-project/vllm#60827, in v0.33.0); the image
  holds transformers 5.20.2, which has `kolibri1` (checked in the image on the node).
- Reasoning and tool calls: `--reasoning-parser hf --tool-call-parser hf`, which read the
  `response_template` Aleph Alpha added to `tokenizer_config.json` on 2026-10-20 (thinking in
  `<think>`, tool calls as JSON in `<tool_call>`).
- 1M context: `--max-model-len 1048576 --hf-overrides '{"max_position_embeddings": 1048576}'`,
  as the model card says. Positions are encoded only in the sliding-window layers, so no
  position scaling is needed.
- `--kv-cache-dtype fp8`, as Aleph Alpha evaluated it. With FP8, a token's KV cache in the 10
  full-attention layers is 10 x 2 x 4 x 128 = 10,240 bytes; the 40 sliding-window layers hold
  only their last 513 tokens per request.
- Sampling from the checkpoint (`--generation-config auto`): temperature 1.0, top_p 0.97,
  top_k 128.

## Initial resource profile

Hardware booking 128 GiB, engine booking 120 GiB (`--gpu-memory-utilization 0.8584`): the
weights take about 73 GiB, which leaves about 40 GiB for the KV cache. Starting values for
the tuning.

## Logo

`assets/logo.svg` is Aleph Alpha's mark from the favicon of aleph-alpha.com, reduced to its
five paths.
