> Moved here from the BF16 card (`nvidia-h200/muse-glimmer-30b-bf16`), which was removed on 2026-11-11: this card replaced it. It records the BF16 bring-up that led to FP8 and DFlash.

# 2026-11-10 - Initial Muse Glimmer BF16 Catalog Entry for H200

**Goal:** Add an experimental Muse Glimmer 30B card for one NVIDIA H200 (141 GB) that serves
the reference BF16 checkpoint on vLLM, with image input, reasoning and tool calls.

## Research

- Target: Meta's own `meta-models/Muse-Glimmer-30B` at revision
  `cec94fcfd6512f2cfe708138c32533b346ea4f69` (Apache-2.0, not gated). Two weight shards,
  55.5 GiB in total. It is the only published checkpoint that keeps the perception encoder,
  so it is the only one that takes images.
- Quantized builds were not chosen: the GGUF builds are made for llama.cpp on 24/32 GB cards
  and SGLang serves them text-only and unoptimized; the NVFP4 builds need Blackwell, and the
  H200 is Hopper. The H200 holds BF16 with room for a large KV cache.
- Engine: vLLM. Meta's deploy guide names it for production serving on NVIDIA GPUs, and
  the vLLM recipe (https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B) lists H200. The
  image is the release `vllm/vllm-openai:v0.33.0`, pinned to its index digest
  `sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e`. Its model
  registry has `MuseGlimmerForConditionalGeneration` and the DFlash drafter, and it ships the
  `muse_glimmer` reasoning and tool-call parsers. The image was published on 2026-11-02, more
  than 7 days before it can reach a stable card.

## Launch flags

- `--tool-call-parser muse_glimmer --reasoning-parser muse_glimmer
  --enable-auto-tool-choice`: the model writes ATEM tool calls and channel-scoped reasoning;
  both parsers key off that framing.
- `--generation-config auto`: the checkpoint's sampling (temperature 1.0, top_p 0.95,
  top_k 64) and stop tokens `[200001, 200008]`.
- `--max-model-len 131072`, the trained context.
- `--enable-prefix-caching`, stated rather than left to the engine default.
- `--max-num-seqs 128 --max-num-batched-tokens 8192`: starting values, to be tuned with
  the concurrency measurements.

## Initial resource profile

- Hardware booking 120 GiB, engine booking 112 GiB (`--gpu-memory-utilization 0.8011` of
  the 139.8 GiB the GPU shows). These are bring-up values, not measured ones.

## Outcome

Created the card with the pinned checkpoint and image. Download, deployment, capability
checks, concurrency tuning, memory measurement and the benchmark remain to do.
