# 2026-11-10 - Initial Ripple 65B Preview FP8 Catalog Entry for H200

**Goal:** A Ripple-65B Preview card for one H200, then 3 hours of tuning and the full
benchmark.

## Weights

- Target: `ricursiveAI/Ripple-65B-Preview` at `b172323ffd63e62de9ce1ba3830dfffe493c7671`
  (Apache-2.0, not gated). Dense, 80 layers (four 1,024-token sliding-window layers to one
  NoPE full-attention layer: 16 full layers), d_model 5,120, 80 query and 8 KV heads, gated
  attention, value residuals. Text only. 129.6 GB (120.7 GiB) of BF16 weights.
- The official checkpoints are BF16 and NVFP4. NVFP4 needs Blackwell; BF16 leaves almost no
  KV cache on one H200 (vLLM's recipe asks 156 GB for it). No FP8 build from Ricursive or
  another trusted publisher exists: the FP8 repositories on Hugging Face are community builds
  with 57 to 70 downloads (`primitive-ai/Ripple-65B-Preview-FP8`,
  `notSnix/Ripple-65B-Preview-MXFP8-GPTQ`; `TheHouseOfTheDude/...-INT8-W8A8` targets Ampere).
- So the card loads the official BF16 weights and vLLM quantizes the linear layers to FP8 at
  load (`--quantization fp8_per_block`: FP8 weights in 128x128 blocks, activations in
  dynamic 1x128 blocks, the scheme of the official FP8 checkpoints of Kolibri and Muse
  Glimmer). Embeddings, LM head and norms stay BF16. This needs the full BF16 download
  (host RAM booking 64 GiB) and no third-party weights. Answer quality must be checked
  against the BF16 numbers of the model card.
- Drafter: `ricursiveAI/Ripple-65B-Preview-DFlash2` at `35609b3be3dd08a822b3ba1257c1ea5dd6f87632`
  (Apache-2.0, the same license text), 7.5 GiB, five layers, NoPE in its last layer.

## Engine

- vLLM supports Ripple natively from vllm-project/vllm#65745 (merged 2026-11-05): the model,
  `--reasoning-parser ripple`, `--tool-call-parser ripple`, and the DFlash2 drafter with its NoPE
  layer (mean acceptance on GSM8K 7.34 with 15 tokens, per the PR). The release v0.33.0
  predates it. The image is the nightly `nightly-cf5cea2da37d3dfed27d2e0aa3041c3e553c2d57`
  (2026-11-08), pinned by digest; it holds #65745 and the later spec-decode fix
  (b936e1427f). It is 2 days old: younger than the 7-day rule for a stable card, so the card
  stays experimental until a release or an older image holds the support.
- Ricursive's own server is SGLang 0.5.19 with an out-of-tree plugin on two B200: not used.
- Context: the model supports 2M positions; the card starts at 262,144 (the KV cache is the
  limit on one GPU, see the tuning).
- Sampling: the checkpoint ships no `generation_config.json`; vLLM's defaults apply.

## Initial resource profile

Hardware booking 132 GiB, engine booking 120 GiB (`--gpu-memory-utilization 0.8584`), as for
Kolibri; weights in FP8 should take about 62 GiB plus 7.5 GiB for the drafter.

## Logo

`assets/logo.svg` holds Ricursive's mark, cropped from the PNG logo on the model card (the
company publishes no SVG).
