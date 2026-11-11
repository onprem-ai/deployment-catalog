# 2026-11-11 - Initial Arete 27B FP8 Catalog Entry for H200

**Goal:** One Arete-27B card for one H200, then a short tuning for the best performance.

## Weights

- `NousResearch/Arete-27B` at `108f0afbdcfaa1ceeda4a635004304dfff6d4da2` (Apache-2.0, not
  gated). Supervised fine-tuning of `Qwen/Qwen3.8-27B` by Nous Research; no change to the
  architecture (`Qwen3_5ForConditionalGeneration`: 64 layers, 48 linear-attention and 16
  full-attention layers, 4 KV heads of 256, a vision encoder, 262,144 tokens). 51.0 GiB of
  BF16 weights. Nous froze the vision encoder and trained on text only.
- No FP8 build from Nous Research or another trusted publisher exists (the Hugging Face
  builds are GGUF, MLX, NVFP4 and EXL3 from individuals). The card loads the official BF16
  weights and vLLM quantizes the linear layers to FP8 at load (`--quantization
  fp8_per_block`), as on the Ripple card.

## Speculative decoding

- Arete has **no MTP head**: Qwen3.8-27B ships one (`mtp.*`, 15 tensors), but Nous switched
  it off in training ("Disabled during training: Multi-token-prediction (MTP) head") and the
  checkpoint has no `mtp.*` tensors (`mtp_num_hidden_layers: 0`). Qwen's MTP head belongs to
  Qwen's weights and is not used.
- Drafter: `z-lab/Qwen3.8-27B-DFlash2` at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
  (Apache-2.0, 3.6 GiB; the z-lab mirror of `incoai/Qwen3.8-27B-DFlash2`, made by the DFlash
  authors), a DFlash2 drafter for the base model with 7 draft tokens in its own evaluation on
  one H200. vLLM v0.33.0 holds DFlash2 (vllm-project/vllm#52816). It was trained on
  Qwen3.8-27B's hidden states, not Arete's; its acceptance on Arete is measured in the tuning.

## Starting settings

From the official catalog's `nvidia-rtx-pro-6000-blackwell/qwen38-27b-nvfp4` card: FP8 KV cache,
128 sequences, 4,096 batched tokens, `qwen3` reasoning parser, `qwen3_coder` tool parser,
prefix caching. vLLM v0.33.0 (the Qwen3.8 card pins v0.27.1). Context 262,144 tokens, the
native length. Engine booking 112 GiB, hardware booking 120 GiB.

## Logo

`assets/logo.svg` holds Nous Research's avatar from Hugging Face.
