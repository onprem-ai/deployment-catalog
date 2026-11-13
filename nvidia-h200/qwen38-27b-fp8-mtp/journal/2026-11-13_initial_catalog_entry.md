# 2026-11-13 - Port of Qwen3.8 27B to One H200

**Goal:** The official catalog's RTX PRO 6000 Blackwell card (`nvidia-rtx-pro-6000-blackwell/qwen38-27b-nvfp4`)
as a card for one H200, with the settings of the catalog's Arete 27B H200 card (`nvidia-h200/arete-27b-fp8`,
a fine-tune of this same architecture), then a short tuning and the full benchmark.

## From the Blackwell card

Unsloth's NVFP4 build through an onprem-ai mirror, vLLM v0.27.1, MTP with 2 draft tokens, 128 sequences,
4,096 batched tokens, FP8 KV cache, prefix caching, the qwen3 / qwen3_coder parsers and the bundled
chat template. NVFP4 suits Blackwell, not Hopper.

## Choices for the H200

- **Weights:** Qwen's own block-scaled FP8 checkpoint `Qwen/Qwen3.8-27B-FP8` at
  `c3a462cc3d38cf351f3b26cbd763469eeb74ef81` (Apache-2.0, 28.8 GiB), which keeps the MTP head
  (`mtp.safetensors`, `mtp_num_hidden_layers: 1`). vLLM's recipe lists this checkpoint for H200.
- **Speculative decoding:** the built-in MTP head, 3 draft tokens (vLLM's recipe; measured acceptance
  on short prompts 84.8 % for FP8). Arete has no MTP head and used the DFlash2 drafter instead.
- **From the Arete card:** FP8 KV cache, 128 sequences, 16,384 batched tokens, prefix caching, 128 GiB
  engine booking (Arete's tuning: the linear-attention state fills the KV cache before compute is
  the limit), 262,144 tokens of context, the qwen3 / qwen3_coder parsers, vLLM v0.33.0 (which carries
  the gated-delta-net speculative fixes the recipe asks for, vllm-project/vllm#51812 and #51674).
- **Inputs:** images (4 per request) and one video, as the model supports them.
- **Bookings (starting values):** engine 128 GiB, hardware 136 GiB (Arete peaked at 132.9 GiB with a
  4 GiB DFlash2 drafter); RAM 8 GiB reserved, 16 GiB limit; CPU 2 reserved, 4 limit.
