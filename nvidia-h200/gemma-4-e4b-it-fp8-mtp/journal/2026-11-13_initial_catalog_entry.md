# 2026-11-13 - Port of Gemma 4 E4B IT to One H200

**Goal:** The official catalog's RTX PRO 6000 Blackwell card (`nvidia-rtx-pro-6000-blackwell/gemma-4-e4b-it-nvfp4`)
as a card for one H200, then a short tuning and the full benchmark.

## From the Blackwell card

The Blackwell card runs `cosmicproc/gemma-4-E4B-it-NVFP4` (an individual's NVFP4 conversion, through an
onprem-ai mirror without a revision) on vLLM's `gemma4-cu130` image, with a modelopt quantization flag
and its own chat template, 128 sequences and 4,096 batched tokens. NVFP4 suits Blackwell, not Hopper.

## Choices for the H200

- **Weights:** Google's own `google/gemma-4-E4B-it` at `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`
  (Apache-2.0, BF16, 14.9 GiB with the vision and audio encoders). No trusted FP8 build exists: Red Hat's
  `RedHatAI/gemma-4-E4B-it` is BF16 too. vLLM quantizes the linear layers to FP8 at load
  (`--quantization fp8_per_block`), as on the catalog's Ripple and Arete cards.
- **Drafter:** Google's MTP drafter `google/gemma-4-E4B-it-assistant` at
  `8d0031ea8c2109e2b1e86bb9368a4539b537f80a` (0.18 GiB), as on the 26B and 31B H200 cards; vLLM 0.33
  serves `Gemma4AssistantForCausalLM` natively.
- **Settings:** the 26B-A4B H200 card's tuned set as the start (FP8 KV cache, 256 sequences, 16,384
  batched tokens, prefix caching, the gemma4 reasoning and tool parsers, 1 draft token), on the
  catalog's vLLM v0.33.0 image; the model's bundled chat template carries tool calls.
- **Bookings (starting values):** engine 20 GiB, hardware 24 GiB (the Blackwell card booked 22 GiB of
  hardware for NVFP4); RAM 8 GiB reserved, 12 GiB limit; CPU 2 reserved, 4 limit.
