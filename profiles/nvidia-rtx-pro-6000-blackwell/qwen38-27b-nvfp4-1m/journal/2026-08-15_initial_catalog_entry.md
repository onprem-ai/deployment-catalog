# 2026-08-15 - Initial 1M Catalog Entry

**Goal:** Add a separate 1M-context sibling for the validated
`unsloth/Qwen3.8-27B-NVFP4` card, following the established Qwen3.6 catalog
layout while using the Qwen3.8-specific vLLM recipe.

## Why This Is a Separate Card

The 1M variant changes the model's effective context capability and position
configuration, not only its memory allocation. A separate card lets the
catalog, gateway context limit, deployment annotation, and UI all report
1,010,000 tokens accurately; putting this in the 262K card's memory profiles
would leave its card-level `capabilities.context_length` ambiguous.

The sibling reuses exactly the same pinned Hugging Face checkpoint and vLLM
image as `qwen38-27b-nvfp4`, so selecting it does not duplicate model or image
storage.

## Configuration

The official vLLM Qwen3.8-27B recipe documents this exact extension syntax:

```text
--max-model-len 1010000
--hf-overrides '{"text_config": {"max_position_embeddings": 1010000}}'
```

The override must be nested under `text_config` for the 27B model. The card
retains the validated sibling settings for FP8 KV cache, two-token MTP,
reasoning parsing, tool parsing, and the bundled Unsloth chat template.

The initial 84 GiB allocation comes from the sibling's live RTX PRO 6000
bring-up: at that allocation vLLM reported a 1,700,147-token FP8 KV pool,
which exceeds the 1,010,000-token target. It is an explicit validation
allocation, not a completed minimum-VRAM bisection; the card remains
experimental until the 1M startup and request path are tested and the true
minimum is measured.

## Sources

- https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4

## Outcome

Created the experimental 1M sibling card. It still requires deployment,
long-context inference validation, capability tests, and VRAM bisection before
promotion to stable or describing 84 GiB as a minimum working floor.
