# 2026-08-15 - Initial Catalog Entry

**Goal:** Add an experimental RTX PRO 6000 Blackwell catalog card for
`unsloth/Qwen3.8-27B-NVFP4`, following the established Qwen3.6 cards while
using current model-specific documentation.

## Research

The upstream Hugging Face repository was inspected at commit
`16b6615af3548b88e2d8e382457bc705b00479cf`. Its file tree contains one main
weight file (`model.safetensors`, 22,568,192,096 bytes), a separate MTP head
(`model_mtp.safetensors`, 849,400,392 bytes), multimodal processor files, the
chat template, and tokenizer files. The complete selected download is
23,444,503,536 bytes (21.83 GiB).

The checkpoint config identifies `Qwen3_5ForConditionalGeneration`, 64 dense
language layers, 48 linear-attention layers, 16 full-attention layers, native
262,144-token context, a vision encoder, compressed-tensors mixed precision,
and an MTP head. It was written by Transformers 5.14.1.

Unsloth's Qwen3.8 guide requires vLLM 0.25.0 or newer and documents:

```text
vllm serve unsloth/Qwen3.8-27B-NVFP4
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

The guide also documents a 24GB VRAM target, FP8 KV-cache calibration, and
133.7 decode tokens/s at batch size 1 on one B200. That performance result is
not copied into the card because it was not measured on the target RTX PRO
6000 hardware.

The official vLLM Qwen3.8-27B recipe confirms the `qwen3` reasoning parser,
`qwen3_coder` tool parser, FP8 KV cache, and MTP. It states that text serving
has been verified; multimodal serving still needs validation for this exact
checkpoint and runtime combination.

## Decisions

- Marked the card `experimental`; no deployment or inference validation has
  been completed on this node.
- Pinned the Hugging Face revision and vLLM v0.27.1 image manifest.
- Used the model's bundled chat template to preserve Unsloth's developer-role
  and nested tool-call improvements.
- Enabled two-token MTP exactly as documented by Unsloth.
- Added only the published 24 GiB memory floor. It is not presented as a
  context-specific measured floor; 32K, 128K, and 262K profiles require
  empirical bisection on the RTX PRO 6000 before promotion to stable.
- Omitted performance data until the internal benchmark API measures this
  deployment on target hardware.

## Sources

- https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4
- https://huggingface.co/api/models/unsloth/Qwen3.8-27B-NVFP4
- https://unsloth.ai/docs/models/qwen3.8.md
- https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- https://docs.vllm.ai/en/v0.27.1/

## Outcome

Created the experimental catalog card and mandatory journal entry. Remaining
work is to download through the model-catalog API, deploy through the API,
validate text, vision, reasoning, and tool calling, then measure minimum VRAM
and performance through the internal APIs before changing maturity.
