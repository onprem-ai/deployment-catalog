# Session: Initial Deployment

**Date**: 2026-04-05
**Model**: cosmicproc/gemma-4-E4B-it-NVFP4
**Hardware**: 1x NVIDIA RTX 6000 Blackwell (96GB)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML frameworks.
> Information 2+ weeks old may be outdated. Verify against current releases.

## Model Info

- **Model**: cosmicproc/gemma-4-E4B-it-NVFP4
- **Source**: https://huggingface.co/cosmicproc/gemma-4-E4B-it-NVFP4
- **Hardware**: 1x NVIDIA RTX 6000 Blackwell (96GB)
- **Effective Params**: 4.5B (8B total with embeddings)
- **Context**: 128K native

## Final Working Configuration

**Status: WORKING**

```bash
vllm serve /models/gemma-4-e4b-it-nvfp4 \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --chat-template /models/gemma-4-e4b-it-nvfp4/chat_template.jinja \
  --enable-prefix-caching
```

**Key settings**:
- `--quantization modelopt` - NVIDIA's modelopt quantization format
- `--tensor-parallel-size 1` - Single GPU (small model fits easily)
- `--chat-template` - Required for Gemma 4 (not embedded in tokenizer)
- `--enable-prefix-caching` - 22% faster for repeated prefixes (tested on 31B)
- Image: `vllm/vllm-openai:gemma4-cu130` - Official Gemma 4 image

---

## What We Tried (Chronological)

### Attempt 1: Initial deployment
**Status**: SUCCESS

Based on working configuration from Gemma 4 31B IT NVFP4. Same architecture, same chat template format.

**Results** (2026-04-05):
- Model loads successfully with `vllm/vllm-openai:gemma4-cu130`
- Uses repo's native `chat_template.jinja` (no custom template needed)
- VRAM: 9.9 GiB model + KV cache
- **Decode TPS: ~153** (3.8x faster than 31B!)
- Prefix caching working

---

## Known Issues

1. **Chat template not embedded**: Same as 31B - Gemma 4 tokenizer doesn't include chat_template in tokenizer_config.json. Must provide via `--chat-template` argument.

2. **vLLM version requirement**: Gemma 4 architecture requires `vllm/vllm-openai:gemma4-cu130` or newer image with transformers 5.5.0+.

## Constraints

- **vLLM image**: `vllm/vllm-openai:gemma4-cu130` (or newer with Gemma 4 support)
- **Quantization**: `--quantization modelopt` (NOT compressed-tensors)
- **Chat template**: Must provide via `--chat-template` argument
- **Context**: 128K native

---

## Key Learnings from Gemma 4 31B

### Chat Template Format
Gemma 4 uses different control tokens from Gemma 2/3:

| Version | Turn Start | Turn End | Assistant Role |
|---------|------------|----------|----------------|
| Gemma 2/3 | `<start_of_turn>` | `<end_of_turn>` | `model` |
| Gemma 4 | `<\|turn>` | `<turn\|>` | `model` |

### Thinking Mode
- Enable: Add `<\|think\|>` to system prompt
- Disable: Add empty `<\|channel>thought\n<channel\|>` before model response

Our chat template disables thinking by default.

---

## Performance Optimization

### Tested on 31B (applicable here)

| Method | Result | Recommendation |
|--------|--------|----------------|
| N-gram speculative | 0% acceptance | Not useful for open-ended generation |
| EAGLE/P-EAGLE | Not available | Gemma 4 draft model not yet ready |
| Prefix caching | 22% faster | Enabled in this config |

---

## References

- [Gemma 4 vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- [Gemma 4 E4B HuggingFace](https://huggingface.co/google/gemma-4-E4B-it)
- [NVFP4 Quantization Blog](https://developer.nvidia.com/blog/bringing-ai-closer-to-the-edge-and-on-device-with-gemma-4/)
- [cosmicproc/gemma-4-E4B-it-NVFP4](https://huggingface.co/cosmicproc/gemma-4-E4B-it-NVFP4)
