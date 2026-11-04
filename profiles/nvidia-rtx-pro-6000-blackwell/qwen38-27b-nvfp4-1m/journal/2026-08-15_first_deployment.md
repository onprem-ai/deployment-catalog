# 2026-08-15 - First 1M Deployment

**Goal:** Replace the running native-context Qwen3.8 deployment with the new
1,010,000-token sibling and verify that vLLM accepts the extended context.

## Deployment

The running `rtx6k--qwen38-27b-nvfp4` InferenceService was removed through
the control-plane API. The 1M sibling reused the already-downloaded checkpoint
and vLLM image, then deployed through the model-catalog API on GPU 0 with its
84 GiB validation allocation.

The rendered deployment used the expected model-specific settings:

```text
--max-model-len 1010000
--hf-overrides '{"text_config": {"max_position_embeddings": 1010000}}'
--kv-cache-dtype fp8
--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

The InferenceService became Ready without a restart. Its annotations correctly
record `ai-node.io/max-context-tokens: 1010000` and associate it with the 1M
catalog config.

## Runtime Measurements

vLLM 0.27.1 confirmed `max_seq_len=1010000` and reported:

```text
Weights plus non-torch memory: 23.40 GiB
Peak activation memory: 2.33 GiB
CUDA graph memory: 0.71 GiB
Available FP8 KV cache: 57.84 GiB
GPU KV cache capacity: 1,755,288 tokens
Maximum 1,010,000-token concurrency: 1.74x
```

The allocation therefore provides enough KV capacity for one full 1,010,000
-token request. It does not establish that 84 GiB is the minimum working
allocation; VRAM bisection remains required.

vLLM logged that the checkpoint was treated as multimodal but no multimodal
processor was registered, so this deployment entered text-only mode. The
native-context deployment previously passed vision tests, making this warning
a behavior difference that must be investigated before claiming vision support
for the 1M variant.

## Smoke Test

A chat request through the internal inference-service API returned
`QWEN38_1M_OK` with `finish_reason: stop`, confirming that text generation and
the reasoning parser work after extending the context.

## Outcome

The 1M sibling is deployed, Ready, and left running. vLLM recognizes the full
1,010,000-token context and allocates sufficient FP8 KV capacity. Remaining
work includes a genuine long-context request, capability tests, investigation
of the text-only multimodal warning, performance measurement, and minimum-VRAM
bisection.
