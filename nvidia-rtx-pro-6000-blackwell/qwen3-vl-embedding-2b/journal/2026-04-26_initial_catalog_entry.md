# 2026-04-26 — Initial catalog entry

**Goal:** Add Qwen3-VL-Embedding-2B as a small multimodal embedding model (text + images + video).

## Research

- Model: `Qwen/Qwen3-VL-Embedding-2B` (2.13B params, fine-tuned from Qwen3-VL-2B-Instruct)
- Multimodal: text, images, screenshots, video in unified embedding space
- Embedding dim: 2048 (Matryoshka: 64–2048)
- Context: 32K tokens
- Image: min 4096 to max 1,843,200 pixels
- Video: max 64 frames, 1 fps default
- Last-token pooling
- 2.1M downloads/month on HuggingFace — trending hard
- License: Apache-2.0
- ArXiv: 2601.04720

## Serving decision

vLLM confirmed working: `LLM(model=..., runner="pooling", dtype="bfloat16", trust_remote_code=True)`
Also works with sentence-transformers (native config shipped) and SGLang.
Needs `--trust-remote-code` for custom embedder class.

May need `qwen-vl-utils>=0.0.14` — verify NGC container has this.

## Input format

Multimodal inputs use dict format:
```python
{"text": "description", "image": "path_or_url"}
{"video": "/path/to/video.mp4"}
{"text": "query", "instruction": "Retrieve relevant images"}
```

## VRAM estimate

- Weights: 4.26 GB (single safetensors, BF16)
- Estimated total: ~5-6 GB
- Smallest practical multimodal embedding model
- FP8 variant exists (~2.5-3 GB): `alexliap/Qwen3-VL-Embedding-2B-FP8-DYNAMIC`

## Storage

- Single model.safetensors + configs: ~4.3 GB on disk

## Outcome

Catalog entry created as experimental. Best multimodal-to-VRAM ratio in the catalog. Needs:
1. Verify vLLM pooling runner handles vision inputs correctly
2. Check qwen-vl-utils availability in NGC container
3. Test image + text embedding in same request
4. Measure actual VRAM usage
