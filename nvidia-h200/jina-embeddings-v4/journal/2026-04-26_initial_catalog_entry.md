# 2026-04-26 — Initial catalog entry

**Goal:** Add Jina Embeddings v4 as a multimodal embedding model (text + images + PDFs).

## Research

- Model: `jinaai/jina-embeddings-v4` (3.8B params, built on Qwen2.5-VL-3B-Instruct)
- Multimodal: text, images, and visual documents in unified vector space
- Dense mode: 2048-dim (Matryoshka: 128, 256, 512, 1024, 2048)
- Multi-vector mode: 128-dim per token (ColBERT-style late interaction)
- Context: 32K tokens
- 3 task-specific LoRA adapters: retrieval, text-matching, code
- License: **Qwen Research License** (non-commercial — requires Alibaba Cloud license for commercial use)
- ArXiv: 2506.18902

## Serving decision

Using pre-merged vLLM retrieval variant: `jinaai/jina-embeddings-v4-vllm-retrieval`
Base model uses dynamic LoRA adapters — vLLM needs pre-merged weights.
Requires `--trust-remote-code` (custom architecture: JinaEmbeddingsV4Model).

Dependencies in custom code: transformers>=4.52.0, torch>=2.6.0, peft>=0.15.2, torchvision, pillow.
NGC vLLM container should have these.

## VRAM estimate

- Weights: ~7.5 GB (2 safetensors shards)
- Estimated total: ~10-12 GB (weights + vision encoder overhead + KV cache)
- Comfortable on RTX 6000 BW

## Storage

- Pre-merged retrieval variant: ~7.5 GB on disk

## Outcome

Catalog entry created as experimental. Key risks:
1. Custom model code (modeling_jina_embeddings_v4.py + qwen2_5_vl.py) — may need specific dependency versions
2. Vision input through vLLM embedding endpoint needs testing
3. Restrictive license (Qwen Research) — flag for user awareness
