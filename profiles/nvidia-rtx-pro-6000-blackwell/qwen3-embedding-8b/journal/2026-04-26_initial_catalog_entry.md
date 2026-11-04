# 2026-04-26 — Initial catalog entry

**Goal:** Add Qwen3-Embedding-8B to the catalog as a high-quality text embedding model.

## Research

- Model: `Qwen/Qwen3-Embedding-8B` (8B params, based on Qwen3-8B-Base)
- **#1 on Multilingual MTEB** (70.58), strong English performance
- Architecture: Qwen3ForCausalLM (decoder-only fine-tuned for embeddings)
- 36 layers, 32 attn heads, 8 KV heads (GQA), head_dim 128
- Embedding dim: 4096 (Matryoshka: 32–4096)
- Context: 32K tokens (max_position_embeddings: 40960, documented as 32K)
- Last-token pooling, instruction-aware queries
- License: Apache-2.0

## Instruction-aware queries

Queries should use format:
```
Instruct: <task description>
Query: <actual query>
```
Documents do NOT need instructions. Without instructions, retrieval drops ~1-5%.

## Serving decision

vLLM native support (>= 0.8.5): `vllm serve Qwen/Qwen3-Embedding-8B --task embed`
Standard Qwen3 architecture, no trust_remote_code needed.
Also available as GGUF via `Qwen/Qwen3-Embedding-8B-GGUF`.

## VRAM estimate

- Weights in BF16: ~14.1 GB (4 safetensors shards)
- Estimated total: ~18 GB with overhead
- Fits comfortably on RTX 6000 BW (48 GB) with plenty of headroom

## Storage

- 4 safetensors shards + configs: ~14.1 GB on disk

## Outcome

Catalog entry created as experimental. Most straightforward deployment of the batch — standard Qwen3 on vLLM. Needs:
1. Deployment test and VRAM measurement
2. Benchmark embedding quality vs Nomic Embed v2
3. Verify instruction-aware prompting works through AI Gateway
