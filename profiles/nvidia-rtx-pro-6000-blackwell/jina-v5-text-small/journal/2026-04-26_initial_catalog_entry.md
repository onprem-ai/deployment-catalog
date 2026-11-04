# 2026-04-26 — Initial catalog entry

**Goal:** Add Jina Embeddings v5-text-small to the catalog as a text embedding option.

## Research

- Model: `jinaai/jina-embeddings-v5-text-small` (677M params, distilled from Qwen3-Embedding-4B)
- MTEB v2: 71.7 avg (English), MMTEB: 67.7 — best quality-to-size under 1B params
- Architecture: custom `JinaEmbeddingsV5Model`, last-token pooling, 1024-dim output
- Matryoshka dims: 32, 64, 128, 256, 512, 768, 1024
- Context: 32K tokens (RoPE, theta=3.5M)
- 4 LoRA adapters: retrieval, text-matching, clustering, classification
- License: **CC BY-NC 4.0** (non-commercial — commercial use requires Jina AI license)

## Serving decision

Using the pre-merged retrieval variant `jinaai/jina-embeddings-v5-text-small-retrieval` for vLLM.
The base model uses dynamic LoRA adapters which vLLM cannot switch at runtime.
Pre-merged variant has adapter baked into weights — single model.safetensors (~1.14 GB).

vLLM >= 0.15.1 required. Needs `--trust-remote-code` for custom architecture.
Also available as GGUF (F16, Q8_0, Q4_K_M) for llama.cpp.

## VRAM estimate

- Weights in float16: ~1.14 GB
- Estimated total with KV cache + activations: ~2 GB
- Trivially small on RTX 6000 BW (48 GB)

## Storage

- Retrieval variant: ~1.14 GB (model.safetensors)
- Full repo with all adapters: ~4.59 GB
- Using retrieval variant only: ~1.2 GB on disk

## Outcome

Catalog entry created as experimental. Needs deployment testing to verify:
1. vLLM `--runner pooling` with trust_remote_code on NGC container
2. Actual VRAM usage measurement
3. Embedding quality validation against Nomic Embed v2

## Optimization pass (same day)

Updated inference.yaml based on research of vLLM docs, Jina HuggingFace model card,
and Blackwell SM120 optimization guides:

1. **`--task embed` -> `--runner pooling`**: `--task embed` was deprecated in vLLM 0.10.x.
   NGC 26.02 ships vLLM 0.15.1, so use `--runner pooling` (consistent with qwen3-embedding-8b,
   qwen3-vl-embedding-2b catalog entries).

2. **Added `--pooler-config`**: Explicit `{"seq_pooling_type": "LAST", "normalize": true}` per
   Jina's official HuggingFace recommendation. `--override-pooler-config` was removed in vLLM 0.12.0;
   replaced with `--pooler-config`.

3. **Added `--enable-prefix-caching`**: Model is decoder-only with last-token pooling.
   vLLM V1 prefix caching has <1% overhead at 0% hit rate. Retrieval queries share "Query:"
   and "Document:" prefixes, enabling KV cache reuse.

4. **Added memory profiles**: Two profiles (32K recommended, 8K low-VRAM) with explicit
   `--max-model-len` and `--enable-prefix-caching` in params.

5. **Kept `--dtype float16`**: Jina's official vLLM code sample uses `dtype="float16"`.
   On Blackwell SM120, FP16 and BF16 have identical Tensor Core throughput. float16 matches
   Jina's tested configuration.

6. **Not added**: `--max-num-seqs` / `--max-num-batched-tokens` left at vLLM defaults (1024/dynamic).
   For a 677M model on 48 GiB VRAM, defaults are fine. Snowflake's embedding optimizations
   (pretokenization, multi-replica GPU packing) are application-level and don't apply to
   single-model vLLM serve flags.

Sources:
- https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval
- https://huggingface.co/jinaai/jina-embeddings-v5-text-small/discussions/13
- https://docs.vllm.ai/en/stable/models/pooling_models/
- https://docs.vllm.ai/en/stable/models/pooling_models/embed/
- https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-02.html
