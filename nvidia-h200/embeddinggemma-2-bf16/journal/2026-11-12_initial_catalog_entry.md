# 2026-11-12 - Initial EmbeddingGemma 2 Catalog Entry for H200

**Goal:** One card for a recent, popular open embedding model on one H200, for the playground's
embedding mode.

## Choice

Candidates on 2026-11-12 (MTEB Multilingual v2 from the MTEB results, licenses from the cards):
Microsoft's harrier-oss-v1-27b (74.27, MIT, 27B), Qwen3-Embedding-8B (70.58, Apache-2.0, June
2025), harrier-oss-v1-0.6b (69.01, MIT), jina-embeddings-v5 (CC BY-NC, not for commercial use),
EmbeddingGemma 2 (60.52, Apache-2.0). **EmbeddingGemma 2** is the most recent and the most popular
of them on Hugging Face (released 2026-10-06, 3,342 likes, 503,609 downloads), from Google
DeepMind, commercially usable, and vLLM v0.33.0 serves its architecture (`EmbeddingGemma2Model`)
natively. It is small (740M; 270M for text), so it shares an H200 with a large model.

## Weights

`google/embeddinggemma-2` at `f9972ac22e696a105c482202646979e33e4b5a1e` (Apache-2.0, not gated),
BF16, 1.42 GiB. Text, images, audio and video map into one 768-dimensional space; 100+ languages;
8,192 tokens; Matryoshka truncation to 512, 256 and 128 dimensions. The repository bundles no
license file; the card ships the Apache License 2.0 text from apache.org.

## Settings

- `--runner pooling`: vLLM's model class sets mean pooling and applies the checkpoint's 512 -> 768
  projection; vLLM normalizes the embedding.
- `--hf-overrides '{"matryoshka_dimensions":[768,512,256,128]}'`: the checkpoint's `config.json`
  does not declare its Matryoshka dimensions, so vLLM would refuse the `dimensions` request field;
  the four values are the model card's.
- Images on (one per request), audio and video off: the API gateway's `/v1/embeddings` takes text
  and, through vLLM's chat-style embedding input, images.
- Prefix caching off; 256 sequences, 16,384 batched tokens, BF16.
- Bookings: engine 8 GiB (`--gpu-memory-utilization 0.0572`), hardware 12 GiB, starting values.

## Task prefixes

The model is trained with short prefixes, which the client adds: `task: search result | query: `
for search queries, `title: none | text: ` for documents (or `title: {title} | text: `), and
`task: sentence similarity | query: `, `task: classification | query: ` and others for symmetric
tasks (`config_sentence_transformers.json` lists all of them).

## First deployment

Deployed from this catalog on one H200 (inference node `nebius-e00pn5kkfnayh8wn6j`), next to a
120.7 GiB Nemotron 3.5 Super VL replica on the same GPU. The deploy warned that only 8.8 GiB
were free against the 12 GiB booking; the replica started and used 1.9 GiB, and the Nemotron
replica kept running.

Through the API gateway (`POST /v1/embeddings`, with Google's task prefixes):

- One query and three documents in one request, 0.15 s, 81 input tokens. Vectors of 768 values
  with length 1.0. Cosine similarity to "What causes the northern lights?": the English answer
  0.87, the German answer about the northern lights 0.72, "Zurich is the largest city in
  Switzerland." 0.54.
- `dimensions` 512, 256 and 128 return vectors of that length. vLLM accepts at most 512: a
  request for 768 is refused ("only supports dimensions in range [1, 512]"), because vLLM checks
  against the text backbone's 512 values, not the 768 the projection returns. Leaving
  `dimensions` out gives the full 768.


## Measured use under load (2026-11-12)

Alone on one H200 (inference node `nebius-e00pn5kkfnayh8wn6j`), with a load probe on the node sending
straight to the engine: unique texts (no cache hits), 10 s of warm-up, then a 25 s window per level;
the GPU's memory read with `nvidia-smi` every 0.5 s. Short: 32 texts of about 80 tokens per request;
medium: 8 texts of about 500 tokens; longest: one text near the card's context length.

GPU memory: 1,941 MiB right after start, **3,595 MiB at the peak** (3.51 GiB), within the
12 GiB hardware booking. The memory does not grow with load: the engine reserves its KV cache at
start.

| Shape | Requests in flight | Inputs/s | Tokens/s | Latency p50 | Errors |
|---|---|---|---|---|---|
| short (32 x ~80 tokens) | 1 | 992.0 | 67,716 | 0.032 s | 0 |
| short (32 x ~80 tokens) | 16 | 2,944.9 | 201,029 | 0.173 s | 0 |
| short (32 x ~80 tokens) | 64 | 2,810.9 | 191,884 | 0.728 s | 0 |
| medium (8 x ~505 tokens) | 1 | 251.8 | 97,777 | 0.033 s | 0 |
| medium (8 x ~505 tokens) | 16 | 1,378.2 | 535,113 | 0.09 s | 0 |
| medium (8 x ~505 tokens) | 64 | 1,380.1 | 535,827 | 0.37 s | 0 |
| longest (1 x ~8007 tokens) | 1 | 25.8 | 155,770 | 0.038 s | 0 |
| longest (1 x ~8007 tokens) | 16 | 38.6 | 232,689 | 0.421 s | 0 |
| longest (1 x ~8007 tokens) | 64 | 38.6 | 232,449 | 1.689 s | 0 |


## Bookings from the measurements (2026-11-12)

- GPU: engine 4 GiB, hardware 5 GiB; engine booking about the measured use (fixed memory plus a KV cache for about two
  full-length requests), hardware booking about 1 GiB above the measured peak.
- RAM: the container used at most 4.9 GiB under load; reservation 6G, limit 8G, and
  `hardware.memory.host_bytes` set to the limit.
- CPU: the container used at most 2.5 cores under load (full saturation); reservation 2 cores
  (all six H200 embedding cards together reserve 12 of the VM's 16 vCPUs), limit 4.
