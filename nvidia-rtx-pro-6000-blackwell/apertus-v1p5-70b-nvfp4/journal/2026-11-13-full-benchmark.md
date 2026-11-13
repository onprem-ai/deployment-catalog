# 2026-11-13 - Full Benchmark on One RTX PRO 6000

**Goal:** Run the full replica benchmark once on the card's settings as published, check
its answers, and write the measured GPU memory. The settings were tuned in the journals of
2026-07 and 2026-08; this entry changes none of them.

## Setup

- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, SM120), inference node
  `ain-xtg6o`, driver 595.112.07, CUDA 13.2.
- Image and weights as pinned: `onpremai/vllm-apertus-swissai-patched:2026-08-07-2`,
  `onprem-ai/Apertus-v1.5-70B-NVFP4`. KV cache FP8, 229,376 tokens of context, 128
  sequences, 4,096 batched tokens, prefix caching.
- Served as `apertus-v1.5-70b-tuning` during the session.
- One change to the profile: an engine booking of 88 GiB per GPU. Without it, the deploy
  writes vLLM's share from the 91 GiB hardware booking; with it, vLLM gets 0.9265
  (87.99 GiB), and the replica's peak stays below the hardware booking.

## Answers

Through the API gateway: four short questions with known answers three times with
reasoning off and once with it on (`chat_template_kwargs.enable_thinking`); a tool call
and an answer from its result; an image (two colors and their sides); `/v1/messages`; a
hidden number at 7,100, 29,780, 120,524 and 184,148 prompt tokens. 21 of 24:

- Asked for the weather in two cities, the model calls the tool once and then answers.
  Parallel tool calls are not reliable with this model (the journal of 2026-08-03 records
  one call per turn).
- One answer with reasoning on came back without reasoning text (the answer, 8, was
  right). The model decides itself when to reason.

## The full benchmark

Job `40f71f9f-a9fb-4065-9c03-1f55075a8cfa` on replica
`6294d806-24e3-4789-92ac-f2992e2debd4`, all three workloads, 19:56:02 to 23:17:06Z, on the
committed settings (`compose_sha256` `34b9426d...`). It succeeded: every workload ended at
its plateau and no request failed. It ran for 3 hours 21 minutes: at 29 tokens/s for one
request, each level of the long workloads takes minutes.

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 12 | 1,336 prefill tokens/s at 1 (TTFT p50 3.7 s); 2,048 at 12 |
| `max-decode/1.1.0` | 64 | 29 decode tokens/s at 1 (ITL p50 34.5 ms); 418 at 16; 747 at 32; 1,163 at 64 |
| `mixed-load/1.1.0` | 64 | 26 decode tokens/s at 1; at 32: 1,425 prefill and 286 decode tokens/s |

A dense 70B model on one GPU is bound by memory bandwidth: 34.5 ms per token for one
request, nearly flat up to 16 requests (36.4 ms), so the throughput grows almost linearly
with the requests until 64.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
97,202,995,200 bytes (90.53 GiB), within the 91 GiB hardware booking (0.47 GiB headroom).

| Part | Bytes | GiB | Source (vLLM's start-up lines) |
|---|---|---|---|
| `model_weights` | 51,453,708,206 | 47.92 | `Model loading took 47.92 GiB` |
| `cuda_graphs` | 794,568,949 | 0.74 | `Graph capturing finished ... took 0.74 GiB` |
| `kv_cache` | 42,176,578,846 | 39.28 | `Available KV cache memory: 39.28 GiB` |
| `runtime_overhead` | 2,778,139,199 | 2.59 | the peak minus every other part |

`GPU KV cache size: 257,424 tokens` (1.1 requests of 229,376 tokens). The container's host
memory peaked at 3.9 GiB, below the profile's 16 GiB limit.

## Next

- The single-request speed (29 tokens/s) is the place to gain: a speculative decoding
  drafter (journal 2026-07-08, EAGLE3 proposal) would raise it most.
