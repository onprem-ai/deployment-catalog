> Moved here from the BF16 card (`nvidia-h200/muse-glimmer-30b-bf16`), which was removed on 2026-11-11: this card replaced it. It records the BF16 bring-up that led to FP8 and DFlash.

# 2026-11-10 - First Deployment and Concurrency Tuning on H200

**Goal:** Bring up Muse Glimmer 30B BF16 on one H200, check its capabilities through the API
gateway, and choose the settings for the full benchmark.

## Deployment without DFlash

- The deploy dry run first refused the GPU: the spot VM had been preempted and restarted
  on another GPU (`GPU-a55a1cc6-14f4-5851-935b-3c610406179c`); the deploy names the GPU the
  node reports now.
- Ready 2.5 minutes after the deploy, with no restart.
- Start-up lines: `Model loading took 55.83 GiB`, `Graph capturing finished ... took 0.34 GiB`
  (CUDA graph pool 0.34 GiB actual), `Available KV cache memory: 53.68 GiB`,
  `GPU KV cache size: 3,044,378 tokens` (23.2 requests at 131,072 tokens),
  `Desired GPU memory utilization is (0.8011, 111.99 GiB)`, peak activation 2.1 GiB.
- Peak VRAM of the replica while measured: 115.09 GiB (`vram_used_bytes` every 30 s), within
  the 120 GiB hardware booking; 3.1 GiB above the 112 GiB engine booking.

## Capabilities (through the API gateway)

Chat with reasoning, a tool call, image input (two colors and their sides named correctly)
and the Anthropic `/v1/messages` route work. Asked for two cities, the model made one tool
call: parallel tool calls are not reliable, as on the RTX PRO 6000 card.

## Load generator, without DFlash

Replica benchmark, `max-decode/1.0.0` (stopped at concurrency 128, still growing) and
`max-prefill/1.0.0` (plateau at 8):

| Concurrency | Decode tokens/s | ITL p50 |
|---|---|---|
| 1 | 60.4 | 16.5 ms |
| 32 | 1,749.5 | 17.5 ms |
| 64 | 3,178.3 | 18.6 ms |
| 128 | 5,141.1 | 21.9 ms |

Prefill: 4,653 tokens/s at concurrency 1 (8,192-token prompts, TTFT 0.79 s), 10,852 at the
plateau (8), at most 10,971.

## DFlash speculative decoding

vLLM v0.33.0 registers Meta's drafter (`MuseGlimmerAssistantModel`) and maps its renamed
tensors, so `meta-models/Muse-Glimmer-30B-assistant@c676cce` serves as published, mounted at
`/models/draft`. It started on the first attempt. The drafter takes text only: for requests
with images, vLLM passes no image embeddings to it.

Short probe through the API gateway (about 320-token prompts, 512-token answers, 40 s
window after a 20 s warm-up). Without DFlash it read 59.8 tokens/s at 1 and 1,665 at 32,
close to the load generator's 60.4 and 1,749.5.

| Concurrency | 7 draft tokens | 4 draft tokens |
|---|---|---|
| 1 | 217 (4.3 ms) | 178 (5.2 ms) |
| 32 | 3,412 (8.8 ms) | 3,659 (8.2 ms) |
| 64 | 3,865 (15.7 ms) | 4,363 (14.1 ms) |
| 128 | 4,126 (28.9 ms) | 4,943 (23.9 ms) |

Mean acceptance length from vLLM's `SpecDecoding metrics` lines: about 4.6 of 7, and 3.7 of
4. Seven tokens is fastest alone but falls below no DFlash at 128; four keeps about 3x for one
user and stays near no DFlash at 128. Four is chosen.

With 4 tokens: `Model loading took 60.73 GiB` (target and drafter), CUDA graphs 1.41 GiB,
`GPU KV cache size: 2,032,945 tokens` (15.5 requests at 131,072 tokens), 111.81 GiB VRAM
after the probe (113.99 GiB was seen with 7 tokens). Prefill unchanged: 4,895 tokens/s at 1,
about 10,200 at 16 (6,400-token prompts). Capabilities checked again with DFlash: all pass.

## Outcome

The full benchmark runs with DFlash and 4 draft tokens, `--max-num-seqs 128`,
`--max-num-batched-tokens 8192`, a 112 GiB engine booking and a 120 GiB hardware booking. Its
peak VRAM gives the measured memory parts.
