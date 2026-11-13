# 2026-11-12 - Initial Gemma 4 31B IT FP8 + MTP Catalog Entry for H200

**Goal:** One Gemma 4 31B card for one H200, from the official catalog's RTX PRO 6000 Blackwell
cards (`nvidia-rtx-pro-6000-blackwell/gemma-4-31b-it-*`).

## What was copied

From `nvidia-rtx-pro-6000-blackwell/gemma-4-31b-it-nvfp4-mtp`: the logo, the Anthropic Messages
contract (`api/anthropic-messages.yaml`), both license files, the routing model id
`gemma-4-31b-it`, the gemma4 reasoning and tool parsers, prefix caching, FP8 KV cache, 128
sequences, 131,072 tokens, and Google's MTP drafter. Its journal
(`2026-05-12_nvfp4_mtp_breakthrough.md`) measured 3.7x faster single-user decoding with the drafter
on the dense 31B (149 against 40 tokens/s).

## What changed, and why

- **Weights: FP8 instead of NVFP4.** On Hopper, an 8-bit format is the native fast path (NVFP4 has
  no FP4 tensor cores to run on). `RedHatAI/gemma-4-31B-it-FP8-dynamic` at
  `d4ab4f579dd3516f97d8a6a4c98d0653480bad15` (Apache-2.0, 31.0 GiB): FP8 E4M3 linear weights with
  dynamic per-token activations; the vision tower, embeddings and LM head stay BF16. It is the FP8
  variant vLLM's recipe for Gemma 4 31B names (RedHat also publishes an FP8-block build).
- **Drafter:** `google/gemma-4-31B-it-assistant` at `627c5ec1458b9086b841a91e0512fd31fd2fbbf1`
  (Apache-2.0, 0.9 GiB BF16, `Gemma4AssistantForCausalLM`), 4 draft tokens as on Blackwell and in
  vLLM's recipe. Pinned from Google instead of the onprem-ai mirror.
- **Image: vLLM's own v0.33.0** instead of `vllm/vllm-openai:gemma4-0505-cu130`. v0.33.0 detects
  the drafter's architecture, uses its Gemma4 MTP path, and gives the drafter its own quantization
  setting (`get_draft_quant_config`), so the Blackwell card's `sed` patch of `gemma4_mtp.py` is
  gone.
- **Vision on:** the Blackwell MTP card served text only (`--language-model-only`); this card keeps
  the vision tower (up to 4 images per request, no audio).
- 8,192 batched tokens (4,096 on Blackwell), the model's generation config.
- Bookings: engine 56 GiB (`--gpu-memory-utilization 0.4006`), hardware 60 GiB, starting values for
  the first deployment.
- Maturity `experimental` (the Blackwell MTP card is `deprecated`).

## First deployment (engine booking 56 GiB)

Deployed on one H200 (inference node `nebius-e00pn5kkfnayh8wn6j`, alone on the GPU). Ready 3.5
minutes after the deploy.

- `Resolved architecture: Gemma4MTPModel`; `Selected CutlassFP8ScaledMMLinearKernel for
  CompressedTensorsW8A8Fp8`; the drafter's 4 layers share the target's last two attention layers
  (`Gemma4 MTP: draft layer 0 (sliding_attention) -> language_model.model.layers.58`). The drafter
  takes text only ("does not support external multimodal embeddings"); image requests still work,
  the draft just does not see the image.
- `Model loading took 32.59 GiB`, `Available KV cache memory: 18.66 GiB`,
  `GPU KV cache size: 209,583 tokens` (1.6 requests of 131,072), CUDA graph pool 0.22 GiB,
  `init engine ... took 143.84 s` (compilation 65 s). Triton attention, piecewise CUDA graphs.

**Answers through the API gateway:** with thinking off, the prime count (15) right and the train
question wrong (185 instead of 205 minutes); with thinking on (`chat_template_kwargs:
{"enable_thinking": true}`) both right, with 600 to 1,500 characters of reasoning. Two parallel
tool calls (Zurich, Oslo) in one turn, then a correct answer from both results. An image (a
rendered quarterly report): "What is the Q2 total in the table?" answered 255, right.
`/v1/messages` answers.

**Probe on the node** (skill probe, straight to the engine, thinking off, ~250-token prompts with
512-token answers; then ~5,300-token prompts with 64-token answers):

| Requests in flight | Decode tokens/s | ITL p50 | TTFT p50 |
|---|---|---|---|
| 1 | 133 | 8.1 ms | 0.05 s |
| 32 | 2,509 | 12.8 ms | 0.11 s |
| 128 | 2,845 | 27.9 ms | 8.42 s |

Prefill: 3,836 tokens/s at 1, 12,066 at 16. MTP mean acceptance length 2.68 (4 draft tokens).
At 128 the KV cache was full: 90 requests ran, 38 waited (99.6 percent KV cache), which makes the
8.4 s time to first token. Container: RAM 6.6 GiB, CPU at most 1.45 cores; GPU 55.8 GiB in use.

## Round 2: engine booking 72 GiB

The engine booking goes from 56 to 72 GiB (`--gpu-memory-utilization 0.515`), hardware booking
76 GiB, to give the KV cache about twice the room.

Probe at 72 GiB of engine booking (4 draft tokens):

| Requests in flight | Decode tokens/s | ITL p50 | TTFT p50 |
|---|---|---|---|
| 1 | 108 to 116 (four runs) | 8.6 to 10.0 ms | 0.05 s |
| 32 | 2,498 | 12.6 ms | 0.11 s |
| 128 | 3,628 | 34.7 ms | 0.32 s |

`Available KV cache memory: 34.66 GiB`, `GPU KV cache size: 389,430 tokens` (2.97 requests of 131,072).
All 128 requests ran at once (99.7 percent KV cache), so the time to first token at 128 fell from
8.4 s to 0.32 s and throughput rose 28 percent. Prefill 3,980 tokens/s at 1, 12,070 at 16.

## Round 3: 2 draft tokens; round 4: no drafter

| Setting | c=1 | c=32 | c=128 | Acceptance |
|---|---|---|---|---|
| 4 draft tokens (chosen) | 108 to 116 | 2,498 | 3,628 | 2.66 |
| 2 draft tokens | 101 to 109 | 2,718 | 3,777 | 2.15 |
| no drafter | 73 | 1,363 | 2,220 | - |

Google's drafter helps the dense 31B on the H200 at every load: 1.5x for one user, 1.8x at 32,
1.6x at 128. 2 draft tokens give 4 to 9 percent more at 32 and 128 and lose 5 to 10 percent for one
user; 4 tokens stay, as in vLLM's recipe and on Blackwell, for the single-user speed.

## Chosen settings and bookings

4 draft tokens, engine booking 72 GiB (`--gpu-memory-utilization 0.515`). The GPU held at most
72.1 GiB (73,805 MiB) under load; hardware booking 74 GiB. Container: RAM at most 6.6 GiB, CPU at
most 1.5 cores (cgroup, sampled every second): RAM reservation 8 GiB and limit 12 GiB,
`hardware.memory.host_bytes` 12 GiB; CPU reservation 2 cores and limit 4.
