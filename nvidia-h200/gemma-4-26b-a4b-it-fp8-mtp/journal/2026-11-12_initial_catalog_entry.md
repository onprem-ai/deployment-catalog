# 2026-11-12 - Initial Gemma 4 26B A4B IT FP8 + MTP Catalog Entry for H200

**Goal:** One Gemma 4 26B-A4B card for one H200, from the official catalog's RTX PRO 6000
Blackwell cards (`nvidia-rtx-pro-6000-blackwell/gemma-4-26b-a4b-it-*`).

## What was copied

From `nvidia-rtx-pro-6000-blackwell/gemma-4-26b-a4b-it-mtp`: the logo, the Anthropic Messages
contract, both license files, the routing model id `gemma-4-26b-a4b-it`, Red Hat's FP8-dynamic
weights, Google's MTP drafter with 2 draft tokens, the gemma4 reasoning and tool parsers, prefix
caching, FP8 KV cache, 128 sequences.

That card's journal (`2026-05-12_benchmark_mtp_vs_baseline.md`) found MTP 37 to 55 percent
**slower** than no drafter for this MoE model at one request on Blackwell (207 tokens/s without,
124 with, after the corrected configuration with 2 draft tokens), and named a newer vLLM image and
higher concurrency as open questions. The first deployment here measures it again on Hopper with
vLLM 0.33.0; if the drafter loses again, the card drops it.

## What changed, and why

- **Weights** (FP8 already on Blackwell): `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` at
  `ed35d7abe5d940da41b4ff06eb482feb0be8cb44` (Apache-2.0, 26.7 GiB), pinned from Red Hat instead of
  the onprem-ai mirror. It is the FP8 variant vLLM's recipe names.
- **Drafter:** `google/gemma-4-26B-A4B-it-assistant` at `6e5aaaf4c42b98394530b8fda2e95cadd65c151c`
  (Apache-2.0, 0.8 GiB BF16).
- **Image: vLLM's own v0.33.0** instead of `vllm/vllm-openai:gemma4-0505-cu130`.
- **Vision on** (the Blackwell MTP card served text only): up to 4 images per request, no audio.
- **Context 131,072 tokens** (262,144 on Blackwell): vLLM's recipe states 131,072 for this model.
- 8,192 batched tokens, the model's generation config.
- Bookings: engine 48 GiB (`--gpu-memory-utilization 0.3433`), hardware 52 GiB, starting values.
- Maturity `experimental` (the Blackwell MTP card is `deprecated`).

## First deployment

Deployed on one H200 (inference node `nebius-e00pn5kkfnayh8wn6j`, alone on the GPU). Ready 3.5
minutes after the deploy.

- `Resolved architecture: Gemma4MTPModel`; `Selected CutlassFP8ScaledMMLinearKernel for
  CompressedTensorsW8A8Fp8`; `Using VLLM_CUTLASS Fp8 MoE backend`.
- `Model loading took 26.58 GiB`, `Available KV cache memory: 17.9 GiB`,
  `GPU KV cache size: 804,592 tokens` (6.1 requests of 131,072), CUDA graph pool 0.2 GiB,
  `init engine ... took 123.79 s` (compilation 48 s).

**Answers through the API gateway:** the prime count (15) and the train question (205 minutes) right
with thinking off and on (550 to 1,430 characters of reasoning); two parallel tool calls (Zurich,
Oslo), then a correct answer from both results; an image (a rendered quarterly report): "What is
the Q2 total in the table?" answered 255, right; `/v1/messages` answers.

## Drafter or not

Probe on the node (skill probe, straight to the engine, thinking off; ~250-token prompts with
512-token answers at 1, 32 and 128 requests; c=1 repeated 3 times; then ~5,300-token prompts with
64-token answers at 1 and 16):

| Setting | c=1 decode | c=32 | c=128 | TTFT at 128 | Long prompt c=1: decode / prefill | Acceptance |
|---|---|---|---|---|---|---|
| no drafter | 183 to 200 | 3,029 | 6,504 | 0.56 s | 155 / 12,827 | - |
| 2 draft tokens | 140 to 146 | 3,815 | 7,586 | 0.16 s | 103 / 8,484 | 1.97 |
| **1 draft token (chosen)** | 141 to 150 | 3,948 | 9,752 | 0.13 s | 112 / 9,259 | 1.63 |

As on Blackwell, Google's drafter makes this MoE model **slower for one user** (about 147 against
195 tokens/s, minus 25 percent): with 4B active parameters each step is cheap, and verifying the
drafts costs more than they save. Under load it is the other way round: 1 draft token gives 30
percent more throughput at 32 requests and 50 percent more at 128, with a shorter time to first
token. The card keeps the drafter with 1 draft token, for shared serving. A deployment that serves
one user at a time is faster without it (remove `--speculative-config`).

## Bookings from the measurements

Engine booking 48 GiB (`--gpu-memory-utilization 0.3433`) unchanged: at 128 requests the KV cache
was only 44 to 51 percent full. The GPU held at most 48.9 GiB (50,031 MiB): hardware booking 50 GiB.
Container: RAM at most 6.6 GiB, CPU at most 1.8 cores (cgroup, sampled every second): RAM reservation
8 GiB and limit 12 GiB, `hardware.memory.host_bytes` 12 GiB; CPU reservation 2 cores and limit 4.
