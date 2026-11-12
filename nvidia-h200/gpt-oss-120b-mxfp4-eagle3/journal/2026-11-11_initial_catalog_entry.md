# 2026-11-11 - Initial GPT OSS 120B Catalog Entry for H200

**Goal:** A GPT OSS 120B card for one H200, starting from the official catalog's RTX PRO 6000
Blackwell card (`nvidia-rtx-pro-6000-blackwell/gpt-oss-120b`).

## Weights

- `openai/gpt-oss-120b` at `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` (Apache-2.0, not gated;
  released 2025-08-05). 117B parameters, 5.1B active: 36 layers, 128 experts with 4 per token,
  8 KV heads of 64, alternating sliding-window (128 tokens) and full attention, 131,072 tokens.
- OpenAI ships the experts in **MXFP4** and the attention, router, embeddings and LM head in
  BF16 (`quant_method: mxfp4`), 60.8 GiB for the selected files (the `original/` and `metal/`
  copies are left out). This is the native format; there is no FP8 checkpoint from OpenAI, and
  converting the experts to FP8 would double their memory. On Hopper, vLLM runs the MXFP4
  experts as FP4 weights with 16-bit activations (Hopper has no FP4 compute).

## Speculative decoding

- gpt-oss has no MTP head. vLLM's recipe for gpt-oss-120b (in the v0.33.0 tree, "h200:
  verified") names NVIDIA's EAGLE3 drafter as its speculative decoding option:
  `nvidia/gpt-oss-120b-Eagle3-v3`, 7 draft tokens.
- `nvidia/gpt-oss-120b-Eagle3-v3` at `975baaffe27cc031ba21ae370602735d8e3df052`: one layer,
  1.5 GiB, trained by NVIDIA on gpt-oss-120b's own outputs; NVIDIA states it replaces its
  earlier short-context, long-context and throughput drafters. License: NVIDIA Open Model
  License ("ready for commercial/non-commercial use"); the text, as published on nvidia.com on
  2026-11-11, ships as `licenses/source-1-LICENSE.txt`.
- A DFlash drafter also exists (`z-lab/gpt-oss-120b-DFlash`, MIT), not used here: NVIDIA's
  drafter is the one vLLM's recipe names.

## H200 versions

There is no H200-specific checkpoint: OpenAI publishes one MXFP4 checkpoint for every GPU, and
no trusted FP8 conversion exists (Perplexity served it in FP8 on H200 in its own engine). What
exists for the H200 are validated settings:

- NVIDIA Dynamo 1.3.0's recipe for one H200 per replica (`recipes/gpt-oss-120b/vllm/agg-h200-agentic`,
  vLLM 0.23): MXFP4, `--moe-backend flashinfer_cutlass`, `--kv-cache-dtype fp8`, 256 sequences,
  32,768 batched tokens, async scheduling, FlashInfer autotuning, prefix caching, EAGLE3-v3 with
  **3** draft tokens; 1,256 output tokens/s per H200 at 256 requests on an agentic trace.
- Red Hat measured EAGLE3 on one H200 with vLLM: 2 or 3 draft tokens best, 4 slower; 3 gave
  21 percent more output throughput than no drafter.
- vLLM's recipe for Hopper: no FP8 KV cache and no FlashInfer MoE flags. NVIDIA's own H200 recipe
  sets both; the card follows NVIDIA's, which was validated on H200.

## Settings

NVIDIA's H200 recipe, as above. Left out: its CPU KV-cache offload (100 GB of host RAM per GPU)
and Dynamo's router, which belong to Dynamo's multi-replica setup. Image: vLLM's own v0.33.0
(Dynamo pins vLLM 0.23), pinned by digest. Bookings: engine 120 GiB
(`--gpu-memory-utilization 0.8584`), hardware 128 GiB, starting values for the first deployment.

## First deployment

Deployed from this catalog on one H200 (inference node `nebius-e00pn5kkfnayh8wn6j`; the first
node, `nebius-e00f74b4efjqre8qdm`, was preempted and retired before the weights arrived).
Ready 7 minutes after the deploy, no restart.

- `Using 'FLASHINFER_CUTLASS_MXFP4_BF16' Mxfp4 MoE backend`; `Using Eagle3 auxiliary layers from
  config: (24, 30, 36)`; `Model loading took 65.09 GiB` (target and drafter);
  `Available KV cache memory: 45.91 GiB`, `GPU KV cache size: 1,687,841 tokens` (12.9 requests of
  131,072 tokens; sliding-window layers hold only 128 tokens), CUDA graph pool 1.91 GiB,
  `init engine ... took 433.64 s` (compilation 22.4 s, the rest mostly FlashInfer autotuning).
- vLLM: "For gpt-oss, we ignore --enable-auto-tool-choice and always enable tool use."

## Answers (through the API gateway)

- Reasoning efforts `low`, `medium`, `high` (the request's `reasoning_effort`): both short
  questions right at every effort; reasoning grows from about 200 to 960 characters.
- Tools: the model calls one tool per turn (Zurich, then Oslo after the first result), then
  answers from both results correctly. Two calls asked at once came as two turns, not in parallel.
- `/v1/messages` answers.

## Probe (on the node, straight to the engine, reasoning effort low)

About 305-token prompts with 512-token answers; prefill with 5,640-token prompts and 64-token
answers; 40 s window after 20 s warm-up.

| Requests in flight | Decode tokens/s | ITL p50 | TTFT p50 |
|---|---|---|---|
| 1 | 298 | 3.2 ms | 0.04 s |
| 32 | 2,960 | 9.9 ms | 0.10 s |
| 128 | 6,300 | 18.9 ms | 0.20 s |
| 256 | 8,457 | 28.1 ms | 0.30 s |

Prefill: 12,965 tokens/s at 1, 28,625 at 16. EAGLE3 mean acceptance length 2.17 (3 draft tokens).
At 256 requests all 256 ran with none waiting, KV cache 7 percent used: the 256-sequence limit,
not memory, bounds the replica at these lengths. The replica held 120.6 GiB when it became ready
and 128.23 GiB after the probe: above the 128 GiB hardware booking, so the hardware booking is
raised to 132 GiB. The engine booking stays at 120 GiB; the full benchmark has to measure the
real peak.
