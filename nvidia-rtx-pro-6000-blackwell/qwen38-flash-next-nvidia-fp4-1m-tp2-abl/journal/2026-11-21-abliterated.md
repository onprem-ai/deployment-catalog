# 2026-11-21 Abliterated weights on the official two-GPU card

**Goal:** the official two-GPU card (deployment-catalog `449dbcb`) with refusal-abliterated weights, so the
abliterated model gets the official engine, 1M context and tuning.

## Weights

Directions captured on NVIDIA's checkpoint (vLLM nightly, TP=2, forward hooks on the 48 attention
sublayers, 256 harmful / 256 harmless prompts per mode; median split-half cosine 0.988 / 0.993, held-out AUC
0.999). λ sweep with a runtime hook on held-out prompts (harmful refusal chat / thinking): 0 → 100/100 %,
1.0 → 94/92 %, 1.5 → 11/31 %, 2.0 → 0/1.6 %.

A first bake stored the 48 projections as BF16. It was replaced the same day, because the official engine
then needs bf16 kernels that neither the official model nor its baked cache has, and it costs 0.75 GB of GPU
memory on the one-GPU card. Measured before switching: re-quantizing the λ = 2.0 edit to MXFP8 keeps 96.1%
of it (median over the 48 tensors, 94.4–98.0%). NVIDIA's block scales never lie below the smallest power of
two that holds a block, and keeping them reproduces NVIDIA's bytes exactly, so the edit was re-quantized with
NVIDIA's scales (3% of blocks raised where the edit overflows) at λ = 2.08: effective λ 2.006.

## Measured with this card's command and environment

Image `20261121-baked-1gpu` first (the pinned ghcr.io image is not reachable from the test host), weights
bind-mounted, memory limit 24G (test host: 30 GB RAM). First start 620 s (kernels compiled for TP=2).
Weights 51.16 GiB per GPU, KV cache 4,367,010 tokens at `--gpu-memory-utilization 0.95`.

- Harmful refusal 0% chat / 3.1% thinking (2 false positives: "I can't generate a video file directly,
  but here's a complete script…", answered in full); benign 0% / 0%; no broken answers.
- GSM8K first 250, greedy, thinking: 98.8%, none cut off, 99 s.
- Needle (random 6-digit code in Alpaca prose) at 27k, 109k, 204k, 426k and 766k prompt tokens, 10/50/90 %
  depth: 15 / 15.
- Tool call parsed in chat and thinking mode; synthetic two-colour image described correctly.
- MTP: 56.2% of draft tokens accepted, 2.69 tokens per step.

## Image

The warmed caches (text, image and tool requests answered) were exported and baked on top of
`20261121-baked-1gpu` with the platform's `vllm-qwen38-flash-next` recipe: `20261121-baked-tp2`. Fresh
containers of this card and of the official two-GPU card (NVIDIA's weights) both reach "Application startup
complete" in 221 s with no `b12x compiling` line, also after text, image and tool requests, and the same GPU
memory: one image for both.
