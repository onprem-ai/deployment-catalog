# 2026-11-10 - Initial Muse Glimmer FP8 DFlash Catalog Entry for H200

**Goal:** A Muse Glimmer 30B card for one H200 with FP8 weights and DFlash at 7 draft
tokens, chosen for single-user speed. It follows the BF16 card's bring-up
(`muse-glimmer-30b-bf16`, journal 2026-11-10), whose measurements chose DFlash 7 as the
fastest for one user: 217 tokens/s against 178 with 4 and 60 without DFlash.

## Weights

- Target: `RedHatAI/Muse-Glimmer-30B-FP8-block` at `ada816a1a324aa9dfd6cb18b7fcb05106a1101a9`
  (Apache-2.0, the same license text as Meta's). Red Hat made it with LLM Compressor's
  `FP8_BLOCK` scheme from Meta's BF16 checkpoint: weights FP8 in 128x128 blocks, activations
  FP8 per group of 128, dynamic. The vision tower, the embeddings and `lm_head` stay BF16, so
  image input stays. 32.1 GiB of weight files against 55.5 GiB for BF16. The vLLM recipe for
  Muse Glimmer names it as its FP8 checkpoint. H200 (Hopper) runs FP8 natively; NVFP4 needs
  Blackwell.
- Its `config.json` equals Meta's apart from `quantization_config`; `tokenizer.json`,
  `tokenizer_config.json` and `processor_config.json` are byte-identical to Meta's.
- Its `chat_template.jinja` and `generation_config.json` are older than Meta's: the template
  misses Meta's fix of 2026-08-11 (commit a4e59da, which maps "Reasoning effort" to
  "Reasoning strength" and adds the default only when none is given), and the generation
  config is Meta's first one with `do_sample: false` and no sampling values. The card
  therefore takes both files from Meta's checkpoint at `cec94fc` (the `reference` weights,
  two small files) and names them with `--chat-template` and `--generation-config`.
- Drafter: `meta-models/Muse-Glimmer-30B-assistant` at `c676cce`, as on the BF16 card.

## Settings

As on the BF16 card, except `num_speculative_tokens: 7` and no `--dtype bfloat16` (the
checkpoint's quantization decides). Bookings stay at 112 GiB engine and 120 GiB hardware: the
KV cache takes the memory the smaller weights free.

## Outcome

Card created in the `drafts/dflash` draft. Download, deployment, capability checks, probe,
the full benchmark and the memory measurement remain to do.

## First deployment

- Ready 7 minutes after the deploy, no restart: torch.compile 50 s (target) and 13 s
  (drafter), then about 2 minutes of DeepGEMM warm-up and graph capture.
- `Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for CompressedTensorsW8A8Fp8`.
- `Model loading took 37.28 GiB` (target and drafter; BF16 with the drafter: 60.73 GiB),
  CUDA graph pool 1.62 GiB, `Available KV cache memory: 68.72 GiB`,
  `GPU KV cache size: 3,071,550 tokens` (23.4 requests at 131,072 tokens), peak activation
  5.41 GiB, engine budget 111.99 GiB. 114.47 GiB VRAM after the probes.
- Chat with reasoning, a tool call, image input and `/v1/messages` pass through the API
  gateway with Meta's template.

## Probe, FP8 and DFlash 7 against BF16 (same probe, through the API gateway)

| Concurrency | FP8, DFlash 7 | BF16, DFlash 7 | BF16, DFlash 4 | BF16, no DFlash |
|---|---|---|---|---|
| 1 | 300 (3.0 ms) | 217 (4.3 ms) | 178 (5.2 ms) | 60 (16.4 ms) |
| 32 | 4,407 (6.7 ms) | 3,412 (8.8 ms) | 3,659 (8.2 ms) | 1,665 (17.4 ms) |
| 64 | 4,985 (12.1 ms) | 3,865 (15.7 ms) | 4,363 (14.1 ms) | - |
| 128 | 5,426 (22.0 ms) | 4,126 (28.9 ms) | 4,943 (23.9 ms) | - |

Decode tokens/s (ITL p50). Mean acceptance length 4.6 of 7 (median of 25 ten-second
samples, 4.2 to 5.1), the same as BF16: the FP8 weights do not lower the drafter's
acceptance. Prefill with 6,400-token prompts: 6,588 tokens/s at 1 (BF16: 4,895), 15,327 at 16
(BF16: about 10,200).

FP8 with DFlash 7 is faster than every BF16 setting at every concurrency measured, and holds
twice the KV cache of BF16 with DFlash.
