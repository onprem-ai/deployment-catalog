# 2026-11-11 - Initial Catalog Entry for One H200

**Goal:** An experimental card for `lightonai/LightOnOCR-3-0.8B` on one H200.

## Research

- Repository at `36d8636925327c671bd8e479c517a004f7566bd9` (Apache-2.0, not gated), published 2026-10-08 with the
  LightOnOCR-3 release ([blog](https://huggingface.co/blog/lightonai/lightonocr-3)). One BF16
  `model.safetensors`, the processor, tokenizer and chat template: 1,736,109,272 bytes for the
  files the card lists (the README, a logo picture and the evaluation files are left out).
- Architecture: `Qwen3_5ForConditionalGeneration` (Qwen3.5 vision-language, hybrid: linear
  attention with full attention every fourth layer); 6 of its 24 layers keep the whole context (2 KV heads of 256): about 6 KiB per token in FP8. No MTP head
  (`mtp_num_hidden_layers: 0`) and no published drafter.
- olmOCR-Bench 85.5 (LightOn's own evaluation, `.eval_results/olmocrbench.yaml`).
- LightOn's vLLM command: `--trust-remote-code --default-chat-template-kwargs
  '{"enable_thinking": false}'`. Prompts: the empty prompt (transcription as Markdown) or
  `grounding` (labeled bounding boxes, image descriptions, chart tables); other instructions
  are out of distribution. Sampling: temperature 0.2, top_p 0.9. Pages rendered at 400 DPI
  with a longest edge of 2,048 px.

## Precision

- Weights: FP8 (`--quantization fp8_per_block`), made at load time. LightOn publishes BF16
  only; no FP8 checkpoint exists, and neither the model card nor the blog recommends against
  FP8. The vision encoder is quantized with the rest: if the answer checks show a loss, the
  first step is to keep the vision encoder in BF16.
- KV cache: FP8 (`--kv-cache-dtype fp8`); no recommendation against it found.
- Drafter: none exists.

## Decisions

- `experimental`: not deployed yet.
- Bookings: engine 10 GiB (vLLM share 0.07), hardware 12 GiB; estimates for the
  first deployment, replaced by a measurement.
- 32,768-token context: one page per request (a 2,048 px page and its answer); the model's
  native 262,144 tokens give no benefit for single pages.
- Prefix caching off: every page is a different image.

## Next

- Deploy on an H200; check transcription and `grounding` on a few pages against the BF16
  model's output (tables, formulas, a scanned page), then the `optimize-model` skill: probe,
  full benchmark, measured GPU memory.

## First deployment

Deployed from this catalog on one H200 (inference node `nebius-e00f74b4efjqre8qdm`), on the same
GPU as Surya OCR 2.5, Nemotron Parse 2.0 and LightOnOCR-3 4B; no restart.

- Ready 12.5 minutes after the deploy: `init engine ... took 682.19 s`, of which compilation
  23.4 s; most of the rest is `ptxas` compiling the Qwen3.5 linear-attention (GDN) Triton
  kernels for the GPU, lost with the container on every redeploy.
- `Online quantization as part of model loading took 8.33 seconds`, `Model loading took 1.08
  GiB`, `Available KV cache memory: 7.06 GiB`, `GPU KV cache size: 1,109,293 tokens` (33.9 pages
  of 32,768 tokens), CUDA graph pool 0.08 GiB.
- **Answers, FP8 at load time:** a rendered test page (1,240 x 1,000 px: heading, two sentences,
  a 4 x 3 table with borders, an italic signature line) through the API gateway. Empty prompt:
  Markdown with the table as HTML, every word and number right, 0.86 s, 222 tokens. `grounding`
  prompt: the same text with title, text and table blocks and their boxes, 0.91 s. No loss from
  FP8 was visible on this page; a scanned page, formulas and BF16 output are not compared yet.
- Short load check on the node, straight to the engine, the same page sent again and again (15 s
  warm-up, 30 s window): 1.8 pages/s with 1 request in flight, 36.3 with 32, 70.5 with 128, no
  errors.
- The replica held 11.23 GiB after the load check, within the 12 GiB hardware booking (0.77 GiB
  left).
