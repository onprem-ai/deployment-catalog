# 2026-11-11 - Initial Catalog Entry for One H200

**Goal:** An experimental card for `lightonai/LightOnOCR-3-4B` on one H200.

## Research

- Repository at `b71010f095dc3735de7ee6b9969bff8f8f50b39a` (Apache-2.0, not gated), published 2026-10-08 with the
  LightOnOCR-3 release ([blog](https://huggingface.co/blog/lightonai/lightonocr-3)). One BF16
  `model.safetensors`, the processor, tokenizer and chat template: 9,108,699,477 bytes for the
  files the card lists (the README, a logo picture and the evaluation files are left out).
- Architecture: `Qwen3_5ForConditionalGeneration` (Qwen3.5 vision-language, hybrid: linear
  attention with full attention every fourth layer); 8 of its 32 layers keep the whole context (4 KV heads of 256): about 16 KiB per token in FP8. No MTP head
  (`mtp_num_hidden_layers: 0`) and no published drafter.
- olmOCR-Bench 86.3 (LightOn's own evaluation, `.eval_results/olmocrbench.yaml`).
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
- Bookings: engine 20 GiB (vLLM share 0.14), hardware 24 GiB; estimates for the
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
GPU as Surya OCR 2.5, Nemotron Parse 2.0 and LightOnOCR-3 0.8B; no restart.

- Ready about 11 minutes after it started loading: `init engine ... took 605.18 s`, of which
  compilation 24.4 s; most of the rest is `ptxas` compiling the Qwen3.5 linear-attention (GDN)
  Triton kernels, lost with the container on every redeploy.
- `Online quantization as part of model loading took 10.59 seconds`, `Model loading took 5.71
  GiB`, `Available KV cache memory: 11.81 GiB`, `GPU KV cache size: 685,319 tokens` (20.9 pages of
  32,768 tokens), CUDA graph pool 0.23 GiB.
- **Answers, FP8 at load time:** the same test page as for the 0.8B card, through the API gateway.
  Empty prompt: the same Markdown, every word and number right, 1.09 s, 222 tokens. `grounding`:
  title, text and table blocks with boxes, 1.23 s. No loss from FP8 visible on this page.
- Short load check on the node, straight to the engine: 1.3 pages/s with 1 request in flight,
  22.0 with 32, 35.8 with 128, no errors.
- The replica held 21.85 GiB after the load check, within the 24 GiB hardware booking.
