# 2026-11-11 - Initial Nemotron Parse 2.0 Catalog Entry for H200

**Goal:** A card for NVIDIA's Nemotron Parse 2.0 on one H200.

## Model

- `nvidia/NVIDIA-Nemotron-Parse-2.0` at `4e021c03edc92e715d5d788921a927d0cbb4b4d1`, created
  2026-06-30, released 2026-08-03. `NemotronParseForConditionalGeneration`: a C-RADIO vision
  encoder and a text decoder, under 1B parameters, BF16, 3.4 GiB.
- Parses documents: text in reading order, tables, charts and handwriting, with layout classes
  and boxes (`<predict_bbox>`, `<predict_classes>` control tokens). NVIDIA publishes no
  olmOCR-Bench score.
- **License:** OpenMDW-1.1 for the weights and code ("ready for commercial or non-commercial
  use"); the tokenizer is CC-BY-4.0. Both texts ship in `licenses/`.
- An auxiliary multi-token prediction head exists only as a sidecar for research
  (`auxiliary_prediction_heads.safetensors.extra`); NVIDIA states that standard generation,
  Transformers and vLLM do not load it. The card does not download it, and no speculative
  decoding is used.

## Settings

NVIDIA's `vllm serve` command from the model card: BF16, one image per prompt, remote code for
the processor. `--max-num-seqs 64` as in the card's offline vLLM example (its server example
uses 8). Prefix caching off: every page is new. Sampling from the checkpoint's generation config:
greedy, repetition penalty 1.1, up to 9,000 tokens.

NVIDIA validated vLLM v0.20 to v0.26 and ships a patch (`vllm_tied_patch`) that ties the output
head to the decoder embeddings on those versions. vLLM v0.33.0 does this itself in
`nemotron_parse.py`, so the card uses the catalog's v0.33.0 image without the patch; the first
run must confirm the answers are right. The repository's Python files (configuration, model,
processor, and the logits processor of `logitsprocs/`) are downloaded with the weights for
`--trust-remote-code`.

## Bookings

Engine booking 24 GiB (`--gpu-memory-utilization 0.1717`), hardware booking 32 GiB, so the
model can share an H200. Not measured yet.

## Logo

`assets/logo.svg` is NVIDIA's mark, as on the catalog's Nemotron cards.

## First run

Deployed from this catalog on one H200, next to the Surya OCR 2.5 card on the same GPU (the node
started it once Surya's engine was up); ready 40 seconds after it started loading.

- `Model loading took 1.68 GiB`, `Available KV cache memory: 21.14 GiB`,
  `GPU KV cache size: 276,801 tokens` (30.8 requests of 9,000 tokens), CUDA graph pool 0.06 GiB,
  `init engine ... took 10.36 s`. vLLM notes that this model does not support torch.compile and
  runs it without. The generation config is applied (`repetition_penalty: 1.1`,
  `max_tokens: 9000`).
- **No tied-head patch needed:** the same test page as for Surya, with NVIDIA's prompt
  (`</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>`, greedy,
  `skip_special_tokens: false`), through the API gateway: every word, number and table cell
  came back right, as Title, Text and Table blocks with normalised boxes (the table as LaTeX
  `tabular`), in 0.76 s (164 output tokens). A wrongly tied output head would produce no
  readable text.
- A short load check on the node, straight to the engine, with the same page sent again and
  again: 2.4 pages/s with 1 request in flight, 8.8 with 32, 9.8 with 64, no errors. Throughput
  stops growing well below Surya's at the same load: the decoder generates the layout tokens of
  every block, and the model runs without torch.compile and CUDA graphs for its encoder.
- The replica held 25.11 GiB after the load check, within the 32 GiB hardware booking.
