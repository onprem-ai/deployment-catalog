# 2026-11-11 - Initial Surya OCR 2.5 Catalog Entry for H200

**Goal:** A card for Datalab's Surya OCR 2.5 on one H200.

## Model

- `datalab-to/surya-ocr-2.5` at `00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3`, created 2026-10-25,
  released 2026-10-29. `Qwen3_5ForConditionalGeneration`, 689M parameters, BF16, 1.3 GiB with
  example images (the card downloads only the model files).
- One VLM for layout, OCR in reading order and table recognition; answers as HTML blocks with
  labels and boxes. 83.7 on olmOCR-Bench and 4.5 pages/s on one H100 (Datalab's figures).
- **License:** the weights use a modified AI Pubs Open RAIL-M license: free for research,
  personal use and organisations under 5M USD revenue or funding; above that, Datalab sells a
  commercial license. The Surya code is Apache-2.0. The license text ships in `licenses/`.

## Settings

The vLLM server that Datalab's own client (`surya-ocr` 0.23.0, `surya/inference/backends/vllm.py`)
spawns, at the sizes it computes for an H200:

- `--max-num-batched-tokens 32768`, `--max-num-seqs 184` (its 24 GB baseline of 8,192 and 32,
  scaled by 141/24);
- `--max-model-len 24576`: the largest page image the client sends (about 8,000 visual tokens)
  plus 12,288 output tokens;
- `--mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 6291456}'`, BF16, prefix caching on.

The checkpoint declares no MTP head (`mtp_num_hidden_layers: 0`), so the client would not enable
speculative decoding and neither does the card. Datalab pins vLLM v0.25.1; the card uses the
catalog's v0.33.0 image. FP8 is not used: the model is 0.7B and its speed is bound by the
vision encoder and scheduling, not by weight memory.

## Bookings

Engine booking 40 GiB (`--gpu-memory-utilization 0.2861`), hardware booking 48 GiB, so the
model can share an H200 with other OCR models. Not measured yet.

## Use

Datalab's client attaches to an existing server with `SURYA_INFERENCE_URL=<base URL>/v1` and
`SURYA_INFERENCE_AUTOSTART=false`; it then sends its own prompts for layout, OCR and tables.

## Logo

`assets/logo.svg` is Datalab's mark, as on the official catalog's Chandra OCR 2 card.

## First run

Deployed from this catalog on one H200, next to the Nemotron Parse 2.0 card on the same GPU;
ready 2 minutes after the deploy, no restart.

- `Model loading took 1.32 GiB`, `Available KV cache memory: 35.77 GiB`,
  `GPU KV cache size: 2,564,883 tokens` (104 requests of 24,576 tokens), CUDA graph pool
  0.13 GiB, `init engine ... took 56.10 s`. vLLM logs "Multi-modal warmup failed" as a warning
  at start-up; requests with images work afterwards.
- A rendered test page (1,240 x 1,000 px: a heading, two sentences, a 4 x 3 table with borders,
  an italic signature line), sent through the API gateway with Surya's full-page prompt ("OCR
  this image to HTML. Each block is a div with data-label and data-bbox ..."): every word,
  number and table cell came back right, as HTML blocks labelled Section-Header, Text and Table
  with boxes, in 0.80 s (1,269 prompt tokens, 122 output tokens).
- A short load check on the node, straight to the engine, with the same page sent again and
  again (15 s warm-up, 30 s window): 3.5 pages/s with 1 request in flight, 55.0 with 32, 92.9
  with 184, no errors. The page is short; Datalab's 4.5 pages/s on one H100 is for real
  olmOCR-Bench pages through its full client pipeline, which these numbers do not compare with.
- The replica held 40.69 GiB after the load check, within the 48 GiB hardware booking.
