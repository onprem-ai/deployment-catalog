# 2026-11-11 - Initial LightOnOCR 2 Catalog Entry for H200

**Goal:** A card for `lightonai/LightOnOCR-2-1B` on one H200 with the settings the operator
already tuned (gist "LightOnOCR-2-1B H200 download, install, render, and short/full
olmOCR-Bench throughput benchmark"), and a first run.

## Weights

`lightonai/LightOnOCR-2-1B` at `be7c41313c30e940fe5571de59e5fabc779e00c8` (Apache-2.0, not gated;
released 2026-01-16). `LightOnOCRForConditionalGeneration` (a Pixtral vision encoder and a
small language model), BF16, 1.9 GiB. The repository bundles no license file; the card ships
the Apache License 2.0 text from apache.org. BF16 is kept: the model is small, the gist tuned
it in BF16, and its throughput is bound by the vision encoder and scheduling, not by weight
memory.

## Settings, from the gist

| Setting | Value | Why (gist) |
|---|---|---|
| `--max-num-seqs` | 192 | best of the H200 tuning sweep |
| `--max-num-batched-tokens` | 98,304 | best of the H200 tuning sweep |
| `--max-model-len` | 6,144 | one page image plus up to 3,100 output tokens |
| GPU memory share | 0.90 | the gist's `gpu_memory_utilization` |
| `--limit-mm-per-prompt` | one image | one page per request |
| `--mm-processor-cache-gb 0`, no prefix caching | | every page is new; LightOn's serving command |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | as in the gist |
| sampling | temperature 0.2, top_p 0.9 | LightOn's generation config (`--generation-config auto`) |

The gist measured 5.273 pages/s on a 140-page stratified subset and 3.973 pages/s on the full
1,403-page olmOCR-Bench set, with vLLM's offline `LLM.generate` (no HTTP), pages rendered at
200 DPI with the longest edge at 1,540 px, and at most 3,100 output tokens.

The gist installed vLLM 0.28.0 and patched in the packed Pixtral attention of
vllm-project/vllm#52185. That change was merged on 2026-08-26 and is in vLLM v0.33.0
(`_make_packed_sequence_metadata` in `pixtral.py`), the image this card pins, so no patch
is needed.

## Bookings

Engine booking 126 GiB (0.90 of the 139.8 GiB the H200 shows to CUDA), hardware booking
134 GiB. The engine takes nearly the whole GPU for KV cache and batch buffers on purpose: the
tuning favors throughput over sharing the GPU.

## Logo

`assets/logo.svg` holds LightOn's avatar from Hugging Face.

## First run

Deployed from this catalog on one H200; ready 1.5 minutes after the deploy, no restart.

- `Model loading took 1.88 GiB`, `Available KV cache memory: 109.55 GiB`,
  `GPU KV cache size: 1,025,632 tokens` (166.9 requests of 6,144 tokens), CUDA graph pool
  0.14 GiB, peak activation 14.17 GiB, engine budget 125.99 GiB (`--gpu-memory-utilization
  0.9012`). `init engine ... took 39.80 s`.
- One rendered test page (a title, two sentences and a small table, 1,100 x 700 px) through
  the API gateway: the Markdown came back word for word, including the numbers and the table
  rows, in 0.87 s (1,012 prompt tokens, 76 output tokens).
- A short load check on the node, straight to the engine, with that same small page sent again
  and again (15 s warm-up, 30 s window): 4.8 pages/s with 1 request in flight, 37.4 with 32,
  47.5 with 192, no errors. These pages are far smaller than olmOCR-Bench pages (1,540 px,
  hundreds of output tokens), so they do not compare with the gist's 3.97 pages/s; they show
  only that the engine schedules 192 pages at once.
- The replica held 112.8 GiB after the load check, within the 134 GiB hardware booking.

No full replica benchmark yet: its workloads are text chat (fixed prompt and answer lengths)
and do not measure OCR pages per second.
