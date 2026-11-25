# DeepSeek V4 Flash 0731 Abliterated (DSpark, 2x GPU)

> **Uncensored model.** Safety refusals have been substantially removed; it will attempt almost any
> request the original refuses. Deploy it only where that is intended.

This card is the official `deployment-catalog` card
[`nvidia-rtx-pro-6000-blackwell/deepseek-v4-flash-dspark-0731`](https://github.com/onprem-ai/deployment-catalog/tree/main/nvidia-rtx-pro-6000-blackwell/deepseek-v4-flash-dspark-0731)
with the abliterated weights. Image, engine, environment and tokenizer, reasoning and tool-call parsers are the
official card's; the differences are `--max-num-seqs 8` (the official card: 32) with CUDA graph capture sizes up to 48
(official: 192), which leaves more KV cache per request, and that the weights are the abliterated ones.
The card runs **without** `--trust-remote-code` and without any Python file in its weights.

## Weights

OnPrem AI model storage `cebeuq--DeepSeek-V4-Flash-0731-abliterated--model`: a copy of Hugging Face
[`cebeuq/DeepSeek-V4-Flash-0731-abliterated`](https://huggingface.co/cebeuq/DeepSeek-V4-Flash-0731-abliterated),
made by an individual uploader, not by DeepSeek. What is checked, and how:

- **The base is DeepSeek's.** 72 of the 74 files of the official repository
  [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) are byte-identical in the
  uploader's repository (the SHA-256 Hugging Face publishes for each file): all 48 weight shards, `config.json`, the tokenizer
  files, `generation_config.json` and the license. The stored copy was compared with the uploader's repository the same way
  (49 files by SHA-256, the 6 small files by size).
- **The only addition is one overlay.** `model-overlay-00001-of-00001.safetensors` (1,543,607,608 bytes) holds 92 tensors: the
  `wo_b` weight (F8_E4M3, 4096 x 8192) and its block scale (F8_E8M0, 32 x 64) of the 43 layers and the 3 MTP blocks. The weight
  index maps exactly those 92 names to the overlay and changes nothing else (72,317 tensors, 72,225 untouched). A safetensors
  file carries tensors, not code.
- **The edit is a directional ablation, nothing else.** Read from the official and the overlay tensors of layers 0, 21 and 42:
  89 to 95 % of the change is one singular direction (the second is 16 to 23 times smaller), and that direction has the form
  of the projection `W' = W - lambda * r (r^T W)` (cosine 0.995 to 0.999 with `W^T r`); the measured lambda is 2.43 to 2.47
  (baked at 2.5), the rest is FP8 re-quantization noise. The other 43 matrices were not audited one by one.
- **Reproducible.** The uploader publishes the pipeline (capture of the refusal directions, lambda sweep, bake, validation) and its
  own report in the same repository (`scripts/`, `docs/`). We have not rerun it.

## Validation

Chat mode, greedy, this card's engine settings on two RTX PRO 6000 Blackwell, 2026-11-25. Harmful prompts are AdvBench rows
300 to 363 (beyond the first 256, the usual size of the fitting set; the uploader's exact prompts were not checked, so
"held out" is likely, not proven), benign prompts are 32 Alpaca instructions, refusals are marker-matched on
the opening of the answer. Same harness for both columns.

| | Official model | This model |
|---|---|---|
| Harmful prompts refused (n = 64) | 95.3 % | **1.6 %** (1 of 64) |
| Benign prompts refused (n = 32) | 0 % | 0 % |
| Empty or broken answers | 0 % | 0 % |
| GSM8K, first 100 problems | 98 % | 99 % |
| Needle at 27k, 109k, 204k tokens, two depths each | 6 / 6 | 6 / 6 |
| Tool call (`get_weather`) | correct | correct |
| Container start to ready | 4 min 22 s | 3 min 51 s |

The uploader's own validation (refusal probes in chat, think-high and think-max: 0 of 48, 0 of 24, 0 of 24; six tool calls per mode, all
correct) is in `docs/val_abliterated.json` of their repository. Not measured here: the thinking modes, other languages, long
generations, PII and self-harm categories separately, benchmarks beyond the GSM8K subset. The samples are small: 1 of 64 has a
95 % upper bound of about 8 %, and 99 against 98 of 100 on GSM8K is not a difference.

The measured GPU memory of the card (weights 79.36 GiB per GPU, 616,712 KV cache tokens) was reproduced: the engine
logged 79.36 GiB and 616,830 tokens.

## Licenses

MIT, as DeepSeek V4 Flash. `licenses/source-0-LICENSE.txt` is the license file of the weights repository (byte-identical to DeepSeek's).
