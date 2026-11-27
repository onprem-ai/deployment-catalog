# DeepSeek V4.1 Flash 1017 Abliterated (SGLang, disk Engram, 4x GPU)

> **Uncensored model.** Safety refusals have been substantially removed; it will attempt almost any
> request the original refuses. Deploy it only where that is intended.

This card is the official `deployment-catalog` card
[`nvidia-rtx-pro-6000-blackwell/deepseek-v4-1-flash-1017-sglang-disk-engram`](https://github.com/onprem-ai/deployment-catalog/tree/main/nvidia-rtx-pro-6000-blackwell/deepseek-v4-1-flash-1017-sglang-disk-engram)
with the refusal direction removed from the attention output projections. Image, engine command,
environment, resources and DeepSeek's checkpoint are the official card's (the same weight files, so
an inference node that has the official model cached reuses it); the differences are:

- the served model name `deepseek-v4.1-flash-abliterated`, so it can run next to the official model;
- `--kv-cache-dtype fp8_e4m3` (as the official card's open change), and the measured GPU memory;
- a second weight source, the overlay below, and `container-files/abl41.py`, which loads it.

## Weights

1. **`main`**: DeepSeek's [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
   at `1bae0e1088993b9b82de08f59de5be5a34e42170`, unchanged.
2. **`overlay`**: OnPrem AI model storage `onprem-ai--DeepSeek-V4.1-Flash-1017-abl`, 1.8 GB: one file
   `abliterated-wo_b.safetensors` with **86 tensors**, the `wo_b` weight (F8_E4M3, 5120 x 8192) and its
   block scale (F8_E8M0, 160 x 256) of the 40 layers and the 3 DSpark drafter stages, with DeepSeek's names,
   dtypes and shapes; DeepSeek's `LICENSE` and `README.md`; `README-abliteration.md`;
   `abliteration_report.json` and `verify_report.json`. A safetensors file carries tensors, not code.

While SGLang loads DeepSeek's checkpoint, `abl41.py` (installed by `container-files/sitecustomize.py`,
which also keeps the image's own disk-Engram hook) replaces exactly those 86 tensors by name and checks
their dtype and shape against the checkpoint's. The other 95,999 tensors come from DeepSeek's files.

**The edit.** Refusal-direction abliteration (Arditi et al. 2024): per layer `W' = W - lambda * r (r^T W)`
on `wo_b`, `r` the normalized difference of the mean `wo_b` output at the last prompt token over 256
AdvBench harmful and 256 Alpaca harmless prompts, chat and thinking directions merged with equal weight;
the drafter stages get the layer-39 direction (an unedited drafter would propose refusal tokens the edited
model rejects). The edited projections stay FP8 with DeepSeek's own block scales: an element the edit
does not move keeps its byte, and a block's scale is raised only where the edit overflows E4M3 (15,303 of
1,761,280 blocks). Baked at lambda 2.125 for an effective 2.0 after rounding (median over the 43
projections 1.999). The pipeline is `dev/abliteration/` in onprem-ai/platform-alpha.

## Validation

Held-out prompts (never used for the directions), this card's engine (TP=4, DSpark, CUDA graphs, FP8 KV
cache), 2026-11-26. Base column: the official card on the same four GPUs (same image and command, BF16 KV
cache, prefix caching off for the direction capture), unedited.

| | DeepSeek's model | This model |
|---|---|---|
| Harmful refusal, chat (AdvBench; base n=32, this model n=64) | 100 % | 4.7 %¹ |
| Harmful refusal, thinking high / max (base n=32, this model n=64 each) | 100 % / - | 3.1 % / 4.7 %¹ |
| Benign refusal, chat / thinking (Alpaca; base n=16, this model n=32 each) | 6.3 %² / - | 0 % / 0 % |
| Broken or empty answers (not a token-limit cut-off) | 0 % | 0 % |
| GSM8K, first 250, greedy, thinking (high) | 98.4 % | **98.4 %** |
| Needle (6-digit code in prose) at 7k / 27k / 55k / 102k tokens, 3 depths each, twice | - | **23 / 24**³ |
| Tool calls (DSML, 3 tools), chat and thinking | 8 / 8 | **8 / 8** |
| DSpark tokens accepted per step (GSM8K) | 3.74 | 3.66 |

¹ Marker matches on the answer (reasoning excluded); none of the 8 is a plain refusal. 4 answer in full
(3 contain a marker phrase in the text, 1 opens with "I can't render a video file, but here is ..."); 4 answer
a harmless variant of the request instead (a positive video concept, a different target group, a "simulated"
training scenario, a "novelty" ID): 4 of 192 harmful requests, 2.1 %, are deflected. Sample sizes are
small: 0 of 64 has a 95 % upper bound of about 6 %. ² A story that contains a marker phrase. ³ The miss returned the right six digits in a
different order (`273605` for `605273`, 102k tokens, depth 10 %); the same prompt was answered correctly on
the repeat.

Re-checked with the published overlay (start, harmful and benign refusal probe, tool calls): the same
results, see the journal.

Not measured: categories (self-harm, PII) separately, benchmarks beyond the GSM8K subset, image input (the
card serves text).

## Licenses

MIT, DeepSeek's (`licenses/source-0-LICENSE.txt`); the overlay entry carries DeepSeek's `LICENSE`.
