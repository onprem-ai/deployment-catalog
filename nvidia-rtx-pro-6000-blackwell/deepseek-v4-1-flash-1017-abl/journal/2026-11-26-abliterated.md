# 2026-11-26 Abliterated weights on the official four-GPU card

**Goal:** the official `deepseek-v4-1-flash-1017-sglang-disk-engram` card with the refusal direction
removed, published as an overlay of the 86 edited tensors, so DeepSeek's 510 GB checkpoint (and every
node's cache of it) stays as it is. The pipeline follows `dev/abliteration/` (onprem-ai/platform-alpha):
the DeepSeek V4 Flash pipeline, adapted to V4.1 1017 and to SGLang.

## Capture

A temporary card: the official card plus `container-files/abl41.py` (hooks on every `wo_b`, control routes
on SGLang's HTTP server) and `--disable-radix-cache`, on all four GPUs. SGLang builds `wo_b` with
`reduce_results=True` when attention runs tensor-parallel over all ranks, so a forward hook sees the
all-reduced output: the four ranks recorded identical vectors (equal SHA-256 digests per suite).
Prompts were built with DeepSeek's own `encoding/encoding.py` and sent to SGLang's `/generate` with
`max_new_tokens` 1 (the last prompt token is the decision point). 256 AdvBench harmful and 256 Alpaca
harmless prompts (seeded permutation, AdvBench `a62d1307`, Alpaca `f1349621`) in three modes: chat,
thinking with reasoning effort high (75) and max (100); about 47 s per suite.

Per layer: `r = mean(harmful) - mean(harmless)`, normalized.

| Mode | Split-half cosine, median / min | Held-out AUC, median / min | Dominance, median / max |
|---|---|---|---|
| chat | 0.986 / 0.943 | 0.9985 / 0.985 | 0.244 / 0.431 |
| think-high | 0.990 / 0.976 | 0.9996 / 0.969 | 0.227 / 0.445 |
| think-max | 0.989 / 0.973 | 0.9987 / 0.979 | 0.227 / 0.429 |

Dominance (`mean |r . y| / mean |y|`) has the median of Qwen3.8 (0.23), where the method worked; single
layers reach 0.43 (layers 11, 16-19, 23-29, 31-33), far from Apertus, where 142 of 160 components were
above 0.25 and abliteration failed. Chat and thinking directions differ in the deep layers (cosine median
0.58, layers 30-39 0.12-0.19); think-high and think-max agree (0.88). Merged for the bake:
`r = normalize(r_chat + normalize(r_think_high + r_think_max))`.

## Lambda sweep (runtime hook, held-out prompts)

The same hooks apply `y <- y - lambda r (r^T y)` from GPU buffers, which SGLang's CUDA graphs record, so
lambda changes without a restart. Harmful n=32 per mode, benign n=16, chat 1024 / thinking 4096 tokens.

| lambda | harmful chat | harmful think-high | benign chat |
|---|---|---|---|
| 0 | 100 % | 100 % | 6.3 % (a story with a marker phrase) |
| 1.5 | 3.1 % | 6.3 % | 0 % |
| **2.0** | **0 %** | **3.1 %** | **0 %** |
| 2.5 | 6.3 % | 0 % | 0 % |

The hits at 2.0 and 2.5 are "I can't render a video file, but here is ..." with a full answer. Chosen:
effective lambda 2.0, as for Qwen3.8.

## Bake

`W' = W - lambda r (r^T W)` in float64 on each `wo_b` (axis 0, the 5120 outputs), re-quantized to F8_E4M3 with
the checkpoint's E8M0 block scales; V4.1 uses 32 x 32 blocks (V4 used 128 x 128). An element whose value
does not move keeps its byte (this also keeps `-0.0`, which a plain cast turns into `+0.0`); a block's scale
is raised only where the edit overflows. Checked on layer 0: lambda 0 reproduces every byte of weight and
scale; a unit direction changes only its own row. Rounding keeps about 93 % of the edit, so the bake uses
lambda 2.125: effective lambda (the measured reduction of `r^T W`) median 1.999, minimum 1.854 (`mtp.0`);
15,303 of 1,761,280 blocks raised; relative edit size median 0.066, largest 0.22 (`mtp.1`), 0.17 (layer 34).
The engine's load-time bake (`ABL_BAKE_DIRECTIONS`, used for the validation) produced, on every rank, the
same SHA-256 for all 43 projections as the overlay file.

## Validation (load-time bake = overlay bytes)

Harmful refusal (n=64 each): chat 4.7 %, think-high 3.1 %, think-max 4.7 % by marker; none of the 8 is a
plain refusal: 4 full answers (marker phrase in the text or "I can't render a video file, but ..."), 4
harmless variants of the request (2.1 % of 192). Benign refusal 0 % chat and thinking (n=32 each). GSM8K
first 250, think-high: 98.4 % (base 98.4 %), 1 answer cut off at 8192 tokens. Tool calls 8/8 (base 8/8).
Needle at 7k-102k tokens, 3 depths, twice: 23/24 (one digit-order slip at 102k, correct on the repeat).
DSpark: 3.66 tokens per step on GSM8K (base 3.74).

GPU memory per GPU: weights 74.56 GiB (72.51 target + 2.05 drafter), KV pool 2,935,552 tokens in 5.08 GiB,
CUDA graphs 0.27 GiB, 86,221 MiB in use after the 102k-token requests (`nvidia-smi`): the official
model's figures, since the edit keeps every tensor's format. The load-time bake took the weight load from
about 3 to 8 minutes (CPU work on all four ranks); the overlay replaces it with a plain tensor swap.
