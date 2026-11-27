# 2026-11-27 08:30 Expert ranking from prompt routing (first keep set)

**Goal:** a 2-GPU DeepSeek V4.1 Flash 1017 for coding: 210 of 384 routed experts per layer ranked on
coding traffic instead of the keep224r2 set of the prune224 card, all 128 DSpark drafter experts, 8
requests, a KV pool of about 2M tokens. The cyber variant
(`deepseek-v4-1-flash-1017-prune210-cyber-abl`) was built the same way; its journal has the security
corpus.

## Routing counter

`container-files/abl41.py` with `ABL_ROUTING=1` (on a temporary 4-GPU profiling card, the official
1017 card plus the counter) hooks every target layer's router (`mlp.gate`, its output are the router
logits) and recomputes the text-token routing the layer uses: sqrtsoftplus scores plus the gate's
correction bias, top 6, weights renormalized. Per layer and expert it sums the routed-token count and
the gate weight (count, sum, sum of squares, max) in GPU buffers.

- A hook on SGLang's `TopK` counted nothing: V4.1 routes through `vision_topk` (its gate has an
  image-token bias) and fused decode-graph kernels, and the flashinfer MXFP4 backend returns no expert
  ids. The router call is the one every path shares.
- Layers 0-20 see about 11 % more tokens than layers 21-39 (986 against 885 for 888 prompt tokens):
  layer 20 is V4.1's `candidate_source_layer_id`. Consistent, so the ranking per layer is unaffected.
- **This first counter counted prompt tokens only** (one answer token per request): it measured which
  experts read coding text, not which ones the model uses while writing. See
  `2026-11-27-1130-decode-routing.md` for why that matters and what changed.

## Data and the first keep set

8,000 of the 10,000 prompts of `onprem-ai--coding-prompts--v1` (80/20 split per kind, seed 1; the other
2,000 held out), 3,336,608 prompt tokens, 25 minutes on four GPUs. No other corpus. Per layer the 210
experts with the most gate-weight mass (`weight_sum`):

| Keep set | Experts | Coding routing mass kept, mean / worst layer |
|---|---|---|
| keep210-coder (prompt) | 210 | 90.4 % / 83.4 % |
| keep224r2 (prune224) | 224 | 68.9 % / 60.9 % |

On average 147 of the 210 experts are also in keep224r2.

## First results (2 GPUs, against the unpruned 1017 on 4 GPUs)

| | prune210-coder (prompt ranking) | unpruned 1017 |
|---|---|---|
| HumanEval pass@1 (164) | 96.3 % | 96.95 % |
| **GSM8K, first 250, thinking** | **48.8 %** (106 wrong numbers, 22 cut off at 8,192 tokens) | 98.0 % |
| Tool calls | 8/8 | 8/8 |
| Needle, 8k-120k, 3 depths | 12/12 | 11/12 |
| Decode, one request, 512 tokens | 125 tok/s, DSpark 4.51 tokens per step | 146 tok/s (4 GPUs), 4.60 |

Code completion held, arithmetic and step-by-step reasoning collapsed: the answers were not too long
(median 163 tokens against 133) but wrong. The cyber variant fell further (32.8 %). The ranking kept the
experts that read code and dropped experts the model needs while it reasons.
