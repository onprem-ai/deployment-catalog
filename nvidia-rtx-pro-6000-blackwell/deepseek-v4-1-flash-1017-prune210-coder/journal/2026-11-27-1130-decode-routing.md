# 2026-11-27 11:30 Decode routing: ranking on what the model uses while it generates

**Why:** the first keep sets (`2026-11-27-0830-prompt-routing-ranking.md`) were ranked on prompt
tokens only. They measured which experts *read* code or security text, not which ones the model uses
while it *generates*: thinking, step-by-step reasoning, writing code. Both pruned cards lost reasoning
(GSM8K 98 % -> 48.8 % coder, 32.8 % cyber). This entry adds the generated tokens to the measurement.
It applies to both 210 cards; the cyber card's journal links here.

## Measurement design

Each choice and the reason for it:

- **Prompt and decode tokens counted separately.** The counter keeps two sets of buffers per layer. An
  eager forward (prefill) writes the prompt buffers; a forward SGLang captures into a CUDA graph
  (decode) writes the decode buffers, and every graph replay counts there. Reason: the two phases use
  experts differently, and the keep set should weigh what generation needs without losing what reading
  needs.
- **DSpark off in the profiling runs.** With speculative decoding each verify pass routes the drafted
  tokens through the main model, rejected ones included, so a generated token would be counted several
  times and draft tokens the model never emits would add mass. Without it every generated token passes
  the main model exactly once. The drafter's own experts are not ranked at all (it keeps all 128).
- **A CUDA graph for every batch size 1-16** (`--cuda-graph-bs 1 ... 16`, `--max-running-requests 16`).
  SGLang's default graph sizes (1, 2, 4, 8, 12, 16) pad a batch of 3, 5, 6, 7, ... up to the next size;
  the router does not know which rows are padding, so padded rows would be counted. With a graph per
  size no batch is padded.
- **Thinking on** (DeepSeek's encoding, `reasoning_effort` high, sent to SGLang's `/generate`). Reason:
  the failure was in reasoning, and thinking tokens are where the model reasons; a chat-mode answer would
  under-represent them.
- **Reasoning prompts from training splits only:** GSM8K train (1,000) and MATH train (1,000, all seven
  subjects, `EleutherAI/hendrycks_math` `21a5633873b6`, MIT). The evaluations use GSM8K test, HumanEval,
  CyberMetric, CTIBench and SecQA; none of those items is in any profiling corpus. The coding evaluation
  prompts (2,000 of coding-prompts-v1) are held out of every ranking corpus as well.
- **Answers are not stored.** The driver keeps only token counts; the routing statistics are per-expert
  sums. Nothing the model generated (also not for security prompts) is written anywhere.
- **The profiling model matches the product.** Coder statistics come from the official 1017 card, cyber
  statistics from the abliterated card (overlay active), because the abliteration changes the attention
  outputs and so, slightly, the routing. The reasoning statistics are measured on the official model and
  used for both (their purpose is to protect reasoning experts; the routing shift from the overlay is
  small next to that).
- **Rank 0's statistics, not a sum.** Under tensor parallelism every rank computes the router for the
  same tokens; summing over ranks would only rescale. Checked on this run (24 thinking requests, 2,446
  prompt and 1,536 generated tokens): ranks 0 and 1 route every token to the same experts (0.000 % of
  routing slots differ, prompt and decode); the gate-weight sums agree up to float rounding of the
  all-reduced logits.
- **Counts verified against the request totals.** In that check the decode buffers held 1,537 tokens
  per layer for 1,536 generated tokens, and the prompt buffers 2,448 for 2,446 prompt tokens, so every
  generated token is counted once and nothing else is.
- **Sampling** temperature 0.6, top-p 0.95, at most 768 generated tokens for reasoning and 512 for coding
  and security: realistic generations without spending hours per corpus; the first tokens of an answer
  and the thinking before it are where the reasoning experts work.
- **Unchanged from the first run:** routing recomputed from the router logits (sqrtsoftplus plus the
  text-token correction bias, top 6, renormalized), `weight_sum` (the gate-weight mass an expert
  carries) as the criterion, prompts from pinned public sources only.

## Runs

Four GPUs, thinking on, 16 requests at a time, no speculative decoding:

| Suite | Model | Prompts | Prompt tokens | Generated tokens | Minutes |
|---|---|---|---|---|---|
| reasoning (GSM8K train 1,000, MATH train 1,000) | official 1017 | 2,000 | 215,560 | 753,907 | 30 |
| coding-gen (random 2,000 of coding-rank) | official 1017 | 2,000 | 892,905 | 1,009,744 | 45 |
| cyber-gen (random 2,000 of the cyber corpus) | 1017 abliterated | 2,000 | 1,181,878 | 1,022,475 | 45 |

## What the first keep sets dropped

"Critical" experts of a corpus: per layer, the fewest experts that carry 80 % of its routing mass.
Shares are absolute (of all 384 experts' mass).

| Keep set | Reasoning decode mass kept, mean / worst | Critical reasoning experts dropped | Coding decode mass kept | Cyber decode mass kept |
|---|---|---|---|---|
| keep210-coder (prompt only) | 66.9 % / 38.3 % (layer 22) | 804 | 82.4 % / 65.6 % | - |
| keep210-cyber (prompt only) | 52.6 % / 32.0 % (layer 21) | 962 | 81.3 % / 66.3 % | 87.3 % / 69.5 % |
| keep224r2 (prune224) | 93.3 % / 85.2 % | 129 | (68.9 % prompt mass) | - |

The prompt-only sets dropped hundreds of the experts the model reasons with, concentrated in layers
15-33 (in layer 22 the coder set kept 38 % of the reasoning decode mass). They also lost part of the
experts that *write* code (coding decode 82 % against 90 % prompt mass). The reasoning experts are
concentrated: in some layers 50-90 experts carry 80 % of the reasoning decode mass, so keeping them is
cheap. keep224r2 (four corpora, math among them) kept them by accident and lost coding mass instead.

## New keep sets

Objective per layer: the domain's prompt mass and decode mass with equal weight; a floor: the kept
reasoning decode mass at least 95 % of the best any 210 experts can carry for it (0/1 program per layer,
`build_keep_set.py`). The cost of the floor, relative to the best 210 experts per corpus (coder):

| Floor (reasoning decode) | Coding prompt mean / worst | Coding decode mean / worst | Reasoning decode mean / worst |
|---|---|---|---|
| none | 98.5 / 95.0 % | 98.9 / 97.4 % | 74.7 / 60.0 % |
| 0.90 | 98.0 / 93.8 % | 98.6 / 97.3 % | 90.4 / 90.0 % |
| **0.95 (chosen)** | **96.4 / 90.3 %** | **97.8 / 95.8 %** | **95.0 / 95.0 %** |
| 0.97 | 93.5 / 85.7 % | 96.3 / 92.8 % | 97.0 / 97.0 % |

Without a floor even the decode-aware ranking keeps only 75 % of the reasoning mass: writing code uses
other experts than step-by-step math. 0.95 costs 1-2 points of coding mass; 0.97 costs 5 points of
prompt mass. The cyber set (cyber prompt + cyber decode, the same floor) relative: cyber prompt 96.3 /
93.3 %, cyber decode 97.8 / 96.5 %, reasoning 95.0 / 95.0 %.

Absolute shares of the chosen sets:

| | keep210-coder (new) | keep210-cyber (new) |
|---|---|---|
| Domain prompt mass, mean / worst | 87.1 / 78.0 % (coding) | 88.7 / 80.2 % (cyber) |
| Domain decode mass | 90.3 / 85.9 % (coding) | 93.3 / 87.8 % (cyber) |
| Reasoning decode mass | 92.8 / 87.7 % | 92.8 / 87.7 % |
| Critical reasoning experts dropped | 29 | 50 |
| Critical domain-decode experts dropped | 70 | 11 |

Results with these sets: `2026-11-27-1500-results-decode-ranking.md`.
