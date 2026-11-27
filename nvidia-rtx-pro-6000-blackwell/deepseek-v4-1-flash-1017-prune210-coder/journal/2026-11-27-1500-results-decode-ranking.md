# 2026-11-27 15:00 Results with the decode-aware keep set

The keep set from `2026-11-27-1130-decode-routing.md` (coding prompt and decode mass, reasoning floor
0.95), measured with that entry's design ("Measurement design"). Two GPUs, 200k context (see
`2026-11-27-1415-context-limit.md`), against the unpruned 1017 on four GPUs and the first, prompt-only
keep set (`2026-11-27-0830-prompt-routing-ranking.md`). Evaluation sets are disjoint from every ranking
corpus.

| | prune210-coder, decode-aware | prune210-coder, prompt-only | unpruned 1017 (4 GPUs) |
|---|---|---|---|
| HumanEval pass@1 (164, greedy, chat; tests run without network) | **97.6 %** | 96.3 % | 96.95 % |
| GSM8K, first 250 of test, thinking | **97.6 %** (0 cut off) | 48.8 % (22 cut off) | 98.0 % |
| Tool calls (3 tools, chat and thinking) | 8/8 | 8/8 | 8/8 |
| Needle, every request fresh (0 cached tokens): 6,935 / 54,461 / 102,472 / 170,476 tokens | 4/4 | - | - |
| Decode, one request, 512 tokens, incl. prefill | 124 tok/s | 125 tok/s | 146 tok/s |
| DSpark accepted length (tokens per step) | 4.49 coding, 3.75 GSM8K | 4.51, 3.72 | 4.60, 3.73 |

The decode-aware set restores reasoning (GSM8K within 0.4 points of the unpruned model) and slightly
improves code (HumanEval 97.6 %): writing code needed the decode experts the prompt-only set had dropped
too. Speed and drafter acceptance are unchanged by the new ranking; the full 128-expert drafter accepts
nearly as many tokens as the unpruned model's.

## Platform benchmark (`benchmarks/2026-11-27-1342Z-llmperfbench.yaml`, all three workloads)

| Workload | 1 request | Plateau | prune224 (keep224r2, 64-expert drafter, 4 requests) |
|---|---|---|---|
| max-decode | 92 tok/s | 317 tok/s at 8 requests (up to 350 at 24) | 76 tok/s, 214 tok/s at 4 |
| max-prefill | 1,505 tok/s | about 2,000-2,100 tok/s from 20 requests | 1,489 tok/s, about 1,850-1,900 |
| mixed-load | 64 tok/s decode, first token 0.46 s | 185 tok/s decode at 8 (first token 2.4 s) | - |

No request failed at any level. The full drafter and 8 requests raise single-request decode by a fifth
and the plateau by half against prune224. Prefill stays bound by the 2,048-token chunks. The control
plane lost its connection to the node many times during this run (it stopped answering for 9 s at a
time), which did not affect the measurement on the node.

Still open: a correctness benchmark beyond HumanEval (for example LiveCodeBench) and the coding-greedy
workload's acceptance on the held-out prompts.
