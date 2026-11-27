# 2026-11-27 15:00 Decode-aware keep set and results

The first keep set (`2026-11-27-0845-security-corpus-prompt-ranking.md`) was ranked on prompt tokens
only and lost reasoning (GSM8K 32.8 %). Why, how generated tokens were measured, and every design choice
of that measurement are in the coder card's journal `2026-11-27-1130-decode-routing.md` (section
"Measurement design"); this entry has the cyber-specific part.

## Decode profiling and the new keep set

`cyber-gen`: 2,000 random prompts of the cyber corpus, thinking on, at most 512 generated tokens,
1,181,878 prompt and 1,022,475 generated tokens, on the abliterated card (overlay active, four GPUs, no
speculative decoding). The reasoning statistics (GSM8K train and MATH train, 753,907 generated tokens)
are the coder card's, measured on the official model.

Ranking: cyber prompt mass and cyber decode mass with equal weight, reasoning decode mass at least 95 %
of the best any 210 experts can carry, per layer.

| | keep210-cyber, decode-aware | keep210-cyber, prompt-only |
|---|---|---|
| Cyber prompt mass kept, mean / worst | 88.7 / 80.2 % | 92.1 / 85.2 % |
| Cyber decode mass kept | 93.3 / 87.8 % | 87.3 / 69.5 % |
| Reasoning decode mass kept | 92.8 / 87.7 % | 52.6 / 32.0 % |
| Coding decode mass kept | 86.5 / 82.1 % | 81.3 / 66.3 % |
| Critical reasoning experts dropped (fewest carrying 80 % of the mass) | 50 | 962 |
| Critical cyber-decode experts dropped | 11 | 325 |

## Results (two GPUs, 200k context; reference: the unpruned abliterated card on four GPUs)

| | prune210-cyber-abl, decode-aware | prompt-only | unpruned abliterated |
|---|---|---|---|
| CyberMetric-500 | **94.8 %** | 94.0 % | 95.6 % |
| CTIBench cti-mcq, first 300 | **78.0 %** | 76.3 % | 75.0 % |
| SecQA v1 test | 100 % | 100 % | 100 % |
| HumanEval pass@1 | 95.1 % | 92.1 % | 97.6 % |
| GSM8K, first 250, thinking | **96.4 %** (0 cut off) | 32.8 % (59 cut off) | 98.0 % |
| Harmful refusal, chat / think-high (64 each) | 0 % / 4.7 % | 3.1 % / 1.6 % | 6.3 % / 3.1 % |
| False refusals, 64 legitimate security questions | 0 % | 1.6 % | 0 % |
| Thinking answers to harmful prompts cut off at 4,096 tokens | 18.8 % | 51.6 % | 12.5 % |
| Tool calls | 7/8 | 8/8 | 8/8 |
| Needle, fresh, 6,859 / 54,509 / 102,521 / 170,052 tokens | 4/4 | - | - |
| Decode, one request | 121 tok/s, DSpark 4.42 | 111 tok/s, 4.49 | - |

Refusal is marker-matched on the answer (reasoning excluded); the three think-high matches are counted,
not read. The abliteration survives the pruning, so no second overlay was needed. The one failed tool
call: in thinking mode the model wrote `</think>` into the `body` argument of `send_email`, so the DSML
call did not parse; one case in eight, too few to call it a pattern; a larger tool-call test is open.
The context limit (200k) and host RAM are in the coder card's `2026-11-27-1415-context-limit.md`, and
apply unchanged: same engine, same memory share.

## Platform benchmark (`benchmarks/2026-11-27-1615Z-llmperfbench.yaml`, all three workloads)

| Workload | 1 request | Plateau |
|---|---|---|
| max-decode | 91 tok/s | 360 tok/s at 12 requests (up to 365) |
| max-prefill | 1,582 tok/s | 2,107 tok/s at 24 requests |
| mixed-load | 62 tok/s decode, first token 0.44 s | 186 tok/s decode at 8 (first token 2.7 s) |

No request failed at any level; the same speed as the coder variant (same engine, drafter and memory
share). Why prompts above about 213k tokens crash the engine (GPU memory fills during the long
prefill) is in the coder card's `2026-11-27-1415-context-limit.md`; this card is set to 200k as well.

