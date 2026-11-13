# 2026-11-13 - Without Python Files, Tuning and Full Benchmark on Two RTX PRO 6000

**Goal:** Run the card without the repository's Python files, make it faster with one
change per round, check its answers, and run the full benchmark once on the chosen
settings. The card had no benchmark yet.

## Setup

- GPUs: two NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, SM120), tensor
  parallel 2, inference node `ain-xtg6o`, driver 595.112.07, CUDA 13.2.
- Image `onpremai/vllm-ds4f:0731-prebaked-v2` as pinned (vLLM build
  `0.11.2.dev280+gilded.gnosis.v20...20260731`).
- Served as `deepseek-v4-flash-tuning` during the session.
- Precision: the checkpoint's FP4 experts and FP8 block weights, as published; KV cache
  FP8 (`--kv-cache-dtype fp8`); the DSpark drafter from the checkpoint.
- Probe as in the Qwen3.8 27B journal of 2026-11-12: unique prompts of about 240 tokens,
  512-token answers with `ignore_eos`, through the API gateway's replica path, sent
  without `chat_template_kwargs`, which is the model's non-think mode.

## Without the Python files

The weights listed `encoding/encoding_dsv4.py` and its compiled
`encoding/__pycache__/encoding_dsv4.cpython-312.pyc`, and the command named
`--trust-remote-code`. `config.json` has no `auto_map` (the architecture
`DeepseekV4ForCausalLM` is built into vLLM), and `--tokenizer-mode deepseek_v4` uses
vLLM's own copy of DeepSeek's encoder. Both files left the list and the flag left the
command (round 0); the engine started and answered without them. The checker rule
`model.weights-executable-files` has nothing to report.

## The engine booking

Round 0 failed at start: without an engine booking the deploy wrote vLLM's share from the
94 GiB hardware booking, 0.9897, and vLLM refused it ("Free memory on device cuda:0
(93.97/94.97 GiB) on startup is less than desired GPU memory utilization (0.9897, 93.99
GiB)", error report `01a1d4ca-7a58-77cb-a138-6eb9aad5b296`). The engine booking is 93 GiB
(0.9792), the 0.98 the card's command named.

## Answers

Through the API gateway, on round 0b and on the chosen settings: four short questions
with known answers three times in non-think mode and once at each reasoning effort
(`high`, `max`, as `thinking: true` with `reasoning_effort`); two parallel tool calls and
an answer built from their results; a hidden number at 6,745, 28,480, 115,443 and 216,229
prompt tokens. 23 of 26 both times. The three failures are the same question in
non-think mode, "How many prime numbers are there below 30? Answer with the number
only": 11 or 5 instead of 10, also at temperature 0. Asked to list them first, or at
`high` and `max`, the model answers 10. It is the model without room to count, not a
setting. Image input and `/v1/messages` are not part of this card.

## Rounds

| Round | Change | c=1 | ITL c=1 | c=8 | c=16 | c=32 | Acceptance |
|---|---|---|---|---|---|---|---|
| 0b | baseline: DSpark 5 probabilistic, 8 sequences, graphs to 48 tokens | 128-130 | 7.7 ms | 481 | | | 2.16-2.20 |
| 1 | 32 sequences | 130 | 7.7 ms | 478 | 478 | 999 | 2.19-2.33 |
| 2 | CUDA graphs to 192 tokens (32 requests) | 126 | 8.0 ms | 472 | 695 | 1,048 | 2.13-2.23 |
| 3 | DSpark 7 draft tokens (graphs in multiples of 8) | 119-123 | 8.2-8.4 ms | 433 | 578 | 937 | 2.16-2.21 |
| 4 | DSpark 3 draft tokens | refused at start | | | | | |
| 5 | greedy draft sampling | 120-124 | 8.0-8.3 ms | 468 | 628 | 992 | 2.06-2.18 |
| 6 | 64 sequences (64 requests: 991) | 131 | 7.7 ms | | | 986 | 2.21-2.24 |

Prefill at 1 request, 5,063-token prompts (round 0b): 0.57 s to the first token (8,877
tokens/s).

## What the rounds show

- The card ran 8 requests at once, with the KV cache 3.4 percent used: 32 sequences
  (round 1) are needed for more than one small team.
- The card's fixed CUDA graph sizes ended at 48 tokens, 8 requests of 6 tokens: with 16
  requests the replica ran no faster than with 8. Graph sizes up to 192 tokens (round 2)
  give 45 percent more at 16 and 5 percent more at 32.
- DSpark's draft length: DeepSeek and vLLM name 7 draft tokens, but on these prompts the
  acceptance stays at 2.2 tokens, so 7 tokens lost 4 to 17 percent (round 3). Fewer than
  5 is refused ("DSpark requires num_speculative_tokens >= dspark_block_size (5) ...
  Smaller values produce incorrect output"). 5 stays.
- Greedy draft sampling, which the model card names, lost up to 10 percent against
  probabilistic (round 5); vLLM's recipe uses probabilistic for chat at temperature 0.6
  to 1.0.
- 64 requests at once give the same throughput as 32 (round 6): the GPUs, not the KV
  cache (29 percent used), are the limit.

## Chosen

Round 2: DSpark with 5 draft tokens and probabilistic draft sampling, 32 sequences, CUDA
graphs up to 192 tokens, FP8 KV cache, 262,144 tokens of context, 93 GiB engine booking,
no Python files and no `--trust-remote-code`. Against the card as it was: about the same
single-user speed (126-130 tokens/s), 32 requests at once instead of 8, and 1,048 against
481 tokens/s of throughput (2.2 times).

## The full benchmark

Job `ecbb42fa-fc2b-4fc4-a7a6-29e07441452f` on replica `1de62c58-3518-4f9e-8deb-0b4b5c5133e2`,
all three workloads, 12:59:24 to 13:49:22Z, on the committed settings (`compose_sha256`
`55363d6d...`). It succeeded: every workload ended at its plateau and no request failed.

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 16 | 6,864 prefill tokens/s at 1 (TTFT p50 0.82 s for about 7,900 tokens); 10,150 at 16 |
| `max-decode/1.0.0` | 32 | 199 decode tokens/s at 1 (ITL p50 4.9 ms); 977 at 16; 1,418 at 32 (TTFT p50 0.25 s) |
| `mixed-load/1.0.0` | 48 | 148 decode tokens/s at 1; at 32: 4,610 prefill and 875 decode tokens/s |

Above 32 requests the time to first token rises (5.2 s at 40 in `max-decode`): 32 is the
replica's limit, as configured.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
101,651,054,592 bytes (94.67 GiB) on each GPU. It was above the card's 94 GiB hardware
booking, so the hardware booking is 95 GiB (0.33 GiB of headroom; the GPU shows 94.97 GiB
to CUDA). The overshoot over the 93 GiB engine booking is 1.67 GiB.

| Part | Bytes | GiB | Source (TP0's log) |
|---|---|---|---|
| `model_weights` | 85,212,151,152 | 79.36 | `Model loading took 79.36 GiB` (the DSpark drafter is in it) |
| `cuda_graphs` | 257,698,037 | 0.24 | `Graph capturing finished ... took 0.24 GiB` |
| `kv_cache` | 9,910,637,035 | 9.23 | `Available KV cache memory: 9.23 GiB` |
| `runtime_overhead` | 6,270,568,368 | 5.84 | the peak minus every other part |

`GPU KV cache size: 539,477 tokens` (2.06 requests of 262,144 tokens). The container's
host memory peaked at 8.4 GiB, below the profile's 20 GiB.

## Next

- Single-user speed depends on DSpark's acceptance, 2.2 tokens on the probe's random
  word lists; DeepSeek reports more on real chat and code. A probe on real prompts would
  show whether 7 draft tokens win there.
- vLLM's recipe for 8 RTX PRO 6000 uses a newer vLLM (nightly or with vLLM #51538) and a
  FlashInfer build with flashinfer-ai/flashinfer#4380 for DSpark on SM120; the pinned image
  is from 2026-07-31. A newer image is worth a round.
- The reasoning effort mapping issue in this image (journal 2026-08-08: `low` and `high`
  collide) was not retested here.
