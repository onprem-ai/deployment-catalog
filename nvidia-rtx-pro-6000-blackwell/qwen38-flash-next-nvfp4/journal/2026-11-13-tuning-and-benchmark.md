# 2026-11-13 - Checker Findings, Tuning and Full Benchmark on Two RTX PRO 6000

**Goal:** Clear the two findings of the deploy profile checker, try the levers that
promise more speed, check the answers, and run the full benchmark once on the chosen
settings.

## Setup

- GPUs: two NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, SM120), tensor
  parallel 2, inference node `ain-xtg6o`, driver 595.112.07, CUDA 13.2.
- Image `lmsysorg/sglang:qwen38flashnext` as pinned, with the card's SM120 QSA guard
  patch.
- Served as `qwen3.8-flash-next-tuning` during the session.
- Precision: NVFP4 routed experts and FP8 PLE n-gram table, as published; KV cache
  BF16 (see below); recurrent state BF16; MTP (NEXTN 3/1/4) from the checkpoint.
- Probe as in the Qwen3.8 27B journal of 2026-11-12: unique prompts of about 260 tokens,
  512-token answers with `ignore_eos`, reasoning off, through the API gateway's replica
  path.

## The checker findings

- `engine.kv-cache-size-fixed`: `--max-total-tokens 262144` capped the KV cache at one
  full context, below what the engine's share of GPU memory holds. Removed.
- `engine.kv-cache-dtype`: the command did not state the KV cache's precision.
  `--kv-cache-dtype fp8_e4m3` (round 0) crashed the engine on the turn that sends two
  tool results back: HTTP 500 and a replica restart (error report
  `01a1d3d3-47e7-70b6-ae43-35bbdfb5780e`). The published compose answered that turn
  (round 0x); adding only the FP8 KV cache to it crashed it again (round 0y). The
  command therefore states `--kv-cache-dtype bfloat16`. The crash itself is not
  explained: the replica's log after start-up is not kept, and the followed live log
  was not delivered (see "Next").
- `runtime.measured-gpu-memory-missing`: measured below.

## Answers

Through the API gateway, on rounds 1 and 4: four short questions with known answers
three times with reasoning off and once at each effort (`low`, `medium`, `xhigh`); two
parallel tool calls and an answer built from their results; an image (two colors in
order); a hidden number at 6,757, 28,492, 115,455 and 231,398 prompt tokens. 31 of 31 on
round 1. On round 4, 29 of 31: the model called the tool for Paris only, once; five
repeats of the same turn afterwards called it for both cities and answered from the
results. This is the model's sampling at temperature 0.7, not a setting.

## Rounds

| Round | Change | KV tokens | c=1 | ITL c=1 | c=16 | c=32 | c=48 / 64 |
|---|---|---|---|---|---|---|---|
| 0 | FP8 KV cache, no token cap | 1,748,096 | crashed on a tool-result turn | | | | |
| 0y | published compose + FP8 KV cache only | | crashed on a tool-result turn | | | | |
| 1 | BF16 KV cache stated, no token cap | 925,376 | 172-174 | 5.8 ms | 1,058 | 1,510 | |
| 2 | default `--mamba-track-interval` (256) | 925,376 | 175 | 5.7 ms | 1,069 | 1,499 | |
| 3 | 64 requests (`--max-mamba-cache-size 384`, graphs to 64) | 259,776 | 173 | 5.8 ms | | 1,492 | 1,916 at 64 |
| 4 | 48 requests (`--max-mamba-cache-size 288`, graphs to 48) | 592,576 | 172 | 5.8 ms | | 1,473 | 1,738 at 48 |

Prefill at 1 request, 5,327-token prompts (round 1): 0.495 s to the first token
(10,754 tokens/s).

## What the rounds show

- Without the token cap the KV cache holds 3.5 times more (925,376 tokens against
  262,144) at the same speed.
- The track interval changes nothing here: the state pool is pinned by
  `--max-mamba-cache-size`, so the KV cache did not grow.
- Each running request reserves five recurrent-state slots (`extra_buffer`). More
  requests at once take memory from the KV cache: 64 requests left 259,776 tokens, less
  than one full context, for 28 percent more throughput; 48 requests keep 592,576 tokens
  (2.3 full contexts) for 15 percent more.

## Chosen

Round 4: 48 requests at once, `--max-mamba-cache-size 288`, CUDA graphs to 48, BF16 KV
cache stated, no token cap, default track interval. Against the card as it was (32
requests, a KV cache capped at 262,144 tokens): about the same single-user speed, 15
percent more throughput at 48 requests, and 2.3 times the KV cache.

## The full benchmark

Job `87eecb13-18bb-483b-8248-40b9399357e8` on replica `0ee2f580-7d99-46e4-b97d-4ef3640b701f`,
all three workloads, 08:22:48 to 09:06:29Z, on the committed settings (`compose_sha256`
`14e13937...`). It succeeded: every workload ended at its plateau and no request failed.

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 12 | 8,310 prefill tokens/s at 1 (TTFT p50 0.62 s for about 7,900 tokens); 12,682 at 32 |
| `max-decode/1.0.0` | 48 | 208 decode tokens/s at 1 (ITL p50 4.6 ms); 1,833 at 32; 2,090 at 48 (TTFT p50 0.19 s) |
| `mixed-load/1.0.0` | 40 | 183 decode tokens/s at 1; at 48: 6,129 prefill and 1,053 decode tokens/s |

Above 48 requests the time to first token rises (5.5 s at 64): 48 is the replica's limit,
as configured.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
99,637,788,672 bytes (92.79 GiB) on each GPU. It was above the card's 88 GiB hardware
booking, so the hardware booking is 94 GiB (1.21 GiB of headroom). The engine booking is
88 GiB, what `--mem-fraction-static 0.92` takes of an empty GPU; the overshoot over it is
4.79 GiB.

| Part | Bytes | GiB | Source (TP0's log) |
|---|---|---|---|
| `model_weights` | 69,814,693,396 | 65.02 | `Load weight end ... mem usage=65.02 GB` (target) |
| `drafters` | 128,849,018 | 0.12 | `Load weight end ... Qwen4ExpForCausalLMMTP ... mem usage=0.12 GB` |
| `cuda_graphs` | 2,587,717,795 | 2.41 | `Capture ... CUDA graph end`: target verify 0.89, draft decode 0.71, draft extend 0.81 GB |
| `kv_cache` | 7,881,264,988 | 7.34 | `KV Cache is allocated`: K and V 3.39 GB each, draft K and V 0.28 GB each |
| `runtime_overhead` | 19,225,263,475 | 17.90 | the peak minus every other part: the recurrent-state pool (`Mamba Cache is allocated`, 13.2 GB), the CUDA context and the activations |

`max_total_num_tokens=592576`. The container's host memory peaked at 7.3 GiB, below the
profile's 16 GiB.

## Next

- The FP8 KV cache crash: reproduce it with the live log captured, and report it to
  SGLang with the request (two tool results). It would hold about twice the tokens.
- The SGLang cookbook for this model notes that `extra_buffer_lazy` takes 4 slots per
  request instead of 5, and `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` takes one more off:
  more requests at once without a smaller KV cache. A newer image
  (`dev-qwen38-next-local` or a release from v0.5.22) is needed for some of it.
