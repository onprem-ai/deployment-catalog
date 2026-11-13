# 2026-11-12 - Tuning and Full Benchmark on One RTX PRO 6000

**Goal:** The best single-user speed that keeps most of the throughput at many users,
on one RTX PRO 6000 Blackwell Server Edition, in four hours; then the full benchmark once
on the chosen settings.

## Setup

- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, SM120), inference node
  `ain-xtg6o`, driver 595.112.07, CUDA 13.2. The node's second GPU ran other work.
- The profile was copied from the official catalog into the `drafts` catalog and served as
  `qwen3.8-27b-tuning` during the session, so it received no client's request.
- Engine: vLLM, from v0.27.1 to v0.33.0 in round 1.
- Precision: weights NVFP4 (Unsloth Dynamic V3.0, mixed with FP8 channel-wise layers, as
  published); KV cache FP8 (`--kv-cache-dtype fp8`); MTP head from the checkpoint, as
  shipped. The linear-attention (Gated DeltaNet) state is bfloat16 from round 10, see below.

## How it was measured

The probe ran from outside the node through the API gateway's replica path
(`/_debug/replicas/<replica>/inference/8080/`), straight to one replica: exactly `c`
requests in flight, unique prompts of about 255 tokens, 512-token answers with
`ignore_eos`, reasoning off, 20 s warm-up and a 40 s window, at 1, 32 and 128 requests.
Prefill: about 5,300-token prompts with 64-token answers at 1 and 16. Two runs at 1 agreed
within 2 percent (inter-token latency 10.7 and 10.9 ms); the first run after each deploy
was discarded (its time to first token holds compilation). The engine's own figures (running
and waiting requests, KV cache use, speculative acceptance, preemptions) were read from its
`/metrics` at the window's start and end, and the replica's GPU memory from the control
plane's metrics.

## Answers

With the baseline and again in rounds 1, 8 and 10 (the rounds that could change answers),
through the API gateway: four short questions with known answers (arithmetic, a count,
a capital, one in German) three times with reasoning off and once at each effort (`low`,
`medium`, `xhigh`); two parallel tool calls and an answer built from their results; an
image (two colors in order); `/v1/messages`; a hidden number at 6,757, 28,492, 115,455
and 231,398 prompt tokens. 32 of 32 in every round.

Round 10 changes the precision of the linear-attention state, so the first 400 GSM8K
test questions (reasoning off, temperature 0) ran on both: bfloat16 385, float32 387;
383 right on both, 4 only on float32, 2 only on bfloat16. The difference is within the
noise of 400 questions.

## Start-up

v0.33.0: `Model loading took 22.09 GiB`; FlashInfer GDN prefill kernel (v0.27.1 used the
Triton/FLA one); `FlashInferCutlassNvFp4LinearKernel`; with the Triton attention backend,
full and piecewise CUDA graphs (FlashInfer allows only piecewise graphs with speculative
decoding). Ready about 2.5 minutes after the container starts. The chat template merges
inline system messages of `/v1/messages` requests into the leading system prompt (vLLM's
start-up warning), which costs prefix-cache hits for clients that send them.

## Rounds

Decode tokens per second; ITL is the median inter-token latency.

| Round | Change | c=1 | ITL c=1 | c=32 | c=128 | TTFT p50 at 128 | Acceptance | KV tokens |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline: v0.27.1, MTP 2, 4,096 batched, 0.88 share | 91 | 10.8 ms | 1,708 | 2,438 | 1.0 s | 2.14-2.17 | 1,683,083 |
| 1 | vLLM v0.33.0 | 100 | 9.9 ms | 1,841 | 2,645 | 1.05 s | 2.15 | 1,638,766 |
| 2 | DFlash2 drafter, 7 tokens | 121 | 8.3 ms | 1,331 | 1,325 | 8.6 s | 2.70 | 1,203,028 |
| 3 | DFlash2, 4 tokens | 120 | 8.3 ms | 1,648 | 1,807 | 7.7 s | 2.60 | 1,254,229 |
| 4 | MTP, 3 tokens | 105 | 9.5 ms | 1,747 | 2,261 | 3.5 s | 2.38 | |
| 5 | MTP 2; 88 GiB engine booking, 91 GiB hardware booking | 101 | 9.9 ms | 1,832 | 2,674 | 0.73 s | 2.15 | 1,751,531 |
| 6 | 8,192 batched tokens | 99 | 10.1 ms | 1,832 | 2,679 | 0.73 s | 2.15 | 1,748,602 |
| 7 | prefix caching off | 99 | 10.1 ms | 1,834 | 2,631 | 1.56 s | 2.14 | 1,809,248 |
| 8 | Triton attention backend (full CUDA graphs) | 104 | 9.6 ms | 1,849 | 2,684 | 0.66 s | 2.15-2.22 | 1,760,318 |
| 9 | MTP, 1 token | 87 | 11.5 ms | 1,749 | 2,898 | 0.53 s | 1.71 | 1,796,282 |
| 10 | linear-attention state bfloat16 | 103 | 9.7 ms | 2,215 | 3,565 | 0.45 s | 2.15 | 1,831,048 |
| 11 | float32 state again (accuracy reference only) | | | | | | | |
| 12 | 256 sequences | 101 | 9.9 ms | 2,219 | 3,607 (192: 3,523) | 0.42 s | 2.15 | 1,821,544 |
| 13 | MTP, 3 tokens (bfloat16 state) | 103 | 9.7 ms | 2,172 | 3,378 | 0.44 s | 2.35 | |
| 14 | async scheduling | 101 | 9.9 ms | 2,238 | 3,585 | 0.43 s | 2.15 | |
| 15 | 16,384 batched tokens | 102 | 9.8 ms | 2,230 | 3,604 | 0.40 s | 2.15 | |

Prefill, time to first token for about 5,300 tokens: round 1 0.49 s at 1 and 1.81 s
(median) at 16; round 6 0.47 s and 1.69 s; round 8 0.50 s and 1.77 s; round 15 0.51 s and
3.21 s.

## What the rounds show

- **The linear-attention state, not compute, limited the replica under load.** Qwen3.8
  keeps a recurrent state for each of its 48 Gated DeltaNet layers; the checkpoint declares
  it float32 (`mamba_ssm_dtype`), about 154 MB per request. vLLM sizes the KV cache block
  to hold one state page (1,600 tokens) and keeps the state per block group and per draft
  position, so each running request held about 9 blocks however short it was. Up to round
  9, 128 requests in flight ran only about 111 to 118 at once with the KV cache at 99 to
  100 percent.
- That is why every option that adds state per request lost under load: DFlash2 (rounds 2
  and 3: 20 percent faster for one user, 10 to 50 percent slower at 32 and 128, 39 to 61
  requests running), MTP with 3 tokens (round 4), and why prefix caching off (round 7) did
  not help.
- **The bfloat16 state (round 10) removed the limit:** the block is 832 tokens, 128
  requests run at once with the KV cache at 55 percent, and the decode throughput rose 20
  percent at 32 and 33 percent at 128, at the same single-user speed. vLLM's Qwen3.5
  throughput work lists the same option ("significantly increases the effective KV cache
  capacity", https://vllm.ai/blog/2026-08-06-qwen35-25k-tps); SGLang's Qwen3.8-27B guide
  asks for an accuracy check with it, which the GSM8K comparison above is.
- After it, the replica is at its plateau near 128 requests (round 12), 3 draft tokens
  still do not pay (round 13), and async scheduling and larger prefill steps change nothing
  or make long prompts wait (rounds 14 and 15).
- The Triton attention backend (round 8) lets vLLM capture full CUDA graphs with MTP and
  makes one user about 5 percent faster; prefill is 4 to 8 percent slower.
- vLLM v0.33.0 (round 1) gave 9 percent for one user and 7 to 8 percent under load.

## Chosen

Round 10: vLLM v0.33.0, MTP with 2 draft tokens, Triton attention, FP8 KV cache, bfloat16
linear-attention state, 128 sequences, 8,192 batched tokens, prefix caching on, 262,144
tokens of context, 88 GiB engine booking. Against the starting settings: one user 103
against 91 tokens per second (+13 percent), 2,215 against 1,708 at 32 (+30 percent), 3,565
against 2,438 at 128 (+46 percent), and the time to first token at 128 from 1.0 s to
0.45 s (p95 from 3.4 to 9.3 s down to 0.7 s). It gives up the 20 percent single-user gain
of DFlash2, which costs too much under load on this GPU.

## The full benchmark

Job `302fd8d3-a788-4967-b1c6-1cc540f04ca9` on replica `67490f5a-7b9b-4f90-8afd-2c7b81acc476`,
all three workloads, 20:34:18 to 21:25:18Z, on the committed settings
(`compose_sha256` `69029e3c...`). It succeeded: every workload ended at its plateau and no
request failed. The replica ran without other traffic (`measure_while_serving_clients`
was set only because the test model name counted as served).

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 12 | 5,838 prefill tokens/s at 1 (TTFT p50 0.71 s for about 7,900 tokens); 10,240 at 32 |
| `max-decode/1.0.0` | 64 | 118 decode tokens/s at 1 (ITL p50 8.3 ms); 2,613 at 32; 3,613 at 64; 3,787 at 96 (TTFT p50 0.24 s) |
| `mixed-load/1.0.0` | 80 | 112 decode tokens/s at 1; at 80: 7,606 prefill and 1,495 decode tokens/s |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
100,124,327,936 bytes (93.25 GiB), during `max-decode`. It was above the 93 GiB hardware
booking of the tuning rounds, so the hardware booking is 95 GiB (102,005,473,280 bytes):
1.75 GiB of headroom, and 5.25 GiB of overshoot over the 88 GiB engine booking.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 23,718,956,892 | 22.09 | `Model loading took 22.09 GiB` (the MTP head is in it) |
| `cuda_graphs` | 697,932,185 | 0.65 | `CUDA graph pool memory: 0.65 GiB (actual)` |
| `kv_cache` | 66,969,277,562 | 62.37 | `Available KV cache memory: 62.37 GiB` |
| `runtime_overhead` | 8,738,161,297 | 8.14 | the peak minus every other part |

`GPU KV cache size: 1,831,048 tokens` (6.98 requests of 262,144 tokens). The GPU shows
94.97 GiB to CUDA, so 95 GiB leaves no room for another replica on it. The container's
host memory peaked at 6.8 GiB (round 12), below the profile's 16 GiB.

## Next

- DFlash2 gave 120 tokens per second for one user. With the bfloat16 state, its extra
  state per request may fit now: worth one round at 4 draft tokens.
- A smaller engine booking (for example 84 GiB) would keep the peak further below the
  hardware booking, at the cost of KV cache that the replica no longer fills at 128.
- The probe and the benchmark differ for one user (103 against 118 tokens per second): the
  probe's path through the API gateway and its prompts lower the acceptance; compare the
  two methods once before the next session.
