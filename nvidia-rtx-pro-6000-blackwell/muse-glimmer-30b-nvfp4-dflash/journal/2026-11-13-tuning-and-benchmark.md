# 2026-11-13 - Tuning and Full Benchmark on One RTX PRO 6000

**Goal:** Make the card faster with one change per round, check its answers, and run the
full replica benchmark once on the chosen settings. This closes the follow-up of
2026-08-16 (memory floor, speed measurements, DFlash block size).

## Setup

- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GiB, SM120), inference node
  `ain-xtg6o`, driver 595.112.07, CUDA 13.2.
- Image and weights as pinned: SGLang `dig-b8022194`, `RadixArk/Muse-Glimmer-NVFP4`
  (mixed NVFP4 and MXFP8), Meta's DFlash drafter, quantized to FP8 at load. KV cache FP8.
- Served as `muse-glimmer-30b-tuning` during the session.
- Probe as in the Qwen3.8 27B journal of 2026-11-12: unique prompts of about 310 tokens,
  512-token answers with `ignore_eos`, through the API gateway's replica path. Muse always
  reasons (no request option turns it off), so the answers include reasoning tokens.

## Answers

Through the API gateway, on the card as published and on the chosen settings: four
short questions with known answers, three times each; a tool call and an answer from
its result; `/v1/messages`; a hidden number at 6,798, 28,533 and 105,698 prompt tokens.
16 of 18 both times. The two failures are one finding: asked for the weather in two
cities, the model calls the tool once (for the first city) and then answers, as in the
journal of 2026-08-16 and on the H200 card. Parallel tool calls are not reliable with
this model.

## Rounds

| Round | Change | c=1 | ITL c=1 | c=16 | c=32 | c=48 | c=64 |
|---|---|---|---|---|---|---|---|
| 0 | as published: DFlash block 5, share 0.78, 48 running | 158-162 | 6.2-6.3 ms | 1,363 | 1,933 | 2,286 | |
| 1 | DFlash block size 8 | 200-204 | 4.9-5.0 ms | 1,599 | 1,981 | 2,073 | |
| 2 | DFlash block size 12 | 175-229 | 4.4-5.7 ms | 1,305 | 1,546 | 1,554 | |
| 3 | block 8, memory share 0.85 | 184 | 5.4 ms | | 1,962 | 2,087 | 1,414 (16 waiting) |
| 4 | block 8, share 0.78, 64 running requests | 173-225 | 4.5-5.8 ms | 1,614 | | 2,095 | 2,137 |

Prefill at 1 request, 5,582-token prompts (round 0): 1.09 s to the first token (5,134
tokens/s).

## What the rounds show

- **DFlash block size.** The drafter is trained for blocks of 16; the card used 5, the
  value of SGLang's regression test. Block 8 gives about 25 percent more for one user and
  17 percent more at 16 requests, and stays within 10 percent at 48. Block 12 is unstable
  for one user and loses 18 to 25 percent under load: a longer block wastes verification
  work once the batch is large.
- **One-user speed varies with the prompts.** With DFlash the speed depends on how many
  draft tokens the target accepts, which changes with the random prompts: four runs of
  round 4 read 173 to 225 tokens/s. The single-user figures of rounds 1, 3 and 4 overlap.
- **The memory share is not the limit.** SGLang capped the replica at 48 running requests
  by its own default; a larger share (round 3) grew the KV cache by 13 percent and changed
  nothing, and at 64 requests 16 waited. `--max-running-requests 64` (round 4) lets 64 run:
  2,137 tokens/s, with the time to first token at 0.7 s instead of 8.2 s.

## Chosen

Round 4: DFlash with block size 8, memory share 0.78, 64 running requests, FP8 KV cache,
131,072 tokens of context. Against the card as published: about 25 percent faster for one
user, 17 percent at 16 requests, and 64 requests at once instead of 48.

## The full benchmark

Job `65fe8451-93f5-44ee-a3b5-8a510deb8b53` on replica
`079e1c2d-a200-4bb9-9d5b-91941b6885c2`, all three workloads, 20:40:54 to 21:31:45Z, on the
committed settings (`compose_sha256` `89f2eacf...`). It succeeded: every workload ended at
its plateau and no request failed.

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 8 | 3,315 prefill tokens/s at 1 (TTFT p50 1.54 s); 4,913 at 8 |
| `max-decode/1.1.0` | 32 | 178 decode tokens/s at 1 (ITL p50 5.4 ms); 1,692 at 16; 2,060 at 32; 2,120 at 48 |
| `mixed-load/1.1.0` | 32 | 99 decode tokens/s at 1; at 32: 3,523 prefill and 654 decode tokens/s |

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
88,174,755,840 bytes (82.12 GiB). It was above the card's 75 GiB hardware booking, so the
hardware booking is 85 GiB. The engine booking is 75 GiB, the card's former hardware
booking: from it the deploy writes SGLang's share (0.785), the 0.78 the card measured with.
Without it, the deploy would hand SGLang the whole 85 GiB, and the checker warns
(`replica.engine-gpu-memory`).

| Part | Bytes | GiB | Source (SGLang's start-up lines) |
|---|---|---|---|
| `model_weights` | 20,626,580,439 | 19.21 | `Load weight end ... MuseGlimmerForConditionalGeneration ... mem usage=19.21 GB` |
| `drafters` | 3,253,437,726 | 3.03 | `Load weight end ... MuseGlimmerAssistantModel ... mem usage=3.03 GB` |
| `kv_cache` | 55,136,642,662 | 51.35 | free memory before the two memory pools (72.06 GB) minus after (20.71 GB): the full-attention and sliding-window KV caches and the drafter's cache |
| `cuda_graphs` | 4,133,906,022 | 3.85 | the three captures: target prefill 0.89, target verify 1.73, draft verify 1.23 GB |
| `runtime_overhead` | 5,024,188,991 | 4.68 | the peak minus every other part |

`max_total_num_tokens=1678156`. The container's host memory peaked at 6.7 GiB, below the
profile's 16 GiB limit.

## Next

- Parallel tool calls: a capability the profile format cannot state yet (`tool_use` is
  one boolean); proposed to be tracked separately.
- The H200 card runs vLLM v0.33.0 with DFlash; vLLM may now run this NVFP4 checkpoint on
  SM120 too, which would drop the patch to SGLang's MXFP8 guard in the command.
