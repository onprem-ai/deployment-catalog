# 2026-11-11 - FP8 and MTP on One H200: Tuning, GPU Memory and Benchmark

**Goal:** The fastest setting on one H200 that is fast for one user and good under load,
without changing the answers. Test service on inference node `nebius-e00pn5kkfnayh8wn6j`
(H200 143,771 MiB), with its own model name, beside the production replica on the same GPU.

## Result

- Weights: `JetBrains/Mellum2.1-12B-A2.5B-Thinking-MTP` at `d558461` (the Thinking weights,
  byte-identical, plus the MTP head shard `model-mtp.safetensors`).
- Engine: vLLM v0.33.0 (the MTP repository needs 0.33 or newer).
- `--quantization fp8_per_block`: FP8 weights and activations with 128x128-block scales,
  made at load time. No trusted FP8 checkpoint exists: two community builds with 56 and 115
  downloads.
- `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`, as JetBrains
  recommends.

Full replica benchmark of the compose file as committed, on the production service
(job `e87e05dd-b0c3-429e-a68e-65c6f15e055c`, replica `bf869b18-0acc-49f0-b4e3-32cda7f15af8`,
13:42 to 13:56Z, no request failed; `benchmarks/2026-11-11-1342Z-llmperfbench.yaml`): 562
decode tokens/s for one user (ITL p50 1.76 ms), 20,368 at 128 requests (TTFT p95 0.38 s),
9,463 in the mixed load at 128, 86,368 prefill tokens/s. The run on the test service under
the tuning name (job `600daff4-fd1d-48b9-9e03-1f6ad2d8cc8d`, 13:00 to 13:13Z) agrees within
3 to 7%:

| Workload | 1 request | 32 requests | 128 requests |
|---|---|---|---|
| max-decode, decode tokens/s | 606 (ITL p50 1.63 ms) | 8,126 | 20,995 (TTFT p95 0.33 s) |
| mixed-load, decode tokens/s | 553 | 5,484 | 9,565 (TTFT p95 0.52 s) |
| max-prefill, prefill tokens/s | 22,941 | 89,660 | - |

## Rounds

Short probe on the engine directly (tmp/tune-model-fast `probe.py`: 20 s warm-up, 40 s
window, about 410-token prompts, 1,024 tokens out, thinking off; an early probe of this
session used 10 s and 30 s with about 320-token prompts and 512 tokens out). Decisions on the
inter-token latency, which repeated within 0.1 ms; the probe's throughput spread up to 17%
between runs (calibration against the replica benchmark: ITL 2.4 and 4.7 ms at 1 and 32 in
both).

| Round | Setting | Result | Kept |
|---|---|---|---|
| m0 | BF16, JetBrains' command, vLLM 0.32.0 | 358 / 3,823 / 10,923 tokens/s at 1 / 32 / 128 | baseline |
| m1 | online FP8 per block | 376 / 5,461 / 13,124; ITL 2.4 ms at 1 | yes |
| m3 | m1 + ngram (5 tokens) | 2x on code editing at 1, 36% slower on free text, 82% slower at 128 | no |
| m4 | m1 + 256 requests, 16,384 batched tokens | no gain, failed requests at 128 | no |
| m2b to m10 | m1 + suffix decoding (Arctic Inference, own image) | see below | no |
| mtp1 | m1 + MTP 1 token, vLLM 0.33.0 | ITL 1.6 ms at 1, 3.2 at 32, 5.2 to 5.6 at 128; about 9,400 and 21,700 tokens/s at 32 and 128 | yes |
| mtp2 | m1 + MTP 2 tokens | ITL 1.6 ms at 1, 3.3 to 3.4 at 32, 6.6 at 128; about 9,050 and 18,280 at 32 and 128 | no |

MTP acceptance (vLLM `SpecDecoding metrics`): 1 token, mean acceptance length 1.90, 90% of
drafted tokens accepted; 2 tokens, 2.28, per position 0.90 and 0.38.

Suffix decoding (`onpremai/vllm-arctic:v0.32.0-arctic0.1.1`, vLLM 0.32.0 plus Arctic
Inference 0.1.1, sources in the platform repository under dev/inference-engine-images/vllm-arctic):
up to 3.2x for one user on code editing, but the probe sent the same edit in every request,
which suffix decoding caches, so that figure is too high; free text 45% slower for one user;
vLLM switches async scheduling off with it. A draft length that changes with the batch size
crashes vLLM 0.32.0 (`AssertionError` in `suffix_decoding.py`, error report
`01a1c9f5-094a-737b-b254-2e90d2946da1`). MTP with one token is faster for one user on every
prompt, without these costs.

## Answers

40 of 40 short questions with known answers (arithmetic and Python, 4 repeats, thinking
on) correct with BF16, FP8 and FP8 + MTP 1 and 2. A hidden fact was found at 6K, 25K, 50K
and 94K prompt tokens with FP8, and at 94K with FP8 + MTP. Before tuning, through the API
gateway: thinking, a tool call, thinking off and `/v1/messages` worked.

## GPU memory

Measured on the committed settings, `vram_used_bytes` read every 30 s during the full
benchmark on the production service: peak 65,519,222,784 bytes (61.02 GiB), reached at
13:53:02Z (mixed load). The run on the test service peaked at 59.39 GiB.

| Part | GiB | Source (deployment log of replica `bf869b18`; the same on `7fcba8e8`) |
|---|---|---|
| `model_weights` | 12.16 | `Model loading took 12.16 GiB memory`, the MTP head included (its layer sits in its own file but loads with the model, so no `drafters` part) |
| `cuda_graphs` | 0.92 | `CUDA graph pool memory: 0.92 GiB (actual)` |
| `kv_cache` | 41.21 | `Available KV cache memory: 41.21 GiB`; `GPU KV cache size: 2,015,247 tokens` |
| `runtime_overhead` | 6.73 | the peak minus the other parts |

Engine budget: `Desired GPU memory utilization is (0.4005, 55.99 GiB)`, the 56 GiB engine
booking. Headroom to the 64 GiB hardware booking: 2.98 GiB. Overshoot: 5.02 GiB. Start:
`init engine ... took 190.34 s` (compilation 26 s).

## Next

- Speculative decoding on top of MTP is not available in vLLM; a newer vLLM may allow more
  than one MTP token with better acceptance.
- Suffix decoding measured on unique agentic tasks, for an agentic-coding card.
