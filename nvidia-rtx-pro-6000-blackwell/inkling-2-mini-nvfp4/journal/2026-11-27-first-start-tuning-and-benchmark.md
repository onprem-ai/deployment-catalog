# 2026-11-27 First start, answer checks, one tuning round, full benchmark, GPU memory

**Hardware:** host `ain-xtg6o`, one NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB;
the host's second GPU, index 1), 30 GB of RAM, driver 595.112.07. **Engine:** vLLM
`0.34.1rc1.dev173+g5aea30579`, image
`vllm/vllm-openai:nightly-5aea30579ad672c3c35be43112b7a1f7b293a09e@sha256:d24e5fc7…`. Deployed
through the control plane's development stack from the `drafts` catalog, inference service
`1f069d58-7a1b-4001-aae3-dcfbf7eff729`.

**The other GPU was busy.** Another model (Ling 3.2 Air) was deployed, probed and redeployed on
GPU 0 of the same host during the probes and the full benchmark. The GPUs are separate, the 24
CPU cores and the host's memory are shared: some levels below look disturbed. Repeat the
benchmark on a quiet host before relying on a single figure.

**Precision:** weights mixed NVFP4 and MXFP8, as Thinking Machines ships them; KV cache FP8
(round 1; the recipe leaves 16-bit); the DSpark draft head is part of the checkpoint
(`mtp.safetensors`, BF16).

## First start

vLLM's recipe lists GB200, B300, GB300, H100, H200 and DGX Spark. On this GPU the nightly
resolved `Inkling2LoopForConditionalGeneration` and `Inkling2DSparkModel` and was ready 5.1
minutes after the deploy (later starts 4.3 to 4.5 minutes): weights 21.87 GiB in 17 s, JIT
warm-up 8 s, FlashInfer autotune, CUDA graph capture 97 to 147 s (1.20 GiB). No error.

Warnings in the start-up log:

- `Draft model Inkling2DSparkModel does not support external multimodal embeddings … using
  text-only draft inputs instead`: drafts for image requests come from the text alone.
- `Add 7 padding layers, may waste at most 350.00% KV cache memory` (and 12.5 % twice): the KV
  cache layout pads layer groups. This is why FP8 gained 13 % and not a factor of two.
- `gpu_memory_utilization=0.8844 caps this instance at 83.99 GiB, keeping 4.93 GiB of free
  device memory out of the KV cache`: the engine booking.

## Answer checks (straight to the replica's port), on 16-bit and again on FP8 KV cache

| Check | 16-bit | FP8 |
|---|---|---|
| Image: left half red, right half blue | correct | correct |
| 8 questions with known answers, 3 times each, temperature 0.7 | 24 / 24 | 24 / 24 |
| Reasoning: default, `reasoning_effort` low and high | correct; 116, 106, 116 reasoning tokens | correct; 166, 103, 166 |
| `chat_template_kwargs: {"enable_thinking": false}` | no effect (116 tokens) | no effect |
| Two tool calls in one turn; answer from both results | yes; yes | yes; yes |
| Hidden code at 54,789 / 185,091 / 445,508 prompt tokens | found; 7 / 20 / 82 s | found; 6 / 20 / 78 s |

The model always reasons; its card says the minimum effort does not fully disable it.

## The probe

`replicas_benchmark_start` with `concurrency_levels: [1, 8, 32]` and `requests_per_level: 8`,
`coding-greedy/1.0.0`; `measure_while_serving_clients` true (the control plane counts every
ready replica as serving clients; the service had no other caller). The first level after a
deploy is not reliable: round 1 measured 221 tok/s for one request right after the redeploy
and 286 when repeated.

| Round | Change | coding-greedy decode tok/s at 1 / 8 / 32 | ITL p50 ms | KV pool | Kept |
|---|---|---|---|---|---|
| 0 | baseline, the recipe (16-bit KV cache, 7 draft tokens) | 283 / 852 / 1,983 | 3.8 / 7.9 / 12.5 | 760,811 tokens | |
| 1 | `--kv-cache-dtype fp8` | 286 / 928 / 2,106 | 3.7 / 8.1 / 13.3 | 861,838 tokens | yes |

- Baseline `mixed-load/1.1.0` at 1 / 8 / 32: 129 / 236 / 496 decode tok/s.
- Draft acceptance by position (engine's figures): on real coding prompts 0.82, 0.67, 0.55,
  0.46, 0.40, 0.34, 0.27 (mean 4.5 to 4.9 tokens per step); on the random-word prompts of
  `mixed-load` and `max-decode` 0.66, 0.40, 0.23, 0.12, 0.06, 0.03, 0.01 (mean 2.5). The 7
  draft tokens stay.
- Not tried: `--max-num-seqs` (vLLM's default), fewer draft tokens, a larger engine booking,
  two GPUs for the full 1M context.

**The full context does not fit one GPU.** One request of 1,048,576 tokens needs a pool of
that size; with 84 GiB for the engine the FP8 pool holds 861,838 tokens. The profile stays at
524,288 tokens per request (1.64 requests).

## Full benchmark

`benchmarks/2026-11-27-0308Z-llmperfbench.yaml`, on the committed settings (its
`compose_sha256` is this profile's `compose.yaml`), load generator `dev-892868be`, 49 minutes,
no failed request at any of 35 levels.

| Workload | 1 request | Plateau | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 5,340 prefill tok/s | 16 | 19,085 prefill tok/s at 32 |
| `max-decode/1.1.0` | 238 decode tok/s, 4.3 ms per token | 40 | 2,376 decode tok/s at 48 |
| `mixed-load/1.1.0` | 70 decode tok/s | 80 | 1,912 decode tok/s at 112 |

- Disturbed levels: `max-prefill` at 8 requests (7,033 prefill tok/s between 14,202 at 4 and
  15,868 at 12) and `mixed-load` at 1 request (70 decode tok/s; the probe measured 129).
- The benchmark's `max-decode` and `mixed-load` prompts are random words, which the drafter
  guesses a third as often as real text: real coding prompts decode 283–286 tok/s for one
  request (the probe), against 238 here.

## GPU memory

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 23,482,733,690 | 21.87 | `Model loading took 21.87 GiB memory` (the draft head's file included) |
| `cuda_graphs` | 1,288,490,188 | 1.20 | `Graph capturing finished in 97 secs, took 1.20 GiB` |
| `kv_cache` | 59,538,984,960 | 55.45 | the engine's 83.99 GiB minus 23.21 GiB consumed, 4.13 GiB peak activation and 1.2 GiB of graphs (`Desired GPU memory utilization is (0.8844, 83.99 GiB)…`) |
| `runtime_overhead` | 10,071,068,346 | 9.38 | the peak minus the parts above |
| **Peak** | 94,381,277,184 | 87.90 | 90,009 MiB, `nvidia-smi` every 30 s during the repeated benchmark |

- KV cache: 861,838 tokens (`GPU KV cache size`).
- Engine booking 84 GiB (the log's 83.99 GiB): the peak is 1.10 GiB above it. Hardware
  booking 88 GiB: 2.90 GiB of headroom.
- After the long prompts alone the GPU held 82,169 MiB; the benchmark's concurrency drove the
  peak.
- Host memory: the container held 7.1 GiB of its 16 GiB limit after the benchmark.

## Repeated on a quiet host

The full benchmark was repeated with nothing else running on the host, on the same compose
file: `benchmarks/2026-11-27-0829Z-llmperfbench.yaml`, 70 minutes, no failed request at any
of 44 levels.

| Workload | 1 request | Plateau | Best | First run (other GPU busy) |
|---|---|---|---|---|
| `max-prefill/1.0.0` | 5,465 prefill tok/s | 24 | 20,801 prefill tok/s at 40 | 5,340; plateau 16; 19,085 |
| `max-decode/1.1.0` | 211 decode tok/s, 4.5 ms per token | 112 | 3,945 decode tok/s at 160 | 238; plateau 40; 2,376 |
| `mixed-load/1.1.0` | 122 decode tok/s | 128 | 2,055 decode tok/s at 128 | 70; plateau 80; 1,912 |

- **The first run undercounted the throughput.** With the host to itself the replica keeps
  gaining to 112–128 requests in flight; in the first run `max-decode` stopped at 40 with
  2,376 tok/s. This file replaces the first run's figures; the first file stays as the record
  of that run.
- `max-decode` for one request is 211 here against 238 before and 283–286 on real coding
  prompts: its prompts are random words, and the drafter's luck on them varies from run to
  run.
- The engine runs vLLM's default number of sequences; the benchmark's ladder ended at 160
  because the gain stopped, not because requests failed or waited (first token after 0.3 s at
  160 in `max-decode`).

After the benchmark: one 441,385-token prompt (hidden code found, 76 s).

**Through the API gateway** (`https://<control plane>/v1`, an API gateway key): image input
correct; 24 of 24 known answers; reasoning default and `reasoning_effort` low and high
correct; two parallel tool calls and an answer from both results; a hidden code found at
54,789 prompt tokens.

**GPU memory on this run:** peak **90,009 MiB (87.90 GiB)** at the high concurrency levels
(87,141 MiB in the first run, which stopped at 48 to 112 requests). It is the value now in
`runtime.measured_gpu_memory_bytes_per_device` (`runtime_overhead` 10,071,068,346 bytes,
9.38 GiB): 3.90 GiB above the 84 GiB engine booking. The hardware booking was 88 GiB,
0.10 GiB above this peak; it is now 90 GiB. Host memory: the container held 11.5 GiB of its
16 GiB limit.

## Next

- Move to the vLLM 0.35.0 release image when it exists.
- `--max-num-seqs`, and a second GPU (tensor parallel 2) for the full 1,048,576-token context.
