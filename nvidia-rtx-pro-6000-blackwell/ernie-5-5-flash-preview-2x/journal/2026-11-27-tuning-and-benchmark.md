# 2026-11-27 Answer checks, two tuning rounds, full benchmark, GPU memory

**Hardware:** host `ain-xtg6o`, 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB
each), 30 GB of RAM, driver 595.112.07. **Engine:** vLLM 0.32.0 with Baidu's vLLM-ERNIE plugin
at `ba322a5` and the SM120 port V2, image
`onpremai/vllm-ernie55:0.32.0-ba322a5-sm120v2@sha256:f6977c10…` (a local build, in no registry).
Deployed through the control plane's development stack from the `drafts` catalog, inference
service `24d0763a-0649-418b-9b2a-ff92eecee613`, both GPUs, tensor parallel 2.

**Precision:** weights blockwise FP8 with MXFP4 routed experts, as Baidu ships them
(quantization-aware trained); KV cache FP8 (`--kv-cache-dtype fp8`); the DSpark draft head is
part of the checkpoint, in its precision. Nothing is 16-bit by choice.

**Goal:** a balance of one-user speed and throughput, keeping 300,000 tokens per request.
Tuning was capped at one hour before the full benchmark.

## Answer checks (baseline settings, straight to the replica's port)

| Check | Result |
|---|---|
| 8 questions with known answers, 3 times each, temperature 0.7 | 24 / 24 |
| Thinking default, `enable_thinking: false`, `reasoning_effort` low and high | all correct; 92, 0, 50 and 92 reasoning tokens |
| Two tool calls in one turn | both, with the right arguments |
| Answer built from the two tool results | uses both |
| Hidden code at 6,680 / 51,475 / 130,160 / 243,702 prompt tokens | found each time; 2.7 / 11.7 / 32.7 / 63.8 s |

Not checked: the API gateway path, languages other than English and German, streaming.

## The probe

`replicas_benchmark_start` with `concurrency_levels` and `requests_per_level: 8`,
`coding-greedy/1.0.0` (real prompts, the workload for a speculative decoding setting). The
baseline also ran `mixed-load/1.1.0` at 1 / 4 / 8: 107 / 232 / 342 decode tokens per second.
`measure_while_serving_clients` was true: the control plane counts every ready replica as
serving clients, and this service had no other caller. One probe per round, no repeat: the
noise is not measured, so differences of a few percent are not results.

## Rounds

| Round | Change | coding-greedy decode tok/s at 1 / 4 / 8 / 12 | ITL p50 ms | Engine's figures | Kept |
|---|---|---|---|---|---|
| 0 | baseline: 7 draft tokens, 8 sequences | 150 / 339 / 501 / – | 6.4 / 11.2 / 16.4 | per-position acceptance 0.75, 0.47, 0.29, 0.17, 0.09, 0.04, 0.02; graphs 1.55 GiB | |
| 1 | `num_speculative_tokens` 7 → 4 | 161 / 356 / 497 / – | 5.9 / 11.2 / 15.2 | acceptance 0.81, 0.61, 0.44, 0.31 (54 % of drafted tokens, 26 % before); graphs 1.36 GiB | yes |
| 2 | `--max-num-seqs` 8 → 12 | 165 / 353 / 498 / 624 | 5.8 / 11.0 / 15.4 / 19.5 | graphs 2.06 and 2.15 GiB; KV pool unchanged, 1,213,175 tokens | yes |

- Round 1: tokens 5 to 7 of the port's 7-token draft were almost never accepted. Four tokens
  are 5–8 % faster for one and four requests and the same at eight.
- Round 2: 12 × (1 + 4) = 60 tokens in a decode batch, below the baseline's 64. No change up
  to 8 requests; 25 % more tokens per second at 12. Before it was kept: one 242,496-token
  prompt, then four 216,416-token prompts at once, every hidden code found, peak 97,023 MiB,
  no engine error.
- No failed request and no engine error in any round. Every redeploy was ready in 6.6 to 7.1
  minutes.

Not tried: 3 or 5 draft tokens, 16 sequences, a larger prefill chunk (the HCA compressor's
scratch state grows with it, and 224 MiB are free at the peak), `--kv-cache-memory-bytes`
other than the port's, Baidu's data-parallel layout.

## Full benchmark

`benchmarks/2026-11-27-0017Z-llmperfbench.yaml`, on the committed settings (its
`compose_sha256` is this profile's `compose.yaml`), load generator `dev-892868be`, 122 minutes,
no failed request at any of 37 levels.

| Workload | 1 request | Plateau | At the plateau and above |
|---|---|---|---|
| `max-prefill/1.0.0` | 3,519 prefill tok/s, first token after 1.7 s | 8 | 4,700–4,820 prefill tok/s |
| `max-decode/1.1.0` | 137 decode tok/s, 7.2 ms per token | 12 | 594–611 decode tok/s |
| `mixed-load/1.1.0` | 109 decode tok/s | 24 | 355–416 decode tok/s |
| `coding-greedy/1.0.0` | 157 decode tok/s, 6.2 ms per token | 12 | 604–623 decode tok/s |

- Above 12 requests the rest wait: at 32, the first token of a `max-decode` request comes
  after 34 s.
- `coding-greedy` was named beside the three default workloads; its full ladder took 50 of the
  122 minutes. Its extra figures are not in the benchmark file and the job's output was gone
  when the run ended, so the draft acceptance is the engine's own, summed over the workload's
  window (306 intervals of 10 s): 1,155,468 of 2,367,036 drafted tokens accepted (48.8 %), by
  position 0.78, 0.56, 0.37, 0.24, a mean of 2.95 tokens per step.
- `max-decode` and `mixed-load` send random words, which the drafter guesses less often; real
  prompts are 15 % faster for one request (157 against 137).

## GPU memory

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 85,534,273,699 | 79.66 | `Model loading took 79.66 GiB memory` (both workers) |
| `cuda_graphs` | 2,308,544,921 | 2.15 | `Graph capturing finished in 35 secs, took 2.15 GiB` (the larger worker) |
| `kv_cache` | 9,680,446,000 | 9.02 | `reserved 9.02 GiB memory for KV Cache as specified by kv_cache_memory_bytes` |
| `runtime_overhead` | 4,214,821,780 | 3.93 | the peak minus the parts above |
| **Peak** | 101,738,086,400 | 94.75 | 97,025 MiB, `nvidia-smi` every 2 s |

- KV cache: 1,213,175 tokens (`GPU KV cache size`), 4.04 requests of 300,000 tokens.
- The peak comes from long prompts, not from the benchmark: 96,001 MiB during the benchmark
  (sampled every 30 s, prompts of at most 8,192 tokens), 97,025 MiB after one 242,496-token
  prompt and four 216,416-token prompts at once on the benchmarked replica. It was 97,027 MiB
  on the baseline and 97,023 MiB after round 2: the engine's ceiling, and the figure the SM120
  port reports.
- Hardware booking 97,152 MiB (101,871,255,552 bytes): 127 MiB above the peak, 99 MiB below
  what an idle GPU has free (97,251 MiB). There is no engine booking: the KV cache is sized
  in bytes and the command has no `--gpu-memory-utilization`, on purpose (`compose.yaml`).
- Host memory: the container held 12.8 GiB during the start and 8.4 GiB after the benchmark
  and the long prompts, of a 20 GiB limit.

## Next

- Put the image in a registry (after clearing the SM120 port's license) and pin that digest.
- Measure the probe's noise; then 3 and 5 draft tokens, and 16 sequences.
- A request cap above 300,000 tokens (the pool holds 1.21M).
- Check the answers through the API gateway.
