# 2026-11-27 First starts (two failed), answer checks, three tuning rounds, full benchmark, GPU memory

**Hardware:** host `ain-xtg6o`, one NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB;
the host's first GPU, index 0), 30 GB of RAM, driver 595.112.07. **Engine:** vLLM
`0.33.0rc1.dev806+ga2bcb5657`, image
`vllm/vllm-openai:cu134-nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121@sha256:76a53347…`.
Deployed through the control plane's development stack from the `drafts` catalog, inference
service `1dee5251-4d6f-42a1-b45a-951b2646e6bd`.

**The other GPU was busy.** Another model (Inkling 2 Mini, then Qwen3.8 27B with a LoRA) was
deployed, probed and benchmarked on GPU 1 of the same host during the probes and the full
benchmark. The GPUs are separate; the 24 CPU cores and the host's memory are shared. Repeat
the benchmark on a quiet host before relying on a single figure.

**Precision:** weights primitive-ai's build: NVFP4 routed experts, the rest BF16 (dequantized
from inclusionAI's FP8). KV cache 16-bit: FP8 does not start (round 3). The MTP layer is part
of the checkpoint, its experts NVFP4.

## Starts

| Try | Setting | Result |
|---|---|---|
| 1 | `--max-model-len 262144`, vLLM's default 1,024 sequences | failed: `torch.OutOfMemoryError … Tried to allocate 23.55 GiB` for the Triton MLA attention workspace, 17.76 GiB free after the weights |
| 2 | `--max-model-len 65536`, 1,024 sequences | failed: `max_num_seqs (1024) exceeds available Mamba cache blocks (406)`; 5.86 GiB were left for the KV cache |
| 3 | `--max-model-len 65536 --max-num-seqs 64` | ready after 3.3 minutes; 14.87 GiB for the KV cache, 1,243,727 tokens |

The workspace and the memory profile grow with the number of sequences, not only with the
context: with 64 sequences the native 262,144 tokens start (round 1).

Model directory: tensors, JSON and the chat template from `primitive-ai/Ling-3.2-air-NVFP4`,
the seven Python files from `inclusionAI/Ling-3.2-air`, linked into `/tmp/model` at start.
The engine loaded them with `--trust-remote-code` and resolved
`BailingMoeV3VLForConditionalGeneration`. Starting without the Python files was not tried.

`--mamba-cache-mode` is deprecated in this build ("It has no effect") and was removed.

## Answer checks (straight to the replica's port)

| Check | First settings (65,536, no MTP) | Final settings (262,144, MTP 3) |
|---|---|---|
| Image: left half red, right half blue | correct | correct |
| 8 questions with known answers, 3 times each, temperature 0.7 | 24 / 24 | 24 / 24 |
| Thinking default; `enable_thinking: false` | correct, 197 reasoning tokens; correct, 0 | correct, 197; correct, 0 |
| Two tool calls in one turn; answer from both results | yes; yes | yes; yes |
| Hidden code | found at 7,911 / 30,212 / 61,497 prompt tokens (1.1 / 2.4 / 4.6 s) | found at 61,462 / 155,387 / 249,609 (4.3 / 14.8 / 31.1 s) |

After its tool results the model opens with "The task has already been completed! Here's a
recap…" before the two answers: correct, and wordy.

## The probe

`replicas_benchmark_start` with `concurrency_levels: [1, 8, 32]` and `requests_per_level: 8`,
`coding-greedy/1.0.0`; `measure_while_serving_clients` true (the control plane counts every
ready replica as serving clients; the service had no other caller). One probe per round, the
other GPU busy: differences of a few percent are not results.

| Round | Change | coding-greedy decode tok/s at 1 / 8 / 32 | ITL p50 ms | KV pool | Kept |
|---|---|---|---|---|---|
| 0 | baseline: 65,536 tokens, 64 sequences, no MTP | 110 / 510 / 1,060 | 9.1 / 15.8 / 30.9 | 1,243,727 tokens | |
| 1 | `--max-model-len` 65536 → 262144 | 105 / 460 / 923 | 9.5 / 17.5 / 35.4 | 1,430,200 tokens | yes |
| 2 | MTP, `num_speculative_tokens` 3 | 174 / 605 / 686 | 5.6 / 12.8 / 14.4 | 421,486 tokens | yes |
| 3 | `--kv-cache-dtype fp8` | did not start | | | no |

- Baseline `mixed-load/1.1.0` at 1 / 8 / 32: 103 / 446 / 890 decode tok/s.
- Round 1: the native context starts. The 5–13 % lower figures are within what the busy host
  may explain; not separated.
- Round 2: MTP is 66 % faster for one request and 31 % for eight, and 26 % slower at 32,
  where the first token came after 60 s: the KV pool is 421,486 tokens instead of 1,430,200
  (the MTP layer's weights, its graphs and its own cache), and requests wait for it.
  Acceptance by position 0.87, 0.60, 0.31 on coding prompts.
- Round 3: `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required:
  102400, Hardware limit: 101376` in the Triton MLA kernel
  (`_fwd_grouped_kernel_stage1`). FP8 KV cache is not available for this model on this image
  and GPU; the cache stays 16-bit.

**The choice:** MTP on. It gives up throughput at many requests and most of the KV pool
(1.61 requests of 262,144 tokens) for the speed of one to eight requests. Without the
`--speculative-config` line the profile is round 1.

Not tried: 2 draft tokens, fewer sequences with MTP, a shorter context with MTP, starting
without the Python files, SGLang (inclusionAI's recommended engine, for its FP8 release).

## Full benchmark

`benchmarks/2026-11-27-0333Z-llmperfbench.yaml`, on the committed settings (its
`compose_sha256` is this profile's `compose.yaml`), load generator `dev-892868be`, 37 minutes,
no failed request at any of 27 levels.

| Workload | 1 request | Plateau | At the plateau and above |
|---|---|---|---|
| `max-prefill/1.0.0` | 9,373 prefill tok/s | 8 | 16,829–17,695 prefill tok/s |
| `max-decode/1.1.0` | 152 decode tok/s, 6.6 ms per token | 12 | 601–622 decode tok/s |
| `mixed-load/1.1.0` | 123 decode tok/s | 12 | 508–543 decode tok/s |

- Above 12 requests the rest wait for KV cache: at 32, the first token of a `max-decode`
  request comes after 36 s.
- Draft acceptance on the benchmark's random-word prompts: 0.70, 0.39, 0.16 by position
  (2.2 to 2.3 tokens per step); on real coding prompts 0.87, 0.60, 0.31 (2.8).

## GPU memory

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 78,275,778,969 | 72.90 | `Model loading took 72.9 GiB memory` (the MTP layer included) |
| `cuda_graphs` | 1,299,227,607 | 1.21 | `Graph capturing finished in 10 secs, took 1.21 GiB` |
| `kv_cache` | 6,968,584,437 | 6.49 | `Available KV cache memory: 7.7 GiB` minus the graphs |
| `runtime_overhead` | 14,594,709,915 | 13.59 | the peak minus the parts above |
| **Peak** | 101,138,300,928 | 94.19 | 96,453 MiB, `nvidia-smi` after a long prompt on the quiet-host run |

- KV cache: 421,486 tokens (`GPU KV cache size`).
- The engine's log: `Desired GPU memory utilization is (0.9265, 87.99 GiB). Actual usage is
  78.97 GiB for consumed memory (weights + non-torch), 1.32 GiB for peak activation, and
  1.21 GiB for CUDAGraph memory`. The engine booking is 88 GiB.
- Peak 95,487 MiB during the benchmark (every 30 s), 96,309 MiB after prompts of up to
  249,609 tokens: 6.05 GiB above the engine booking. The hardware booking was 94 GiB
  (96,256 MiB), 53 MiB below this peak; it is now 96,768 MiB (101,468,602,368 bytes), 459 MiB
  above the peak and 483 MiB below what an idle GPU has free (97,251 MiB). Lowering the
  engine booking instead would shrink a KV pool that is already small.
- Host memory: the container held 3.9 GiB after the benchmark.

## Repeated on a quiet host, with the final compose file

After the session above the compose file changed twice, without changing what runs:

- **`--kv-cache-dtype auto`** is written out, with the reason for 16-bit (the profile check's
  hint `engine.kv-cache-dtype` asks for it).
- **A start without the Python files was tried and failed**: with the `code` download and
  `--trust-remote-code` removed, the engine stops at load with `Failed to load the processor.
  If the processor is a custom processor not yet available in the HuggingFace transformers
  library, consider … --trust-remote-code` (replica `8b6a1dc3`). The publisher's Python files
  stay.

Then the full benchmark was repeated with nothing else running on the host:
`benchmarks/2026-11-27-0637Z-llmperfbench.yaml` (its `compose_sha256` is the current
`compose.yaml`), 37 minutes, no failed request at any of 27 levels.

| Workload | 1 request | Plateau | At the plateau and above | First run (other GPU busy) |
|---|---|---|---|---|
| `max-prefill/1.0.0` | 9,758 prefill tok/s | 8 | 16,530–17,156 prefill tok/s | 9,373; 16,829–17,695 |
| `max-decode/1.1.0` | 150 decode tok/s, 6.5 ms per token | 12 | 597–616 decode tok/s | 152; 601–622 |
| `mixed-load/1.1.0` | 140 decode tok/s | 12 | 507–553 decode tok/s | 123; 508–543 |

The two runs agree within a few percent; `mixed-load` for one request is the largest difference
(140 against 123). `benchmarks/2026-11-27-0333Z-llmperfbench.yaml` is the first run, on the
compose file before the two changes.

GPU memory on this run: 95,493 MiB at the benchmark's peak, **96,453 MiB** after one
248,521-token prompt (found, 31 s), the highest value measured and the one in
`runtime.measured_gpu_memory_bytes_per_device` (`runtime_overhead` 14,594,709,915 bytes,
13.59 GiB). The hardware booking of 96,768 MiB is 315 MiB above it. Host memory: the
container held 6.0 GiB.

## Next

- Decide MTP on or off per use; try a shorter context with MTP for a larger pool.
- Compare with inclusionAI's FP8 release on two GPUs (the route without a third party).
- Check the answers through the API gateway.
