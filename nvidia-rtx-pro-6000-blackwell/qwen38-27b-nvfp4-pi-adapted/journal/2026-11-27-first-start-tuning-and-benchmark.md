# 2026-11-27 First start with the LoRA, answer checks, three tuning rounds, full benchmark, GPU memory

**Hardware:** host `ain-xtg6o`, one NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB;
the host's second GPU, index 1), 30 GB of RAM, driver 595.112.07. **Engine:** vLLM 0.33.0,
image `vllm/vllm-openai:v0.33.0@sha256:d825baf9…` (the `qwen38-27b-nvfp4` profile's). Deployed
through the control plane's development stack from the `drafts` catalog, inference service
`5b7b5ba7-fc2b-4ad1-8c75-78edd572108e`.

The baseline probe ran while another model was being benchmarked on the host's other GPU;
rounds 1 to 3 and the full benchmark ran with the other GPU idle.

**Precision:** base weights Unsloth's NVFP4 build (as the template profile); the LoRA BF16;
KV cache FP8 (`--kv-cache-dtype fp8`, as the template); the MTP head as the checkpoint ships
it.

## First start: the LoRA on the NVFP4 base

The open question of the draft: does vLLM apply a LoRA to a compressed-tensors NVFP4 base.
It does. Ready 3.3 minutes after the deploy; the log says `Loaded new LoRA adapter: name
'qwen3.8-27b-pi', path '/models/adapter'` and `Using default LoRA kernel configs`.
`/v1/models` lists `qwen3.8-27b-pi` with parent `qwen3.8-27b-pi-base`.

Warnings in the start-up log: `--mamba-cache-mode is deprecated … It has no effect` (removed
from the command later); `Qwen3.5 model specifies mamba_ssm_dtype='float32' … Using the
user-specified value` (the template's `bfloat16`); `vLLM has deprecated support for
supporting different tokenizers for different LoRAs` (the base model's tokenizer is used;
the adapter's `tokenizer.json` is the base model's).

## Does the adapter change the answers

Three agent-style prompts, temperature 0, the adapter's name against the base model's, on
the first settings and again on the final ones (MTP on): the answers differ on 3 of 3 both
times. On the first settings the adapter answered "List the first three tool calls you
would make" with `read_file`, `grep`, `run_tests` in 8 tokens, the base model with three
sentences in 52. On the final settings both wrote three sentences, differently worded.
This shows the adapter is applied; it does not show how much of its published gain
(SWE-bench Pro 39.2 % → 64.7 % under the pi harness, on the BF16 base) survives the
quantized base. That needs the pi harness and is not measured.

## Answer checks (the adapter's name, straight to the replica's port)

| Check | First settings (no MTP) | Final settings (MTP 4) |
|---|---|---|
| 8 questions with known answers, 3 times each, temperature 0.7 | 24 / 24 | 24 / 24 |
| Thinking: default and `enable_thinking: false` | off: 0 reasoning tokens | off: 0 |
| `reasoning_effort: low` | on: 173 reasoning tokens, correct | on: 204, correct |
| `reasoning_effort: high` | refused (400): "Supported types are xhigh (default), medium, and low" | the same |
| Two tool calls in one turn; answer from both results | yes; yes | yes; yes |
| Hidden code | found at 7,975 / 61,992 / 251,758 prompt tokens (1.0 / 13.2 / 151 s) | found at 61,974 / 251,760 (13.9 / 161 s) |

- The profile is text only (`--language-model-only`): no image check.
- A 252K-token prompt prefills at about 1,600 tokens per second, a 62K-token one at about
  4,500. Not investigated; the template profile's long-prompt speed without a LoRA was not
  measured tonight for comparison.

## The probe

`replicas_benchmark_start` with `concurrency_levels: [1, 8, 32]` and `requests_per_level: 8`,
`coding-greedy/1.0.0`; `measure_while_serving_clients` true (the control plane counts every
ready replica as serving clients; the service had no other caller). One probe per round.

| Round | Change | coding-greedy decode tok/s at 1 / 8 / 32 | ITL p50 ms | Acceptance by position | Kept |
|---|---|---|---|---|---|
| 0 | baseline: the LoRA, no MTP (other GPU busy) | 62 / 450 / 1,467 | 16.1 / 17.7 / 22.0 | | |
| 1 | MTP, `num_speculative_tokens` 2 | 124 / 949 / 2,611 | 8.0 / 7.8 / 12.4 | 0.92, 0.80 | yes |
| 2 | 2 → 3 | 140 / 1,077 / 2,869 | 6.9 / 6.6 / 11.4 | 0.92, 0.81, 0.70 | yes |
| 3 | 3 → 4 | 155 / 1,107 / 2,754 | 6.3 / 5.8 / 11.2 | 0.93, 0.84, 0.75, 0.67 | yes |

- Round 1: vLLM runs the base model's MTP head together with the LoRA, and the head's drafts
  are accepted as often as for the base model. Twice the speed at every level.
- Rounds 2 and 3: each token more is accepted often enough to pay for one request (+13 %,
  +11 %); at 32 requests 4 tokens are 4 % below 3, inside what one probe can tell. 4 was
  kept for the single-request speed. 5 was not tried.
- KV pool: 2,011,652 tokens without MTP, 1,759,036 with 4 draft tokens.
- No failed request and no engine error in any round; every redeploy ready in 3.7 to 3.8
  minutes.

Not tried: 5 draft tokens, the BF16 base (the adapter's own recipe), the adapter merged
into the base and re-quantized, vision (the adapter's evaluations are text only).

## Full benchmark

`benchmarks/2026-11-27-0444Z-llmperfbench.yaml`, on the committed settings (its
`compose_sha256` is this profile's `compose.yaml`), load generator `dev-892868be`, 53 minutes,
the other GPU idle, no failed request at any of 38 levels.

| Workload | 1 request | Plateau | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 6,018 prefill tok/s | 12 | 10,053 prefill tok/s at 32 |
| `max-decode/1.1.0` | 136 decode tok/s, 7.5 ms per token | 96 | 3,529 decode tok/s at 96 |
| `mixed-load/1.1.0` | 109 decode tok/s | 48 | 1,226 decode tok/s at 80 |

The benchmark's random-word prompts are drafted less well (acceptance 0.75, 0.56, 0.34,
0.24) than real coding prompts (0.93, 0.84, 0.75, 0.67): 136 against 155 tokens per second
for one request.

## GPU memory

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 23,310,934,999 | 21.71 | `Model loading took 21.71 GiB memory` (base, MTP head and LoRA) |
| `cuda_graphs` | 1,052,266,987 | 0.98 | `0.98 GiB for CUDAGraph memory` |
| `kv_cache` | 64,682,207,477 | 60.24 | `Available KV cache memory: 61.22 GiB` minus the graphs |
| `runtime_overhead` | 11,721,695,561 | 10.92 | the peak minus the parts above |
| **Peak** | 100,767,105,024 | 93.85 | 96,099 MiB, `nvidia-smi` every 30 s during the repeated benchmark |

- KV cache: 1,759,036 tokens (`GPU KV cache size`), 6.71 requests of 262,144 tokens.
- The engine's log: `Desired GPU memory utilization is (0.9265, 87.99 GiB). Actual usage is
  22.12 GiB for consumed memory (weights + non-torch), 4.65 GiB for peak activation, and
  0.98 GiB for CUDAGraph memory`. The engine booking is 88 GiB; the peak is 4.82 GiB above
  it. The hardware booking is 94 GiB: 1.18 GiB of headroom.
- **Host memory is at its limit.** After the long prompts the container held 15.64 GiB of
  its 16 GiB limit (the template's). Raise `limits.memory` before this profile serves long
  prompts; it was left as measured so that the benchmark's `compose_sha256` still names the
  compose file.

## Repeated with the raised memory limit

The container's memory limit went from the template's 16G to 24G (`host_bytes` 24 GiB),
because it had held 15.64 GiB after long prompts. That changes `compose.yaml`, so the full
benchmark was repeated on it, with nothing else running on the host:
`benchmarks/2026-11-27-0727Z-llmperfbench.yaml` (its `compose_sha256` is the current
`compose.yaml`), 53 minutes, no failed request at any of 38 levels.

| Workload | 1 request | Plateau | Best | First run (16G limit) |
|---|---|---|---|---|
| `max-prefill/1.0.0` | 5,875 prefill tok/s | 12 | 10,061 prefill tok/s at 12 | 6,018; 10,053 |
| `max-decode/1.1.0` | 132 decode tok/s | 96 | 3,479 decode tok/s at 96 | 136; 3,529 |
| `mixed-load/1.1.0` | 112 decode tok/s | 64 | 1,200 decode tok/s at 80 | 109; 1,226 |

The two runs agree within 3 %. `benchmarks/2026-11-27-0444Z-llmperfbench.yaml` is the first
run, on the compose file with the 16G limit.

After the benchmark: one 251,628-token prompt (found, 161 s), then two 125K-token prompts at
once (found, 49 and 91 s).

- **Host memory:** the container held 12.83 GiB of 24 GiB afterwards.
- **GPU memory:** peak **96,099 MiB** during this benchmark (95,047 MiB in the first run), the
  value now in `runtime.measured_gpu_memory_bytes_per_device` (`runtime_overhead`
  11,721,695,561 bytes, 10.92 GiB). That is 157 MiB under the template's 94 GiB hardware
  booking, so the booking is raised to 96,768 MiB (101,468,602,368 bytes): 669 MiB above the
  peak, 483 MiB below what an idle GPU has free.

## Through the API gateway, and one unstable answer

Checks sent through the API gateway (`https://<control plane>/v1`, an API gateway key), on
the compose file with the 24G limit:

- 24 of 24 known answers; two parallel tool calls and an answer from both results; a hidden
  code found at 61,974 prompt tokens; `reasoning_effort: low` correct.
- The gateway routes `qwen3.8-27b-pi` to the LoRA module (the answer's `model` field says
  so). `qwen3.8-27b-pi-base` is not routable: `The API gateway has no route for model
  'qwen3.8-27b-pi-base'`. That is what the profile intends.

**One question is answered unstably by the adapter with thinking off.** "A train leaves at
14:35 and arrives at 17:10. How many minutes does the trip take? Number only." (155), at
temperature 0:

| Path | Model | Answers |
|---|---|---|
| replica's port, three earlier replicas | adapter | 155, 155, 155, 155 |
| API gateway, first check | adapter | 95, 95 |
| replica's port, same replica as the next row | adapter | 95, 95 |
| API gateway | adapter | 95, 155 |
| replica's port, same replica | base model | 155, 155 |

- The adapter is right in 5 of 10 tries, on both paths, so the path is not the cause; the
  base model on the same replica is right in 2 of 2. With `reasoning_effort: low` the adapter
  answers 155 every time.
- The answers of one request at temperature 0 differ between tries: the engine with MTP is not
  bit-reproducible, and this answer sits on an edge.
- One question, and not the adapter's domain (it was trained on agent tasks). But it is the
  only direct sign so far that the adapter without thinking, on the NVFP4 base, can be worse
  than the base. It is not known whether the BF16 base behaves the same.

## Next

- Run the pi harness against both names (SWE-bench Pro hard 51 or Terminal-Bench 2.1) to
  see what the quantized base costs the adapter.
- 5 draft tokens; the long-prompt prefill speed with and without the LoRA.
- The Anthropic API path.
