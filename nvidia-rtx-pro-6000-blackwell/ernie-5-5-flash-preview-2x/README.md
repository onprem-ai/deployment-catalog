# ERNIE 5.5 Flash Preview (FP4 experts, DSpark, 300K context, 2× GPU)

> ⚠️ **Experimental: measured on one host, image in no registry.** On 2026-11-27 this profile
> ran on two RTX PRO 6000 Blackwell: answer checks, a full benchmark and long prompts
> (`journal/2026-11-27-tuning-and-benchmark.md`). Only a node that already holds the locally
> built image can deploy it (see "Image"), and the runtime includes a community patch whose
> repository carries no license.

[`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) on two
RTX PRO 6000 Blackwell GPUs, tensor-parallel, with vLLM 0.32.0 and Baidu's
[vLLM-ERNIE](https://github.com/baidu/vLLM-ERNIE) plugin. The command is the "V2" configuration of
the community SM120 port
[`hyudryu/ERNIE-5.5-Flash-Preview-2-RTXPro6000`](https://github.com/hyudryu/ERNIE-5.5-Flash-Preview-2-RTXPro6000)
at `8c5b856`, with 4 DSpark draft tokens instead of 7 and 12 sequences instead of 8.

## Does it fit

| Repository | Weights on disk | Two 96 GB GPUs |
|---|---|---|
| `ERNIE-5.5-Flash-Preview` (blockwise FP8, MXFP4 routed experts) | 165.9 GB (154.5 GiB) | yes: this profile |
| `ERNIE-5.5-Flash-Preview-FP8` | 301.9 GB | no |
| `ERNIE-5.5-Flash-Preview-BF16` | 599.6 GB | no |

The port's author reports 80.45 GiB of weights on each GPU, which leaves about 12.8 GiB per GPU
for activations, CUDA graphs and the KV cache. That is why the context per request is below the
model's 1,048,576 tokens:

| Port configuration | Tokens per request | KV pool | KV cache per GPU |
|---|---|---|---|
| Baseline (smallest patch) | 140,000 | 140,304 tokens | 5.83 GiB |
| V1 | 524,288 | 543,803 tokens | 7.82 GiB |
| **V2 (this profile)** | **300,000** | **1,213,175 tokens** | **9.02 GiB** |

V2's pool is larger than one full-context request, but its author did not test a cap above
300,000 with that memory budget; the longest prompt run was 498K tokens (V1, 119 s to first
token, peak 97,027 of 97,887 MiB). The full context needs four GPUs; a four-GPU profile
exists only as an untested draft and is not part of this catalog.

## Measured (2 × RTX PRO 6000 Blackwell Server Edition, 2026-11-27)

| | One request | Plateau |
|---|---|---|
| Real coding prompts (`coding-greedy`) | 157 tokens per second | 604–623 at 12 requests |
| Short prompts, long answers (`max-decode`) | 137 tokens per second | 594–611 at 12 requests |
| Mixed lengths (`mixed-load`) | 109 tokens per second | 355–416 at 24 requests |
| 8,192-token prompts (`max-prefill`) | 3,519 prefill tokens per second | 4,700–4,820 at 8 requests |

- No failed request in the benchmark (`benchmarks/2026-11-27-0017Z-llmperfbench.yaml`).
- Ready 6.6 to 7.1 minutes after a deploy, with the weights in the page cache.
- Answers: 24 of 24 known answers, every reasoning mode, parallel tool calls and an answer
  from their results; a hidden code found in prompts of up to 243,702 tokens (64 s), and in
  four 216,416-token prompts sent at once.
- GPU memory: 79.66 GiB of weights, 9.02 GiB of KV cache (1,213,175 tokens) and 2.15 GiB of
  CUDA graphs per GPU; peak 97,025 of 97,887 MiB after long prompts. The profile books the
  whole GPU.

## Reported by the port's author (not measured by onprem-ai)

V2, two RTX PRO 6000 Blackwell held at 450 W, 1,024 output tokens, mean of three workloads:

| Requests at once | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Decode, tokens per second (all requests) | 233 | 321 | 613 | 864 |

Prefill 5,000–6,700 tokens per second; 22 s to first token at 128K prompt tokens, 62 s at 289K.
The repository was two days old with no stars or issues when this profile was written
(2026-11-26): one person's report, not reproduced.

## Image

`onpremai/vllm-ernie55:0.32.0-ba322a5-sm120v2` is **in no registry yet**. `runtime/Dockerfile`
builds it (built once on the authoring host, 2026-11-26, 33.1 GB; imports checked, never started
with the model) from three pinned parts: the unmodified `vllm/vllm-openai:v0.32.0` image, Baidu's
plugin at `ba322a5` (Apache-2.0), and the port's four V2 files laid over the plugin
(`attention.py`, `moe.py`) and vLLM (`deep_gemm_moe.py`, `kv_cache_utils.py`).

- The stock plugin refuses to start on these GPUs: its attention runs on FlashMLA, which has
  SM90/SM100 kernels only. The port routes it to FlashInfer's SM120 sparse-MLA kernels.
- **The port's repository has no license file.** Its files are fetched at build time and are not
  in this catalog. Get the author's permission before pushing the built image to a public
  registry.
- After the push, pin the digest in `compose.yaml`
  (`onpremai/vllm-ernie55:0.32.0-ba322a5-sm120v2@sha256:…`).

## Before this profile is more than a draft

1. Push the image to a registry and pin that digest (the pinned one is a local build's), after
   clearing the SM120 port's license with its author.
2. Check the answers through the API gateway, and in the languages you serve.
3. Measure the probe's noise, then try 3 and 5 draft tokens and 16 sequences.
4. Try a request cap above 300,000 tokens: the KV pool holds 1.21M.

## Checker

No errors. Expected warnings: `engine.kv-cache-size-fixed` and `engine.gpu-memory-limit` (the KV
cache is sized in bytes on purpose, see `compose.yaml`) and `model.weights-executable-files` (the
tokenizer's three Python files). Deploy on GPUs nothing else runs on.

## License

Apache License 2.0 for the weights (`licenses/source-0-LICENSE.txt`). Baidu's card says the
license covers the Preview weights only; ERNIE 5.5 Flash Base stays under its own agreement.
