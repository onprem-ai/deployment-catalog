# ERNIE 5.5 Flash Preview (FP4 experts, DSpark, 1M context, 4× GPU)

> ⚠️ **Not run, and nobody else has run it either.** This profile is an
> extrapolation of the two-GPU profile to four GPUs. Its memory figures are estimates. The
> profile cannot deploy until the image is in a registry (see "Image").

[`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview) on four
RTX PRO 6000 Blackwell GPUs, tensor-parallel, with vLLM 0.32.0 and Baidu's
[vLLM-ERNIE](https://github.com/baidu/vLLM-ERNIE) plugin. Same image and flags as
[`ernie-5-5-flash-preview-2x`](../ernie-5-5-flash-preview-2x), with three changes:
`--tensor-parallel-size 4`, a 32 GiB KV cache per GPU, and the model's full 1,048,576-token
context per request.

## Why four GPUs

On two GPUs the weights take 80.45 GiB of each 94.97 GiB card and leave 9 GiB per GPU for the
KV cache: 300,000 tokens per request. Four GPUs halve the weights per card.

| Per GPU | Two GPUs (reported by the SM120 port) | Four GPUs (estimate) |
|---|---|---|
| Weights | 80.45 GiB | about 40 GiB |
| CUDA graphs, activations, overhead | about 5 GiB | about 5 GiB |
| KV cache | 9.02 GiB | 32 GiB (set) |
| Peak | 94.75 GiB | about 78 GiB |
| KV pool | 1,213,175 tokens | about 4.3M tokens, if the cost per token per GPU stays the same |

The booking is 94 GiB per GPU, the whole card, until the peak is measured. If the measured peak
is near 78 GiB, either lower the booking or raise `--kv-cache-memory-bytes`.

The larger repositories also fit four GPUs on paper (`-FP8`: 301.9 GB of 380 GiB) but leave
little for the KV cache; this profile keeps the default repository, whose MXFP4 experts are
quantization-aware trained.

## What is unverified

- **Four tensor-parallel ranks on SM120.** The port's kernels, its padded-row fix under CUDA
  graphs and the FlashInfer PCIe all-reduce were run with two ranks only.
- **The KV pool size.** It assumes the KV cache is replicated on each rank, not divided between
  them. Read `GPU KV cache size` from the startup log.
- **A 1M-token request.** The longest prompt anyone reports on these GPUs is 498K tokens (two
  GPUs, 119 s to first token). Expect minutes of prefill for a cold 1M prompt.
- **Speed.** No figure exists for four GPUs. Decode on two GPUs is bound by memory bandwidth
  and 9–21% of a step is cross-GPU all-reduce, so four GPUs may not be faster per request.
- **Host memory.** The limits in `compose.yaml` are guesses for four workers.
- **Baidu's own layout** (`--data-parallel-size 4 --enable-expert-parallel`) may suit four GPUs
  better than TP4. It targets data-centre Blackwell and has not been tried on SM120.

## Image

`onpremai/vllm-ernie55:0.32.0-ba322a5-sm120v2`, the two-GPU profile's image, is **in no registry
yet**. `runtime/Dockerfile` builds it (built once on the authoring host, 2026-11-26; imports
checked, never started with the model): the unmodified `vllm/vllm-openai:v0.32.0` image, Baidu's
plugin at `ba322a5` (Apache-2.0), and the four V2 files of the community SM120 port
[`hyudryu/ERNIE-5.5-Flash-Preview-2-RTXPro6000`](https://github.com/hyudryu/ERNIE-5.5-Flash-Preview-2-RTXPro6000)
at `8c5b856`.

- **The port's repository has no license file.** Its files are fetched at build time and are not
  in this catalog. Get the author's permission before pushing the built image to a public
  registry.
- After the push, pin the digest in `compose.yaml`.

## Before this profile is more than a draft

1. Bring up the two-GPU profile first: it reproduces a reported configuration.
2. Start this one on four empty GPUs. If it faults at the first request, remove
   `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` (back to NCCL) before changing anything else.
3. Read the KV pool from the log, send one 1M-token prompt, then several long prompts at once.
4. Run the replica benchmark, commit its benchmark file, write
   `runtime.measured_gpu_memory_bytes_per_device`, and set the bookings from the peak.

## Checker

No errors. Expected warnings: `compose.images-pinned` (no image yet),
`engine.kv-cache-size-fixed` and `engine.gpu-memory-limit` (the KV cache is sized in bytes on
purpose, see `compose.yaml`), `model.weights-executable-files` (the tokenizer's three Python
files). Deploy on GPUs nothing else runs on.

## License

Apache License 2.0 for the weights (`licenses/source-0-LICENSE.txt`). Baidu's card says the
license covers the Preview weights only; ERNIE 5.5 Flash Base stays under its own agreement.
