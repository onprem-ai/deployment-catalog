# 2026-11-21 Image with the kernel caches baked in: start in 7 minutes

**Goal:** Stop compiling kernels at every start of this two-GPU profile, without changing what the replica does.

The profile pinned `ghcr.io/local-inference-lab/vllm:release-10152026`, which compiles the engine's kernels at every start. The
kernels this profile needs are known (tensor-parallel size 2, `--max-num-seqs`, `--max-model-len`, the speculative decoding
settings, this GPU), so they can be in the image.

## What changed

Only `image:` in `compose.yaml`. The command, the environment, the resources and the figures in `ai-deploy.yaml` are unchanged.

`onpremai/vllm-qwen38-flash-next:20261121-2gpu-v2` is a neutral base image plus **one layer with the kernel caches this profile
needs**:

- `20261121-base` (7.578 GiB compressed): the pinned image without the files this model never loads, in one layer, with no caches.
  The one-GPU profile's image uses the same base.
- the cache layer (0.239 GiB compressed): every cache file a warmed replica of this profile holds.

A machine that runs both profiles downloads the base once: 7.578 + 0.128 (one-GPU layer) + 0.239 GiB, about 7.9 GiB, instead of
two images of about 7.7 GiB each. A change to the caches of one profile rebuilds one small layer in about a minute and uploads
0.2 GiB. How it is built: `dev/inference-engine-images/vllm-qwen38-flash-next/` in the platform repository.

## Measured

Two RTX PRO 6000 Blackwell Server Edition (GPUs 0 and 1 of a four-GPU machine, `iommu=pt`, driver 595.71.05), the profile's
settings, the same weights. Container start to "Application startup complete":

| Image | Start | Kernels at start-up |
|---|---|---|
| the one-GPU baked image (its caches do not cover tensor-parallel size 2) | 16 min 33 s | 538 compiled, 0 cached |
| base + a layer with only the cache files that differ from the one-GPU caches | 9 min 43 s | 304 cached, 121 compilations |
| `20261121-2gpu-v2`: base + a layer with all the cache files this profile needs | 7 min 9 s | 304 cached, 0 compilations |

The second row is why the layer holds every file a replica of this profile uses and not only the differences: the two-GPU start-up
reuses cache files it shares with the one-GPU profile. The start of the pinned upstream image on two GPUs was not measured. The
remaining 7 minutes are the weights (about 2.5 minutes), the engine's start and two CUDA graph captures (the model and the draft
model).

On `20261121-2gpu-v2`: a text request, a request with an image and a tool call all answer, and the log holds no missing-library
error. Under a load of three rounds of 16 concurrent requests (15 prompts of about 6k tokens and one of about 100k tokens, 1,500
output tokens each) a replica produced 707 to 725 output tokens per second; the highest GPU memory was 95,335 MiB per GPU
(93.1 GiB), the card's 93.11 GiB, so the card in `ai-deploy.yaml` stays. These load and memory figures were taken on the one-GPU
baked image with the same engine settings; `20261121-2gpu-v2` itself was checked with the three requests.

A kernel the cache does not hold is compiled at the start that needs it, as before. A change to the engine, the weights, the
tensor-parallel size, `--max-num-seqs`, `--max-model-len` or the speculative decoding settings needs the caches rebuilt.

## Registry

Docker Hub (`onpremai/`), public, as an experimental image. When this profile is stable, the image moves to
`platform.registry.onprem.ai/inference/nvidia-rtx-pro-6000-blackwell/` and the profile pins its digest.
