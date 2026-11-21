# 2026-11-21 Image with the kernel caches baked in: start in 5.5 minutes instead of 22

**Goal:** Stop compiling kernels at every start, and make the image smaller to download, without changing what the replica does.

The profile pinned `ghcr.io/local-inference-lab/vllm:release-10152026`. Its start compiled the engine's kernels each time
(`b12x compiling` lines, about 22 minutes from the container start to "Application startup complete"). The profile is
specific to one GPU and one model, so the kernels it needs are known: they are in the caches of a warmed replica.

## What changed

Only `image:` in `compose.yaml`. The command, the environment, the resources and the memory figures are unchanged.

`onpremai/vllm-qwen38-flash-next:20261121-s3-1gpu` is the pinned image plus:

1. the warmed caches of this profile (`/root/.cache/b12x`, `/root/.triton/cache`, `/root/.nv/ComputeCache`, about 680 MB),
2. without the files a warmed replica never loads (NVIDIA PyTorch container leftovers, TransformerEngine, apex, DALI,
   TensorRT, documentation), and
3. in one layer, so the removed files are not kept in the layers below.

How it is built, what was removed and why, and how to test it: `dev/inference-engine-images/vllm-qwen38-flash-next/`
in the platform repository.

## Measured

One RTX PRO 6000 Blackwell Server Edition, the profile's settings, the same host and driver for both images:

| | pinned image | this image |
|---|---|---|
| Container start to "Application startup complete" | about 22 min | 5.5 to 5.8 min |
| Download (compressed) | 12.0 GiB | 7.7 GiB |
| GPU memory at the peak | 93.32 GiB | 93.23 GiB |

The GPU memory is the card's, within measurement noise: the card in `ai-deploy.yaml` stays.
A kernel the cache does not hold is compiled at the start that needs it, as before. The cache depends on the engine's code,
the weights, `--max-num-seqs`, `--max-model-len`, the tensor-parallel size and the GPU, so a change to any of them needs a new image.

## Registry

Docker Hub (`onpremai/`), public, as an experimental image. When this profile is stable, the image moves to
`platform.registry.onprem.ai/inference/nvidia-rtx-pro-6000-blackwell/` and the profile pins its digest.
