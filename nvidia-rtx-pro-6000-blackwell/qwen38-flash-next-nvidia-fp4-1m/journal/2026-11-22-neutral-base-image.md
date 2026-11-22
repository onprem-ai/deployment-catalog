# 2026-11-22 One-GPU image on the shared base: the same caches, one cache layer

**Goal:** Let this profile's image and the two-GPU profile's image share their large layer.

## What changed

Only `image:` in `compose.yaml`, from `onpremai/vllm-qwen38-flash-next:20261121-baked-1gpu` to
`onpremai/vllm-qwen38-flash-next:20261121-1gpu`. The kernel caches in the image are the same ones; what differs is the layering:

| | `20261121-baked-1gpu` | `20261121-1gpu` |
|---|---|---|
| Layers | one 7.706 GiB layer (the files without the unused libraries, and the caches) | the neutral base `20261121-base` (7.578 GiB) + one cache layer (0.128 GiB) |
| Download | 7.71 GiB | 7.71 GiB |

The two-GPU profile's image (`20261121-2gpu-v2`) is built on the same base, so a machine that runs both profiles downloads the base
once. How it is built: `dev/inference-engine-images/vllm-qwen38-flash-next/` in the platform repository.

## Measured

One RTX PRO 6000 Blackwell Server Edition, the profile's settings. Container start to "Application startup complete": 5 min 35 s,
with every kernel of the start-up taken from the cache (124 + 10 cached, 0 compilations); the previous image started in
5.5 to 5.8 minutes. A text request, a request with an image and a tool call all answer, and the log holds no missing-library error.
GPU memory was not measured again: the files in the image are the ones of the previous image.
