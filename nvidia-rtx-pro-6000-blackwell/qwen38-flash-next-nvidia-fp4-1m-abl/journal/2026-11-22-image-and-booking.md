# 2026-11-22 Image and GPU booking of the official card's open PRs

- **Image:** `onpremai/vllm-qwen38-flash-next:20261121-1gpu@sha256:3973fb58…`, the image deployment-catalog PR #28 gives the official
  card (one neutral base image plus a kernel-cache layer per profile). Not started here (30 GB host RAM). #28 measured this image with NVIDIA's weights (5 min 35 s, no compiles); the
abliterated weights select the same kernels.
- **GPU booking:** `hardware.gpu.memory_bytes_per_device` 95 GiB -> 94 GiB as in deployment-catalog PR #26: an idle RTX
  PRO 6000 Blackwell shows 94.97 GiB free, so a 95 GiB booking marked every GPU not eligible. The abliterated
  weights take the official checkpoint's GPU memory, so the official measured peak applies.

#26 and #28 were merged into main on 2026-11-22 (53e187f, 5b553c7): this card matches the official one except for the
weights and the served model name.
