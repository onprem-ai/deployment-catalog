# 2026-11-10 - Initial Catalog Entry

**Goal:** Add an experimental single-H200 catalog card for
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, following NVIDIA's
published vLLM recipe for one H200.

## Research

The Hugging Face repository was inspected at commit
`da5b2650a835292c678994a1fa3fb653250fb508`. It holds 88 safetensors shards,
the index, the custom model and processor code (`nemotron_h_omni`, C-RADIO
vision encoder), the chat template, the tokenizer and
`ultra_v3_reasoning_parser.py`. The files the card lists (everything except the
README, the model card subcards and the two benchmark plots) are
75,061,188,509 bytes (69.9 GiB).

The config identifies `NemotronH_Omni_Reasoning_V3`: 88 language layers (40
Mamba-2, 40 MoE, 8 attention), 512 routed experts with 22 active per token,
one MTP layer, 1,048,576 native positions. Mixed precision: NVFP4 for most
linear layers, FP8 for some, BF16 for the vision encoder, the connector, the
Mamba convolutions and the embeddings, FP32 for the MoE router.

The model card lists these vLLM deployments: 1x DGX Spark, 1x RTX PRO 6000,
2x H100, 1x H200 and 1x GB200. For one H200 it says: "use the same command
[as 2x H100] with `--tensor-parallel-size 1`", serving text, image and video
with 1M-token context and 3 MTP draft tokens. It names
`vllm/vllm-openai:v0.32.0` (or a pinned nightly).

The H200 is Hopper (SM 9.0) and has no FP4 tensor cores. The card states the
quantized model runs as W4A16 on non-Blackwell GPUs: the weights stay 4-bit in
memory, the matrix multiplies run in BF16. The profile gets the memory saving
(the model fits on one 141 GB GPU), not Blackwell's FP4 compute speed.

## Decisions

- Marked the card `experimental`: no deployment or inference has run on an
  H200 yet.
- Pinned the Hugging Face revision and the `vllm/vllm-openai:v0.32.0`
  multi-arch manifest
  (`sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`,
  published 2026-10-18, older than the 7-day window).
- Kept NVIDIA's flags unchanged, with three exceptions: the model path, the
  served name and port follow the catalog's conventions, and
  `--enable-prefix-caching` is written explicitly (NVIDIA's GB200 recipe for
  the same model sets it; the checker requires the choice to be stated).
- Booked 125 GiB of the H200's 140.4 GiB, NVIDIA's
  `--gpu-memory-utilization 0.89` in bytes. With no engine booking, the deploy
  writes vLLM's share from this booking, so the engine gets NVIDIA's 0.89. The
  15 GiB the booking leaves free on the GPU covers the overshoot an engine
  shows above its budget under load. A measurement sets the engine booking
  (format 1.1.0) and moves the hardware booking above it.
- Smoke test sends `enable_thinking: false`: the model reasons by default, and
  the default 5-token check would end inside the reasoning.
- License: OpenMDW-1.1. SPDX lists only OpenMDW-1.0 so far, so the expression
  is `LicenseRef-OpenMDW-1.1`.
- Omitted performance data until a benchmark on an H200 measures it.

## Next

- Deploy on an H200, confirm the model loads with vLLM v0.32.0, and run text,
  image, video and tool-call requests.
- Measure the GPU memory under load (`measure-gpu-memory` skill) and record
  `runtime.measured_gpu_memory_bytes_per_device` with a benchmark file.
- One H200 per replica is the intended layout: a second H200 runs a second
  replica, not tensor parallelism. A 2x H200 card is added only if a
  measurement shows a need one replica per GPU cannot meet.
