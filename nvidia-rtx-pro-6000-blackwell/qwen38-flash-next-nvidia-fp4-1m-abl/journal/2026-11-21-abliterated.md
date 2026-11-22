# 2026-11-21 Abliterated weights on the official one-GPU card

**Goal:** the official one-GPU card (deployment-catalog `449dbcb`) with the refusal-abliterated weights of
`onprem-ai--Qwen3.8-Flash-Next-FP4-abliterated-v2`.

Only the weights and the served model name change. The image (`20261121-baked-1gpu`, kernel caches of
this card), the command, NVIDIA's environment, the host-RAM booking (36 GiB) and the measured GPU memory
are the official card's.

**Not started with these weights.** The test host has 30 GB of RAM; this card needs 36 GiB (the PLE table in
host memory). Why it is expected to work as tested:

- The abliterated projections stay MXFP8 with NVIDIA's tensor names, dtypes and shapes; the checkpoint has
  the official one's size. The engine therefore loads it into the same GPU and host memory and selects the
  same kernels, which the image's one-GPU cache holds.
- The abliteration's effect and quality were measured with the same engine build, command and environment
  at TP=2: see the two-GPU abliterated card's journal of 2026-11-21 (harmful refusal 0% / 3.1% false
  positives, GSM8K 98.8%, needle 15/15 up to 766k tokens).

To do before `stable`: start it on a one-GPU host with ≥ 64 GB RAM, check no `b12x compiling` line, the peak
GPU memory against the card, and repeat the refusal probe and GSM8K.
