# LingSante 3.1 Flash (MXFP4 experts, NEXTN, 4× GPU)

> ⚠️ **Not run, and it may not run.** Nobody reports this model on RTX PRO 6000 Blackwell. The
> publisher's recipe is for four 141 GB GPUs; here the weights take 73 of each card's 95 GiB.
> Whether its SGLang kernels work on these GPUs, and which image has them, is unknown.

> **Not a medical device.** The publisher releases this model for research and evaluation. It
> must not be used as an independent basis for diagnosis, prescriptions or treatment.

[`inclusionAI/LingSante-3.1-flash-MXFP4`](https://huggingface.co/inclusionAI/LingSante-3.1-flash-MXFP4),
the publisher's own 4-bit build of its medical model, on four RTX PRO 6000 Blackwell GPUs with
SGLang, tensor- and expert-parallel.

## The model

| | |
|---|---|
| Publisher | inclusionAI (Ant Group), MIT |
| Parameters | about 560B total, about 25B active per token |
| Architecture | hybrid-linear MoE: 63 KDA + 9 gated MLA layers, 512 routed experts (8 active), one MTP layer |
| Context | 262,144 tokens |
| This build | routed experts MXFP4 by round-to-nearest, no calibration; the rest FP8, BF16 and FP32 |
| Checkpoint | 153 safetensors files, 313.2 GB (291.7 GiB) |

The publisher's benchmark scores are for the unquantized source model, not for this build.

## Does it fit

| | Two GPUs | Four GPUs |
|---|---|---|
| GPU memory | about 190 GiB | about 380 GiB |
| Weights, 291.7 GiB | no | 72.9 GiB per GPU, if they divide evenly |
| Left per GPU | | about 22 GiB for KV cache, activations and graphs |

On paper it fits four. `--mem-fraction-static 0.88` gives the weights and the KV pool 83.6 GiB
per GPU: about 10 GiB of KV cache, 11 GiB outside the pool. The publisher's 0.80 would leave
3 GiB. These are estimates; the pool is in the startup log.

## What is unverified

- **The kernels on this GPU.** The MXFP4 card needs SGLang's Humming MoE backend
  (Humming 0.1.12) and names 4 × H20-3e. The source model's recipes name H200, H100, H800,
  B200 and GB300. Nothing names RTX PRO 6000 (SM120).
- **The image.** The MXFP4 card names none. This profile pins
  `lmsysorg/sglang:dev-ling-3.1-flash`, the image inclusionAI names for the FP8 source model;
  it may not contain the Humming backend or MXFP4 loading.
- **Memory.** Whether tensor parallelism divides everything by four, and what the KV cache
  costs per token.
- **Quality.** Round-to-nearest 4-bit without calibration, unmeasured by its publisher.

## Before this profile is more than a draft

1. Start it on four empty GPUs. An unknown `--moe-runner-backend humming` means the image is
   wrong; a kernel error names the GPU as the problem.
2. Read the KV pool from the log; adjust `--mem-fraction-static` and
   `--max-running-requests`.
3. Send requests with `chat_template_kwargs: {"enable_thinking": true}` and
   `temperature 1.0`, `top_p 0.95`, `top_k 20` (the publisher's recommendation).
4. Run the replica benchmark, commit its benchmark file, write the measured GPU memory.

## Checker

No errors. Warning: `model.weights-executable-files` (the publisher's two Python files, needed
by `--trust-remote-code`). Hints: `engine.kv-cache-dtype`,
`runtime.measured-gpu-memory-missing`.

## License

MIT, as the model card states; the repository carries no license file.
