# Qwen3.8 Flash Next Abliterated (NVIDIA FP4, 1M context, 2× GPU)

> ⚠️ **Uncensored model.** Safety refusals have been substantially removed; it will attempt almost
> any request. Deploy it only where that is intended.

This card is the official `deployment-catalog` card
[`nvidia-rtx-pro-6000-blackwell/qwen38-flash-next-nvidia-fp4-1m-tp2`](https://github.com/onprem-ai/deployment-catalog/tree/5b553c7/nvidia-rtx-pro-6000-blackwell/qwen38-flash-next-nvidia-fp4-1m-tp2)
on `main` at `5b553c7` (which includes #26 and #28), with two changes: the weights and the served model name
(`qwen3.8-flash-next-abliterated`, so it can run next to the official model). The engine
command, NVIDIA's environment, the resources and the measured memory are the official card's.

## Weights

OnPrem AI model storage `onprem-ai--Qwen3.8-Flash-Next-FP4-abliterated-v2`: NVIDIA's
[`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) at
`b9ce18a6887e18c089fdea8509a33e63429a3300` (the revision the official card pins) with **97 of 375,487
tensors changed**: the weights and block scales of the 48 text attention output projections
(36 `linear_attn.out_proj`, 12 `self_attn.o_proj`) and the MTP drafter's `self_attn.o_proj`.

- Refusal-direction abliteration (Arditi et al. 2024): per layer `W' = W − λ·r̂(r̂ᵀW)`, r̂ the
  difference of means of the last-token attention outputs on 256 AdvBench harmful vs 256 Alpaca
  harmless prompts, chat and thinking mode merged. The MTP drafter gets the layer-47 direction.
- **The projections stay MXFP8**, with the same names, dtypes and shapes as NVIDIA's, so the
  engine runs exactly the kernels of the official model. They were re-quantized with NVIDIA's own
  E8M0 block scales (unedited values re-encode to their identical bytes; 3% of the blocks, where the
  edit would overflow E4M3, got the smallest sufficient raised scale). Rounding keeps 96.4% of the
  edit, so it was baked at λ = 2.08 for an effective λ = 2.006 (the λ = 2.0 the sweep selected).
- Every other tensor, `config.json` and the 19 other shards are NVIDIA's, byte for byte
  (`verify_report.json`, `abliteration_report.json` in the repository). Same size on disk and in GPU
  memory as the official checkpoint.

## Validation

Held-out prompts (never used for the directions), the two-GPU card's configuration of NVIDIA's
engine (TP=2, 1M context, FP8 KV cache, MTP ×3, CUDA graphs), weights bind-mounted, 2026-11-21:

| | NVIDIA's model | This model |
|---|---|---|
| Harmful refusal, chat / thinking (AdvBench, n=64 each) | 100% / 100% | **0% / 3.1%**¹ |
| Benign refusal, chat / thinking (Alpaca, n=32 each) | 3.1% / – | 0% / 0% |
| Broken or empty answers (not a token-limit cut-off) | 0% | 0% |
| GSM8K, first 250, greedy, thinking | 98.4% | **98.8%** |
| Needle at 27k / 109k / 204k / 426k / 766k tokens, 3 depths each | – | **15 / 15** |
| Tool call (`qwen3_xml`), chat and thinking; image input | – | correct |
| MTP acceptance per draft token | – | 56.2% (2.69 tokens per step) |

¹ both "I can't generate a video file directly, but here's a complete script…" — answered in full;
marker matching counts the opening phrase. NVIDIA's model column: same prompts and harness on a
vLLM nightly (eager mode).

Refusals are marker-matched on the answer (reasoning excluded). Not measured: tool-calling and image
quality beyond a smoke test, benchmarks beyond the GSM8K subset, PII/self-harm categories separately.
Sample sizes are small: 0 of 64 has a 95% upper bound of about 6%. Thinking on harmful prompts runs
longer than on the base model; give thinking requests a generous `max_tokens`.

## Image

`onpremai/vllm-qwen38-flash-next:20261121-2gpu-v2`, the image deployment-catalog the official two-GPU card uses (deployment-catalog #28): one neutral base image (the pinned `ghcr.io/local-inference-lab/vllm:release-10152026` without the
files this model never loads) plus this configuration's kernel-cache layer. The abliterated projections stay
MXFP8, so this card runs exactly the official card's kernels. Tested 2026-11-22 on two RTX PRO 6000 Blackwell, fresh
container: ready 242 s after start, no `b12x compiling` line at the start or after text, image and tool requests,
51.16 GiB of weights per GPU.

## Licenses

[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)
(`licenses/source-1-NVIDIA-Open-Model-Agreement.txt`) for the FP4 checkpoint and its derivatives; the base
model under the Qwen Community License 1.0 (`licenses/source-0-LICENSE.txt`). The weights repository carries
`LICENSE`, `LICENSE-Qwen` and the `NOTICE` the agreement requires.
