# 2026-08-28 - Initial Catalog Entry

**Goal:** Add an experimental RTX PRO 6000 Blackwell catalog card for
`RadixArk/Qwen3.8-Flash-Next-NVFP4` using the current upstream checkpoint and
SGLang's documented day-0 serving configuration.

## Research

The Hugging Face repository was inspected at commit
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`. The weight index references 206
Safetensors files totaling 135,195,303,851 bytes (125.91 GiB). The checkpoint
contains a 125B main model plus a 51B FP8 PLE n-gram embedding table, with 6B
parameters active per token. It has 48 decoder layers, 512 routed experts with
top-10 routing, 262,144-token native context, vision support, and one MTP layer.
Only routed experts use NVFP4; attention, GDN, shared experts, vision, and other
excluded modules retain their checkpoint precision.

The quantized model card names SGLang as the supported runtime and identifies
Blackwell as the validated architecture. The official SGLang cookbook uses its
purpose-built `qwen38flashnext` image, FlashInfer linear attention, BF16
recurrent state, and optional NEXTN MTP. This card uses TP=2 so the complete
checkpoint, including the FP8 PLE table, remains on the two GPUs without host
RAM weight offload.

RTX PRO 6000 is SM120 rather than the B200/B300 SM100 hardware in the official
matrix. SGLang issues 36531, 36532, and 36701 document the SM120 differences.
The pinned day-0 image incorrectly gates its working TRT-LLM QSA decode kernel
to SM100. This card applies the narrow upstream PR 36556 guard change at
startup, with exact-match assertions so image drift fails visibly. It also uses
the independently validated SM120 combination of Triton GDN prefill,
FlashInfer GDN decode, and BF16 recurrent state rather than the cookbook's
all-FlashInfer combination.

## Decisions

- Marked the card experimental because this exact catalog deployment has not
  yet been downloaded and validated through the application APIs.
- Pinned both the Hugging Face revision and SGLang multi-architecture image
  digest.
- Listed every weight file referenced by the pinned index plus the runtime
  configuration, tokenizer, processor, and chat-template files.
- Used the source model's Qwen Community License 1.0 because the quantized
  repository explicitly delegates license terms to the source model.
- Configured the card explicitly for two GPUs with TP=2 and no PLE or model
  weight offload to host RAM.
- Used 76 GiB per GPU from the published TP=2 resident-memory range of about
  63-76 GiB. This is not claimed as a locally binary-searched minimum.
- Kept container host-memory requests at ordinary runtime-overhead levels;
  model weights and the FP8 PLE table belong in GPU memory.
- Enabled the in-checkpoint NEXTN configuration (3 steps, top-k 1, 4 draft
  tokens) and automatic reasoning/tool parser selection from the checkpoint.
- Omitted performance data because it has not been measured through this
  application's benchmark API on the target node.

## Sources

- https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4
- https://huggingface.co/api/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
- https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- https://www.lmsys.org/blog/2026-08-26-qwen-flash-next
- https://github.com/sgl-project/sglang/issues/36531
- https://github.com/sgl-project/sglang/issues/36532
- https://github.com/sgl-project/sglang/issues/36701
- https://github.com/sgl-project/sglang/pull/36556

## Outcome

Created the experimental catalog card, bundled the pinned source license, and
recorded the upstream rationale. Remaining work is to download, deploy, and run
chat, vision, tool-use, long-context, VRAM-bisection, and benchmark tests through
the application APIs before considering stable promotion.
