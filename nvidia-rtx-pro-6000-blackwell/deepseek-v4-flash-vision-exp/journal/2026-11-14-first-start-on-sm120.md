# 2026-11-14 - First Start on Two RTX PRO 6000: Text Only, Without DSpark

**Goal:** Get the model running on two RTX PRO 6000 Blackwell GPUs (SM120) with vLLM's
`deepseekv4-flash-vision` image, as the initial entry planned.

## Setup

- Inference node `ain-4euk3`, both GPUs, tensor and expert parallel 2.
- Image `vllm/vllm-openai:deepseekv4-flash-vision` as pinned (vLLM
  `0.28.1rc1.dev137+g5ab628dd1`, FlashInfer 0.6.18).
- Weights downloaded: 156.3 GiB.

## Five failed starts, one cause each

1. **Adaptive verification.** The weights (80.12 GiB per GPU with the drafter) and the
   DSpark drafter loaded, then: "Adaptive verification trims verification requests on
   device, which the DeepseekV4IndexerBackend attention backend does not support".
   `enable_adaptive_verification` set to false.
2. **Custom all-reduce.** "Cuda error /workspace/csrc/custom_all_reduce.cuh:164 'invalid
   argument'" while capturing CUDA graphs. The two GPUs have no P2P link; the 0731 card
   passes `--disable-custom-all-reduce` for the same reason. Added.
3. **CUDA graphs.** "Check failed: (eidx.IsContiguous()) is false: eidx must be
   contiguous" inside FlashInfer's SM120 sparse MLA decode kernel
   (`sparse_mla_sm120.cu:225`) during graph capture. `--enforce-eager` added.
4. **Sparse MLA prefill.** "Unsupported sparse-MLA prefill configuration: model=DSV4
   num_heads=32 topk=512 page_block_size=64 topk_extra=512 extra_page_block_size=64".
   FlashInfer's SM120 DSV4 dual-cache prefill is built only for a sliding-window width of
   128 (`dispatch_dsv4_dual` in `sparse_mla_sm120_prefill.cu`). This model widens it to
   128 + 384 image tokens = 512. FlashMLA, the other DeepSeek V4 backend, is built only
   for sm_90a and sm_100 in this image. Removing DSpark did not change it.
5. **The width ignores `--language-model-only`.** With the flag vLLM logs "Disabled
   mm_prefix attention mode because multimodal inputs are configuration-disabled", but the
   sliding-window metadata builder (`vllm/v1/attention/backends/mla/sparse_swa.py`) still
   widens the index for the model's image tokens: vllm-project/vllm#56702, cause 1. A
   one-line patch at start makes the builder follow the deployment's setting, as the issue
   proposes; the command applies it and asserts it applied exactly once.

## Running

The sixth start reached ready: weights 73.83 GiB per GPU (without the drafter), KV cache
15.99 GiB, `GPU KV cache size: 582,697 tokens` (2.22 requests at 262,144 tokens), engine
share 0.9792 (93.0 GiB).

Answers through the API gateway: 26 of 26. Four known-answer questions three times without
reasoning and once at `high` and `max` reasoning effort; two parallel tool calls and an
answer from their results; a hidden number at 6,745, 28,480, 115,443 and 216,229 prompt
tokens. Unlike 0731 (journal 2026-11-13 of that card), it counted the primes below 30
right without reasoning.

## What this card is now, and is not

- **Text only.** `--language-model-only`: the vision encoder is not loaded, and image
  input is refused. The card's `capabilities.input` still names `image`; it changes once
  images run on SM120 or the card is renamed.
- **No speculative decoding.** DSpark adds a second index list to the same prefill kernel
  (step 4), so it is out until FlashInfer builds the shape.
- **No CUDA graphs** (`--enforce-eager`), so the decode is slower than it can be.

## Next

- Speed probes and the full benchmark of this configuration.
- Each of the three limits is a FlashInfer kernel gap on SM120 (sliding-window width 512
  prefill, contiguous expert indices in decode capture); a newer FlashInfer in a newer
  image is the way back to images, DSpark and CUDA graphs.

## Under load: the engine crashes

The probe (unique ~320-token prompts, 512-token answers) ran at 16 tokens/s for one
request (61 ms per token: no CUDA graphs, no DSpark) and at 101 and 229 tokens/s with 8
and 16 requests. At 24 and 32 requests the engine crashed:

```
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasGemmEx( ... CUDA_R_16BF ...
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

in the model's forward (`vl_model.py:283`, `model.py:1831`); the workers died and the
inference node restarted the replica. With `--max-num-seqs 16` it crashed the same way at
16, and on a fresh replica already at 2 requests at once: one request runs, two in the
same batch can corrupt memory. An illegal memory access is a kernel defect, not a
setting; it matches the prefill and decode kernel gaps above (FlashInfer's SM120 sparse
MLA for DeepSeek V4 is incomplete in this image).

## Outcome

The card is not deployable on RTX PRO 6000 with this image: it serves one request at a
time at 16 tokens/s and crashes with two. No benchmark was run. The service was removed
and the GPUs freed. The card stays `experimental` in the drafts catalog with this record.

## Next

- A newer vLLM image whose FlashInfer has the SM120 DSV4 prefill for width 512 and the
  decode capture fix (flashinfer-ai/flashinfer#5174 and #5244 are open on these shapes).
- Until then, DeepSeek V4 Flash 0731 (its own card, `onpremai/vllm-ds4f:0731-prebaked-v2`)
  is the DeepSeek that runs on these GPUs, at 1,418 tokens/s at 32 requests.
- A candidate, not tried yet: `vllm/vllm-openai:nightly-153dc046462c4724a9a4eb393cadd60169ed943c`
  (`sha256:cfa6d27b...`, 2026-11-13; vLLM 0.34.0rc1.dev398, FlashInfer 0.7.4). Its SM120
  sparse MLA prefill is rewritten (`sparse_mla_sm120/execution/prefill_launch.cuh`): the
  DSV4 dual-cache prefill dispatches by head count and extra page size with no fixed width
  of 128, and the sliding-window builder now widens the index only for the vision config
  class. FlashMLA stays sm_90a and sm_100 only. Trying it needs both GPUs of `ain-4euk3`.
