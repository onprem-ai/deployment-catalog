# 2026-11-11 - Aleph Alpha's Image, and the MoE Kernel as the Next Lever

**Goal:** Find why one user gets only about 160 tokens/s (Mellum2.1 FP8 on the same GPU:
about 420 without speculative decoding), and test Aleph Alpha's own serving image. Test
service with its own model name on inference node `nebius-e00pn5kkfnayh8wn6j` (one H200),
draft `drafts/kolibri-h200-tune`.

## Probes

`tmp/tune-model-fast` probe, engine reached directly: 20 s warm-up, 40 s window, about
410-token prompts, 1,024 tokens out, reasoning off (`reasoning_effort: none`). Answers: 40
short questions with known answers (English and German arithmetic, Python), 4 repeats each,
reasoning on, the model card's sampling.

| Setting | ITL p50, 1 request | decode tokens/s at 32 | decode tokens/s at 128 | Answers |
|---|---|---|---|---|
| This card (vLLM 0.33.0, Transformers backend, `hf` parsers) | 6.2 ms (161 tokens/s) | 2,990 to 3,080 | 6,090 to 6,590 | 40/40 |
| Aleph Alpha's image (vLLM 0.29.0 + aleph-alpha-inference 1.0.0, native `Kolibri1ForCausalLM`, `kolibri1` parsers) | 5.8 ms (172 tokens/s) | 2,780 to 3,040 | 6,860 to 7,000 | 40/40 |
| Aleph Alpha's image + `--moe-backend flashinfer_cutlass` | 11.2 ms (89 tokens/s) | 2,060 to 2,090 | 5,290 to 5,320 | 40/40 |

The native image is 6 to 13% faster and holds a similar KV cache (3,964,258 tokens); the
FlashInfer CUTLASS MoE backend halves the single-user speed. Both images choose the Triton
FP8 MoE backend by default.

## Why one user is slow

vLLM has no tuned Triton MoE configuration for Kolibri's expert shape on H200 and says so in
both images: `Using default MoE config. Performance might be sub-optimal! Config file not
found at .../fused_moe/configs/E=384,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`.
DeepGEMM's grouped MoE kernel is not used either (vLLM skips it for N <= 512).

One token takes 5.8 ms over 50 MoE layers (Mellum: 2.4 ms over 28 layers with 2.5B active
parameters). Reading Kolibri's 3.5B active FP8 weights takes well under 1 ms at the H200's
memory bandwidth, so the time goes to the kernels, not to memory: per layer the routed MoE,
the shared expert, attention and their launches. Kolibri ships no MTP head, so speculative
decoding cannot hide it.

## Ideas for further optimization

1. **Profile first (about 10 minutes, no restart).** A torch profiler trace of one
   single-user request (vLLM's `--profiler-config` or `VLLM_TORCH_PROFILER_DIR` with the
   `/start_profile` and `/stop_profile` endpoints) shows the share of each token spent in
   `fused_moe_kernel` against attention, the shared expert and launch gaps. Tune only if the
   MoE share is large.
2. **Tune the Triton MoE kernel for this shape.** vLLM's
   `benchmarks/kernels/benchmark_moe.py --model <Kolibri> --dtype fp8_w8a8 --use-deep-gemm
   false --tune`, with the GPU alone, for batch sizes 1 to 16 first (single user and small
   batches, about 1 hour), then up to 256. The earlier attempt next to a live replica took
   more than 35 minutes for one batch size. Ship the result as a container file and point
   `VLLM_TUNED_CONFIG_FOLDER` at it; measure ITL at 1, 32 and 128 before and after. Expected,
   not measured: typical MoE tuning gains of 10 to 30% on the MoE part, so roughly 190 to
   220 tokens/s for one user if the MoE dominates.
3. **Offer the result upstream.** A tuned `E=384,N=512,...H200,fp8_w8a8,block_shape=[128,128]`
   file belongs in vLLM's `fused_moe/configs`; once merged, every image gets it.
4. **Aleph Alpha's image for the card.** It is faster at every level measured and uses
   Aleph Alpha's own parsers, but pins vLLM 0.29 (the plugin supports one vLLM minor). Decide
   with a full benchmark on it, and check tool calls and long context again with the
   `kolibri1` parsers.
