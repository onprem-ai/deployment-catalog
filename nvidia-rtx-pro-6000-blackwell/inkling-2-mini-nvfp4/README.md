# Inkling 2 Mini (NVFP4, DSpark, FP8 KV cache, 512K context)

> ⚠️ **Experimental: runs on a vLLM nightly, measured on one host.** On 2026-11-27 this profile
> started on an RTX PRO 6000 Blackwell (a GPU vLLM's recipe does not list), passed its answer
> checks and ran the full benchmark (`journal/2026-11-27-first-start-tuning-and-benchmark.md`).
> Another model was being measured on the host's other GPU at the same time.

[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on
one RTX PRO 6000 Blackwell with vLLM, following
[vLLM's recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini), with an FP8 KV cache.

## Measured (1 × RTX PRO 6000 Blackwell Server Edition, 2026-11-27)

| | One request | Plateau |
|---|---|---|
| Real coding prompts (`coding-greedy`, probe) | 283–286 tokens per second | 2,106 at 32 requests (not the limit) |
| Short prompts, long answers (`max-decode`) | 238 tokens per second | 2,342–2,376 at 40–48 requests |
| Mixed lengths (`mixed-load`) | 70 tokens per second (129 in the probe) | 1,835–1,912 at 80–112 requests |
| 8,192-token prompts (`max-prefill`) | 5,340 prefill tokens per second | 18,721–19,085 at 16–32 requests |

- No failed request in the benchmark (`benchmarks/2026-11-27-0308Z-llmperfbench.yaml`).
- Ready 4.3 to 5.1 minutes after a deploy.
- Answers: image input, 24 of 24 known answers, parallel tool calls and an answer from their
  results, a hidden code found in a 445,508-token prompt (78 s).
- GPU memory: 21.87 GiB of weights, a 55.45 GiB KV cache of 861,838 tokens; peak 85.10 GiB.

## The model

| | |
|---|---|
| Publisher | Thinking Machines Lab, released 2026-11-20, Apache 2.0 |
| Parameters | 26.4B total, 3.4B active (5.4B effective: the middle 12 of 24 layers run three times) |
| Input / output | text and images / text |
| Context | up to 1,048,576 tokens |
| Checkpoint | one 19.5 GB safetensors file, mixed NVFP4 and MXFP8, plus a 0.6 GB `mtp.safetensors` |
| Reasoning | always on; the minimum effort does not fully disable it |

The publisher's card supports DGX Spark and Apple silicon with 32 GB or more, and says other
GPUs are possible with 32 GB or more of VRAM.

## Does it fit

Yes: 20.2 GB of weights on a 96 GB card, but not with the model's full context.

- With 84 GiB for the engine, the KV cache holds 861,838 tokens (FP8; 760,811 at the recipe's
  16-bit). One request of 1,048,576 tokens does not fit, so this profile caps a request at
  524,288 tokens: 1.64 requests of that size.
- FP8 gains 13 % here, not a factor of two: the engine's KV layout pads its layer groups.
- The full context needs a second GPU; that is untried.

## Image

`vllm/vllm-openai:nightly-5aea30579ad672c3c35be43112b7a1f7b293a09e`, the nightly of 2026-11-26,
pinned by its multi-arch digest. The recipe requires vLLM 0.35.0 or newer, which is not
released, and names the moving `nightly` tag. This build serves the model on this GPU (SM120)
with driver 595.112.07. Move to the 0.35.0 release image when it exists.

## Before this profile is more than a draft

1. Repeat the full benchmark with the host's other GPU idle.
2. Move to the vLLM 0.35.0 release image.
3. Check the answers through the API gateway, and in the languages you serve.
4. Try `--max-num-seqs`, and two GPUs for the full 1,048,576-token context.

Per-request reasoning effort: the checkpoint's chat template maps `none` to `max` onto an
effort value, but the recipe documents no request parameter for it.

## Checker

No errors, no warnings. Hint: `engine.gpu-memory-share-written`.

## License

Apache License 2.0 (`licenses/source-0-LICENSE.txt`; the repository carries no license file,
its card links apache.org). The publisher also names a
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).
