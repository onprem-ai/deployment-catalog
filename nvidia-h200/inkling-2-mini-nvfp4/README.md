# Inkling 2 Mini (NVFP4, DSpark, FP8 KV cache, 1M context)

[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on
one H200 with vLLM, following
[vLLM's recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini), which lists the H200
as verified. The whole card is booked and a request may use the model's full context.

## This profile

| | |
|---|---|
| GPU memory booked | 136 GiB, 128 GiB of it for the engine |
| KV cache (estimated) | about 99 GiB, about 1.5M tokens, FP8 |
| Tokens per request | 1,048,576 |
| CPU limit | 6 |
| Host memory limit | 20G |

## The model

| | |
|---|---|
| Publisher | Thinking Machines Lab, released 2026-11-20, Apache 2.0 |
| Parameters | 26.4B total, 3.4B active (5.4B effective: the middle 12 of 24 layers run three times) |
| Input / output | text and images / text |
| Context | up to 1,048,576 tokens |
| Checkpoint | one 19.5 GB safetensors file, mixed NVFP4 and MXFP8, plus a 0.6 GB `mtp.safetensors` |
| Reasoning | always on; the minimum effort does not fully disable it |

## Does it fit

Yes: 20.2 GB of weights on a 141 GB card.

- The estimate uses the engine's memory as measured on a 96 GB GPU: 21.87 GiB of weights,
  1.20 GiB of CUDA graphs, 5.5 GiB of other engine memory, and 15,543 tokens per GiB of FP8 KV
  cache. With 128 GiB for the engine that leaves about 99 GiB for the KV cache: about 1.5M
  tokens, 1.47 requests of the full context.
- If the engine refuses to start with `--max-model-len 1048576`, lower it to 524288.
- Above the engine's booking that measurement showed 3.9 GiB at 160 requests in flight; this
  profile leaves 8 GiB.

## Speed

Not measured. The checkpoint is NVFP4 and MXFP8, which the H200 does not compute natively.

## Image

`vllm/vllm-openai:nightly-5aea30579ad672c3c35be43112b7a1f7b293a09e`, the nightly of 2026-11-26,
pinned by its multi-arch digest. The recipe requires vLLM 0.35.0 or newer, which is not
released. Move to the 0.35.0 release image when it exists.

## Before this profile is more than a draft

1. Start it on an H200; fill in `runtime.measured_gpu_memory_bytes_per_device` and the KV
   cache's size from the engine's log; correct the two bookings and the context if needed.
2. Run the answer checks and the platform benchmark; commit the benchmark file.
3. One tuning round: FP8 against the 16-bit KV cache, and the number of DSpark draft tokens.
4. Measure host memory against the 20G limit.

## License

Apache License 2.0 (`licenses/source-0-LICENSE.txt`; the repository carries no license file,
its card links apache.org). The publisher also names a
[Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).
