# Ling 3.2 Air (third-party NVFP4 experts, MTP, 256K context, 1× GPU)

> ⚠️ **Experimental, and the weights are a third party's.** inclusionAI publishes this model in
> FP8 only. The NVFP4 build is primitive-ai's. On 2026-11-27 this profile started on an RTX PRO
> 6000 Blackwell, passed its answer checks and ran the full benchmark, the second time with
> nothing else on the host (`journal/2026-11-27-first-start-tuning-and-benchmark.md`).

[`inclusionAI/Ling-3.2-air`](https://huggingface.co/inclusionAI/Ling-3.2-air) in the NVFP4 build
[`primitive-ai/Ling-3.2-air-NVFP4`](https://huggingface.co/primitive-ai/Ling-3.2-air-NVFP4), on
one RTX PRO 6000 Blackwell with vLLM, the model's 262,144-token context, MTP speculative
decoding.

## Measured (1 × RTX PRO 6000 Blackwell Server Edition, 2026-11-27)

| | One request | Plateau |
|---|---|---|
| Real coding prompts (`coding-greedy`, probe) | 174 tokens per second | 605 at 8 requests; 686 at 32 |
| Short prompts, long answers (`max-decode`) | 150 tokens per second | 597–616 at 12 requests |
| Mixed lengths (`mixed-load`) | 140 tokens per second | 507–553 at 12 requests |
| 8,192-token prompts (`max-prefill`) | 9,758 prefill tokens per second | 16,530–17,156 at 8 requests |

- No failed request in the benchmark (`benchmarks/2026-11-27-0637Z-llmperfbench.yaml`, on a quiet
  host; an earlier run with the other GPU busy agrees within a few percent).
- Ready 3.3 to 4.4 minutes after a deploy.
- Answers: image input, 24 of 24 known answers, thinking on and off, parallel tool calls and
  an answer from their results, a hidden code found in a 249,609-token prompt (31 s).
- GPU memory: 72.90 GiB of weights and a 6.49 GiB KV cache of 421,486 tokens; peak 94.19 GiB.

**MTP is a choice.** With it, one request decodes 174 tokens per second instead of 105 and
eight requests 605 instead of 460. It costs KV cache: the pool holds 421,486 tokens (1.6
requests of the full context) instead of 1,430,200, and at 32 requests the throughput is 686
instead of 923. For many requests at once, or several long ones, delete the
`--speculative-config` line in `compose.yaml`.

## What is downloaded from where

| Download | Repository | Files |
|---|---|---|
| `main` | `primitive-ai/Ling-3.2-air-NVFP4` at `a01b9c21` | 44 `.safetensors`, 7 `.json`, `chat_template.jinja`. **No Python file.** |
| `code` | `inclusionAI/Ling-3.2-air` at `ff099487` (the publisher) | the model's 7 `.py` files |

- The model needs `--trust-remote-code`: its configuration, image processor and model class are
  Python files beside `config.json`. They are taken from the publisher's repository, not from
  the third party's. On 2026-11-26 the seven files have the same git blob id in both.
- Of primitive-ai's non-tensor files, `chat_template.jinja`, `tokenizer.json`,
  `tokenizer_config.json`, `generation_config.json`, `preprocessor_config.json` and
  `special_tokens_map.json` are also identical to the publisher's. `config.json` (the
  compressed-tensors quantization block) and `model.safetensors.index.json` differ, as a
  re-quantized build must.
- Safetensors files hold tensors only and cannot run code. The publisher's Python files were
  scanned (imports and risky calls), not read line by line: model code on `transformers` and
  `torch`, and `requests.get` only to fetch an image or audio URL given as input.

## The build, as primitive-ai reports it

- 77.9 GB against 129.3 GB for the FP8 release. The 512 routed experts of every MoE layer are
  NVFP4 (weights and, at runtime, activations); attention, shared experts, dense MLPs, routers,
  vision tower, embeddings and LM head are BF16.
- Made from the FP8 release (inclusionAI publishes no BF16), so its accuracy is measured
  against FP8: top-1 agreement 95.9%, and knowledge / tool-call / abstain scores of 89.9 / 79.8
  / 80.0 against FP8's 90.3 / 75.8 / 79.2, which it calls level within run-to-run noise.
- One RTX PRO 6000 Blackwell, 32,768-token window, `--gpu-memory-utilization 0.92`, no MTP: KV
  pool 1,022,829 tokens, 100.6 tokens per second for one request, 593 at 32 at once.

primitive-ai is a verified organisation on Hugging Face, which says who it is, not how good the
build is. Its accuracy comparison is not reproduced; its speed without MTP is (105–110 tokens
per second for one request here). The publisher's own FP8 release (120 GiB) fits two of
these GPUs and is the route without a third party.

## What the first runs showed

- **The image.** primitive-ai's card names a `nightly-a2bcb565…` tag that Docker Hub does not
  have; this profile pins `cu134-nightly-a2bcb565…`, the tag that exists for that vLLM commit.
  It serves the model.
- **The context needs `--max-num-seqs`.** With vLLM's default of 1,024 sequences the engine
  ran out of memory at 262,144 tokens and refused to start at 65,536; with 64 sequences the
  full context starts.
- **No FP8 KV cache.** With `--kv-cache-dtype fp8` the engine does not start on this GPU
  (a Triton kernel needs more shared memory than the GPU has).
- **The split model directory works, and the Python files are needed.** The two downloads are
  linked into one directory at start. Without the publisher's Python files and
  `--trust-remote-code` the engine fails to load the image processor.

## Before this profile is more than a draft

1. Compare its answers with inclusionAI's FP8 release (two GPUs): the build's accuracy
   figures are its maker's.
2. Decide MTP on or off for your load (see above).

## Checker

No errors. Warning: `model.weights-executable-files` (the publisher's seven Python files).
Hint: `engine.gpu-memory-share-written`.

## License

MIT, the base model's license, kept by the NVFP4 build. Neither repository carries a license
file; the cards state it.
