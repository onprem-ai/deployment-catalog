# 2026-11-26 Initial catalog entry (written from sources, nothing run)

**Hardware:** none used. The profile targets one RTX PRO 6000 Blackwell (96 GB, SM120).

## Sources read

- `thinkingmachines/Inkling-2-Mini` at `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`: model card,
  `config.json`, `generation_config.json`, `tokenizer_config.json`, file sizes from the Hugging
  Face API. 26.39B stored parameters: 20.38B 8-bit integers (packed NVFP4), 4.08B FP8, 1.93B
  BF16. `model_type` `inkling2_loop`, quantization `modelopt` / `NVFP4`, 128 experts with 8 per
  token, layers 6 to 17 run 3 times, sliding window 1,024 on 20 of 24 layers. No Python files
  in the repository.
- vLLM's recipe, `recipes.vllm.ai/thinkingmachines/Inkling-2-Mini` (pages for GB200, B200, B300
  and DGX Spark): requires vLLM >= 0.35.0 (nightly), `--tensor-parallel-size 1`,
  `--reasoning-parser inkling`, DSpark with 7 draft tokens; the `inkling` tool-call parser is
  opt-in; `--max-model-len 262144` on DGX Spark. Verified on GB200, B300, GB300, H100, H200 and
  DGX Spark. No RTX PRO 6000 entry, no memory figures beyond "a GB200 holds about 1.8
  concurrent 1M-token requests".
- Docker Hub: `vllm/vllm-openai:nightly` on 2026-11-26 is
  `nightly-5aea30579ad672c3c35be43112b7a1f7b293a09e`, manifest list
  `sha256:d24e5fc73f0b859bd76c9d2f6f6fa4da3001afd7637f2610876420f3027ea6c7`.

Not read: the SGLang cookbook, the publisher's blog post, the Hugging Face blog post.

## Decisions

- **One GPU.** The weights are 20.2 GB, and the recipe says more tensor-parallel ranks lower
  the throughput per GPU.
- **524,288 tokens per request,** from the recipe's GB200 figure scaled to a 96 GB card. Not a
  measurement.
- **16-bit KV cache,** as the recipe runs it. FP8 is the first thing to try.
- **Bookings 88 GiB (hardware) and 84 GiB (engine):** placeholders in the shape of the
  catalog's other one-GPU vLLM profiles.
- **`speculative_decoding: dspark`:** the recipe's method. That the speculator is
  `mtp.safetensors` is read from `config.json` (`mtp_config` with `block_size` and
  `markov_rank`), not stated by the recipe.
- **License file from apache.org:** the repository has none.

## Not done

No image pulled, no weights downloaded, no start, no benchmark, no measured GPU memory.

## Checker

Run with the control plane's `check_profile` (platform-alpha `b3643190`): no errors, no
warnings, 27 rules passed, 3 hints.
