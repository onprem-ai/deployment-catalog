# 2026-11-11 - Initial Catalog Entry for One H200

**Goal:** An experimental card for `JetBrains/Mellum2.1-12B-A2.5B-Thinking` on one H200,
with thinking and tool calls, as JetBrains documents it for vLLM.

## Research

The Hugging Face repository was inspected at commit
`92ddae9fc7665e9f801d141d2e5a6b2caf2460c4` (Apache-2.0, not gated). It holds five BF16
safetensors shards, the index, `config.json`, `generation_config.json`, the chat
template and the tokenizer: 24,308,172,412 bytes (22.6 GiB) for the files the card lists.
The README, the logo and a summary picture are left out.

`config.json` names `MellumForCausalLM`: 28 layers, all mixture-of-experts (64 experts, 8
active, 896 wide), hidden size 2,304, 32 query and 4 KV heads of 128, sliding window 1,024
on 3 of every 4 layers, YaRN for the full-attention layers up to 131,072 positions,
vocabulary 98,304. 12,149,923,072 parameters, 2.5B active.

vLLM v0.32.0 (the pinned Nemotron 3.5 image) registers `MellumForCausalLM`, checked in the
running image's model registry.

The model card's vLLM command with tool calling: `--max-model-len 131072
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes`. Its chat
template writes `<think>` blocks, `<tool_call>` JSON in Hermes form, and an empty
`<think></think>` when `enable_thinking` is false. Recommended sampling for thinking:
temperature 0.6, top_p 0.95, top_k 20. The MTP head for speculative decoding is announced,
not published yet.

## Decisions

- `experimental`: not deployed yet.
- Bookings for a small model on a large GPU: engine booking 56 GiB (vLLM share 0.39 of
  the H200), hardware booking 64 GiB. The weights take about 23 GiB; the rest is KV cache
  (about 33 GiB, a few hundred thousand tokens: KV heads are few and 3 of 4 layers keep
  only 1,024 tokens). The rest of the GPU stays free for another model. A measurement
  replaces these estimates.
- The checkpoint's `generation_config.json` sets no sampling, so
  `--override-generation-config` makes the model card's sampling the default.
- Prefix caching stated on.
- The smoke test turns thinking off, as on the other thinking models.
- The model card's MTP head is not out; speculative decoding follows when it is.

## Next

- Deploy on an H200, check thinking, a tool call and `/v1/messages`.
- Measure GPU memory and benchmark (skills `measure-gpu-memory`, `optimize-model`).
