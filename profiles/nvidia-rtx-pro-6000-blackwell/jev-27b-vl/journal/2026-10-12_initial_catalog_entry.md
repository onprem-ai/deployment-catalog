# 2026-10-12 - Initial Catalog Entry

**Goal:** Add an experimental RTX PRO 6000 Blackwell catalog card for
`autotrust/JEV-27B-VL`, the multimodal sibling of `rtx6k/jev-9b`: calibrated System 1
decisions over text and images, and System 2 generation, from one vLLM engine.

## Research

The upstream Hugging Face repository was inspected at commit
`fee8c1d19d60216f8e9bcbb348141c0aa87ee2cc` (2026-10-11).

- `config.json`: `Qwen3_5ForConditionalGeneration` (`qwen3_5`), 64 language layers (one
  full-attention layer in four, 4 KV heads of dim 256), a 27-block vision encoder,
  262,144-token native context, bf16. 27,781,427,952 parameters (safetensors metadata).
- `base_model: Qwen/Qwen3.8-27B`. The chat template, `tokenizer.json`, `vocab.json`,
  `merges.txt` and `preprocessor_config.json` are byte-identical to `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- No `auto_map` and no remote-code hook in `config.json` or `tokenizer_config.json`. The
  adapter config names transformers' built-in `Qwen3_5ForCausalLM`.
- Code in the repository root (`serve.sh`, `serve_decide.py`) and the `blog/`, image and
  report files are not selected. The selection is the 18 safetensors shards (named by the
  first; the node downloads the whole `-NNNNN-of-00018` set), the index, the configs,
  tokenizer files, chat template, `calibration.json`, and `adapter_vllm/` (LoRA r=32 with
  the decision head on `lm_head`, 451.5 MB): 56,037,541,924 bytes (52.19 GiB) in 32 files.
- License: Apache-2.0.

The model card's serving requirements:

- `--max-num-seqs 8` is required: above 8 sequences in a batch, vLLM's LoRA path for this
  multimodal model class returns wrong System 1 probabilities.
- `--logprobs-mode processed_logprobs`, and System 1 requests must pass `top_k: 0` and
  `top_p: 1.0`, because `generation_config.json` sets `top_k=20` and `top_p=0.95` as
  request defaults that would truncate the returned probabilities.
- `--trust-request-chat-template` for image decisions, which send a raw template.
- KV cache about 65 KB per token; 52 GB of weights. Tested on one B200 with a September 2026
  vLLM development build.

vLLM v0.31.0 (already pinned by `rtx6k/jev-9b`) registers
`Qwen3_5ForConditionalGeneration`, which derives from `Qwen3VLForConditionalGeneration`,
a `SupportsLoRA` class.

## Decisions

- Marked `experimental`; not deployed yet when written.
- The repository's own `serve_decide.py` (the `POST /v1/decide` route) is not used: it is
  code from the model's publisher. System 1 is reached through `/v1/completions` and
  `/v1/chat/completions` with the client-side calculation the model card documents.
- Routing name `jev-27b-vl`; the decision module `jev-27b-vl-decision`.
- 75 GiB booked (`--gpu-memory-utilization 0.785` of 95.6 GiB): 52.2 GiB weights and
  adapter, the vision encoder's activations, CUDA graphs, and about 16 GiB of cache, enough
  for 32K context at the 8 sequences the model card allows. An estimate, not a measured
  floor.
- `--max-model-len 32768`, the model card's default; a 128K or 256K variant is a separate
  card once memory is measured.
- Up to 8 images per prompt, no video, as in the model card's server.
- 32 GiB of host memory and a 5-minute health start period for the 52 GiB load.

## Sources

- https://huggingface.co/autotrust/JEV-27B-VL
- https://huggingface.co/api/models/autotrust/JEV-27B-VL
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://github.com/vllm-project/vllm/tree/v0.31.0

## Outcome

Created the experimental card. Remaining: download and deploy through the control plane,
then validate System 2 on text and an image, and System 1 on text and an image against
the model card.
