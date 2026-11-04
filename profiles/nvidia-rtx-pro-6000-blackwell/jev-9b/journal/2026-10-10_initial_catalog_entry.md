# 2026-10-10 - Initial Catalog Entry

**Goal:** Add an experimental RTX PRO 6000 Blackwell catalog card for
`autotrust/JEV-9B`, an open-weights decision model ("System 1") on an
unmodified Qwen3.5-9B ("System 2"), served by one vLLM engine.

## Research

The upstream Hugging Face repository was inspected at commit
`d7bf1c0051f78e3a7bd5bbb90add79bfd769079b` (2026-10-10). The commits after
`b63f651ce8ed64481d3f5e73ecdb05f740042f01` (2026-10-03) change only the model
card; every other file is identical between the two.

- `config.json`: `Qwen3_5ForCausalLM` (`qwen3_5_text`), 32 layers (24 Gated
  DeltaNet linear-attention, 8 full-attention, 4 KV heads of dim 256), vocab
  248,320, native 262,144-token context, bf16. The vision tower and MTP head
  of the base model are not in this repository.
- Parameters: 8,953,803,264 (safetensors metadata).
- Selected files: 5 backbone shards, index, config, tokenizer, chat template,
  `calibration.json`, and the vLLM adapter (`adapter_vllm/`: backbone LoRA
  zero-padded to r=32 plus the 24-slot decision head as an `lm_head` LoRA,
  168 MiB). Total 18,104,344,656 bytes (16.86 GiB). The PEFT adapter
  (`adapter/`, `head.safetensors`), the vision variant (`vl/`), reports and
  demo videos are not needed for vLLM serving and are left out.
- `adapter_vllm/adapter_config.json`: r=32, target modules include
  `lm_head`, base class `Qwen3_5ForCausalLM`.
- License: Apache-2.0 (weights, base model, and training corpus).

The model card's vLLM requirements: a build with `qwen3_5` support, LoRA on
`lm_head`, `--logprobs-mode processed_logprobs` (required, so returned
log-probabilities respect `allowed_token_ids`), and `allowed_token_ids`. It
was tested on a September 2026 development build; start-up takes 3-8 minutes.

vLLM v0.31.0 (released 2026-10-04, older than the 72-hour supply-chain
window) was checked at its tag: the registry maps `Qwen3_5ForCausalLM`, the
model class implements `SupportsLoRA` and maps a PEFT `lm_head` target onto
the output embeddings, and `--logprobs-mode` and `--mamba-cache-mode` exist.
The multi-arch manifest digest
`sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`
was read from Docker Hub; its ENTRYPOINT is `vllm serve`.

The chat template is byte-identical to `Qwen/Qwen3.5-9B`'s (Qwen3 XML tool
calls), so the `qwen3` reasoning parser and `qwen3_coder` tool parser apply as
on the other Qwen3.5/3.8 cards.

## Decisions

- Marked the card `experimental`; nothing has been downloaded or deployed yet.
- Weights from Hugging Face directly (pinned commit, explicit file list); no
  OnPrem AI re-publication.
- Routing name `jev-9b` for System 2. The decision LoRA module is
  `jev-9b-decision` (the card's `jev-decision`, prefixed so it cannot collide
  with another deployment's names).
- 30 GiB booked (`--gpu-memory-utilization 0.31` of 95.6 GiB): 16.9 GiB
  weights and adapter, the rest for activations, CUDA graphs and cache. Only
  the 8 full-attention layers hold a KV cache (32 KiB per token in bf16), so
  this is generous for 32K context at 64 sequences. This is an estimate, not a
  measured floor.
- `--max-model-len 32768`: the card sizes 4096 for decisions alone and advises
  more for System 2 reasoning.
- KV cache left in bf16 (no fp8): System 1 returns a probability distribution
  and the card's fidelity numbers were measured without KV quantization.
- Prefix caching with `--mamba-cache-mode align`, as the card recommends for
  many questions about one state (+14-20 % throughput measured on B200).
- Text only. The base model's vision path (`vl/serve.sh`) needs the full
  `Qwen/Qwen3.5-9B` checkpoint as a second weights source, a different adapter
  (`vl/adapter_vllm`), the custom `serve_decide.py` server and
  `--max-num-seqs 8` (the card reports wrong System 1 probabilities above 8
  sequences on that model class). It belongs in a separate card.

## Reaching System 1 through the API gateway

The gateway's body-routed `/v1/...` paths route by the `model` field, and only
`jev-9b` is a routing name; a body naming `jev-9b-decision` there is refused
as an unknown model. System 1 requests go through the path-prefix ingress,
which routes by the first path segment and forwards the body unchanged:

```text
POST /jev-9b/v1/completions
{"model": "jev-9b-decision", "prompt": "[kind] noul\n[state] ...\n[question] ...\n[options]\nfalse\ntrue\n[decision]:",
 "max_tokens": 1, "temperature": 1.0, "logprobs": 2, "allowed_token_ids": [...],
 "add_special_tokens": false, "return_tokens_as_token_ids": true}
```

The client applies the head bias from `adapter_vllm/decision_head.json` and
the per-kind temperature from `calibration.json` to the returned
log-probabilities, exactly as in the model card's quickstart.

## Sources

- https://huggingface.co/autotrust/JEV-9B
- https://huggingface.co/api/models/autotrust/JEV-9B
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://github.com/vllm-project/vllm/tree/v0.31.0
- https://hub.docker.com/r/vllm/vllm-openai/tags

## Outcome

Created the experimental catalog card and this journal entry. Remaining work:
download through the model-catalog API, deploy through the API, validate
System 2 (chat, reasoning, tool calling) and System 1 (the card's two
quickstart decisions, compared against its published distributions), confirm
the routing above, then measure the memory floor and throughput before
changing maturity.
