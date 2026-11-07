# 2026-07-07 -- Chat template fix applied, multimodal pipeline confirmed

Reviewed the model files and upstream notes on tool calling, deployment and
the multimodal pipeline.

## 1. Tool-calling chat template bug -- FIXED in our checkpoints

The shipped 70B `chat_template.jinja` uses Python-style
`+` string concatenation on `tool_call.arguments`. vLLM's OpenAI server passes
tool arguments as dicts, so any tool call crashed with:

```
400 BadRequestError: can only concatenate str (not "dict") to str
```

The fix replaces `+` with Jinja's `~` operator (which
casts to string) and serializes dict defaults. Verified the diff is only
these edits. Applied the fixed template to all three local
checkpoints (three-tier in /persistent/models, FP8 baseline, and the BF16
assembled source), keeping `chat_template.jinja.orig` backups next to each.

Same failure genre as the gemma-4 sglang-mtp template issue (stale/broken
vendored chat template silently breaking tool calls) -- templates are code
and need the same review as code.

## 2. Tool parser: `--tool-call-parser apertus` exists upstream, NOT in our image

Upstream's official quickstart uses
`--enable-auto-tool-choice --tool-call-parser apertus`. Checked our validated
image (vllm/vllm-openai:v0.19.1-cu130): the lazy parser registry has no
`apertus` entry (33 parsers, none apertus). So the card keeps no parser flags
for now; adopt the apertus parser when the card's image is upgraded (open
item, together with the v0.23.0 test).

## 3. Official 70B deployment reference point

Upstream quickstart deploys text-only on 2x H200 with ON-THE-FLY FP8
(`--quantization fp8`, needs full BF16 loadable first) + fp8 KV cache +
100k max-model-len. Our offline three-tier serves on a single 96 GB card --
validation that offline quantization is the right call for this node class.
Their successful `--kv-cache-dtype fp8` also supports the estimate that
three-tier + FP8 KV reaches ~200k context here (journal open item 4).

## 4. Multimodal pipeline confirmed upstream

Upstream has a WORKING vLLM multimodal pipeline for Apertus v1.5, with
these modality tokenizers:

- Vision: Emu3.5 VQ-VAE (BAAI/Emu3.5-VisionTokenizer -- public, Apache-2.0)
- Audio: WavTokenizer-40 (novateur/WavTokenizer-large-unify-40token) --
  this resolves the "audio codec unspecified" open question in
  2026-07-07_omnimodal_token_space_discovery.md: 4,096 audio tokens =
  WavTokenizer codebook
- Existing code: `ApertusMultiModalProcessor` +
  `ApertusForConditionalGeneration.embed_input_ids()` (currently CPU-bound
  token injection; GPU-native execution is proposed upstream)

Implication: the multimodal feasibility path is NOT manual token splicing but
`--trust-remote-code` with the upstream processor code. The existence of a
working pipeline strongly suggests the prerelease weights are
multimodal-trained. Experiment remains pending (needs GPU juggling + the
upstream processor code, and our quantized checkpoints stripped
trust-remote-code files -- only tokenizer/config were downloaded).

## Outcome

Template fix applied to all local checkpoints. Card unchanged except comment.
Open items updated: adopt apertus tool parser on image upgrade; multimodal
experiment via upstream processor.
