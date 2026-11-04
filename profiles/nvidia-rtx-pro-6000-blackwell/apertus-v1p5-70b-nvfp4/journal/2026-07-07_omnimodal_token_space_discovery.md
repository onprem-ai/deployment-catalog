# 2026-07-07 -- Discovery: Apertus v1.5 is Emu3.5-style omnimodal (served text-only)

## What was found

`tokenizer_config.json` of the v1.5 70B prerelease contains:

- `omnimodal_config` with two modalities:
  - vision: 131,072 discrete tokens, offset 131272, start/end tokens 131073/131074
  - audio: 4,096 discrete tokens, offset 262344, start/end tokens 131080/131081
- `vision_tokenizer`: type `Emu3.5`, codebook_size 131072, path pointing to
  CSCS-internal storage (`/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/
  Emu3.5-VisionTokenizer`)
- Special tokens incl. `<|img_start|>`, `<|image|>`, `<|img_end_of_row|>`,
  `<|img_generation_start|>`, `<|audio_start|>`, and 4k `<|audio token N|>`

This exactly explains the vocab size: 131072 (text) + 131072 (vision codebook)
+ 4096 (audio) + 512 (specials) = 266,752.

Emu3-style omnimodality lives in the VOCABULARY, not in model modules: images
and audio are quantized to discrete tokens by an external VQ tokenizer model
and flow through the same decoder. That is why the checkpoint's 1,123 tensors
are pure text-transformer (`ApertusForCausalLM`, no vision tower) and why the
architecture did not change vs v1.

## Why the card still says multimodal: false

As deployable today the model is text-only:

1. Serving: vLLM has no Apertus multimodal processor -- the OpenAI chat API
   path cannot accept images for this model. Only manual token-splicing via
   the completions API (prompt as token ids) is possible.
2. Unknown training status: this is a PRERELEASE; whether these 70B weights
   actually received multimodal training phases (vs merely reserving the
   token space) is not verifiable from the model files. Feeding vision tokens
   to an untrained-on-vision checkpoint yields garbage.
3. Audio codec for the 4,096-token audio space is unspecified in the config.

## The path to multimodal IS open (feasibility experiment)

`BAAI/Emu3.5-VisionTokenizer` is PUBLIC on HF (Apache-2.0, ungated,
Emu3p5VisionVQ, arxiv 2510.26583) -- the exact tokenizer type the config
names. A cheap experiment would settle question 2:

1. Download BAAI/Emu3.5-VisionTokenizer (small VQ model).
2. Encode a test image -> codebook ids -> add offset 131272 -> wrap with
   `<|img_start|>` / row tokens per Emu3.5 conventions.
3. Feed via vLLM /v1/completions with prompt token ids; ask the model to
   describe the image.
4. Coherent description => weights are vision-trained; garbage => token space
   is reserved but untrained in this prerelease.

Image GENERATION test is the reverse: prompt with `<|img_generation_start|>`,
constrain sampling to the vision-token range, VQ-decode the output.

If the experiment succeeds, catalog-grade support still requires real
integration work (processor + vLLM multimodal path or a sidecar encoder
service) -- tracked as future work, not part of this card.

## Outcome

Documented; card annotated. Experiment pending decision.
