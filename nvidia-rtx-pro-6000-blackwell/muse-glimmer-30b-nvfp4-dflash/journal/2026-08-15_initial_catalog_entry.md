# 2026-08-15 - Initial Muse Glimmer NVFP4 DFlash Catalog Entry

**Goal:** Add an experimental Muse Glimmer 30B card that uses both Blackwell NVFP4 inference and Meta's official DFlash speculative drafter.

## Research

The selected target is `RadixArk/Muse-Glimmer-NVFP4` at revision `1416629950afdfb276618fd4810103681dd10f4a`. The official SGLang Muse Glimmer cookbook identifies this artifact as its ready-to-serve mixed NVFP4/MXFP8 checkpoint. The five weight shards total about 18.3 GiB. It is text-only because the artifact omits the vision tower.

The drafter is Meta's official `meta-models/Muse-Glimmer-30B-assistant` at revision `e8192f3a8f617f74be2ce220360c89ef4789f39f`. It is a BF16 five-layer DFlash model with a 16-token native block size and weighs about 4.8 GiB.

The runtime is SGLang's official Muse Glimmer development image for amd64, pinned to digest `sha256:b8022194176049aaa472943cf8ba9ecb0d8740a42442592327832b6887e622a7`.

The launch flags follow SGLang's verified RTX PRO 6000 NVFP4 plus DFlash configuration:

- `--language-model-only`
- `--reasoning-parser muse`
- `--tool-call-parser muse`
- `--speculative-algorithm DFLASH`
- `--speculative-draft-model-path meta-models/Muse-Glimmer-30B-assistant`
- `--speculative-draft-model-quantization fp8`
- `--kv-cache-dtype fp8_e4m3`

A block size of 5 is explicit because SGLang's Muse Glimmer regression test uses it and requires an average accepted length above 2.5. The draft is exposed through a local Hugging Face cache layout so runtime startup remains offline.

## Initial Resource Profile

The initial 82 GiB profile is deliberately a conservative bring-up value, not a measured resource floor. It must be replaced after binary-search measurement. No performance values are recorded until the application benchmark API produces them.

## Outcome

Created an experimental catalog card with pinned model revisions, pinned runtime image, both model sources, DFlash configuration, and application gateway routing. Download, deployment, capability tests, DFlash acceptance measurement, and benchmarking remain pending.
