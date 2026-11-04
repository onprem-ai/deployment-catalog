# 2026-08-29 - Initial Catalog Entry

**Goal:** Add an experimental two-GPU RTX PRO 6000 Blackwell catalog card for
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` using its publisher-qualified language-only
MTP3 profile.

## Research

The Hugging Face repository was inspected at commit
`792d065e90b3b3a9315cc93b396322ccd169da29`. Its weight index references 120
Safetensors shards totaling 175,622,979,576 bytes (163.56 GiB). The checkpoint
uses uniform-K4 EXL3/TR3 compression for routed experts while preserving
non-routed tensors in the official source precision. The config identifies a
321B-total, 18B-active GLM5 Next MoE with a native 1,048,576-token context,
reasoning, tool use, a vision tower, and one built-in MTP layer.

The publisher explicitly states that this checkpoint requires its custom
vLLM/B12X runtime and is not compatible with stock vLLM. The published runtime
image is
`verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-language-only`,
pinned in the catalog to OCI index digest
`sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692`.
It was qualified on two RTX PRO 6000 Blackwell Workstation Edition GPUs with
96 GB VRAM each.

### Published Profile Options

All measurements in this table come from the upstream model card and its v84
`language-only-capacity.json` validation receipt. The KV measurements cover the
aggregate TP2 cache across the qualified two-GPU pair.

| Profile | Vision | Speculator | Max request length | KV cache | KV token capacity | Concurrency at request ceiling |
|---|---|---|---:|---:|---:|---:|
| Multimodal DFlash2-7 | Enabled | External seven-layer DFlash2 checkpoint | 98,304 | 5.68 GiB | 129,473 | 1.32x |
| Text-only DFlash2-7 | Disabled | External seven-layer DFlash2 checkpoint | 98,304 | 8.14 GiB | 184,619 | 1.88x |
| Text-only MTP3 | Disabled | Built-in MTP head | 131,072 | 6.49 GiB | 1,376,256 | 10.50x |

The KV GiB values are not directly comparable because the DFlash2 and MTP3
profiles use different physical cache layouts. Supported token capacity is the
useful comparison.

This card selects the language-only MTP3 profile because it requires no
separately licensed DFlash2 checkpoint, provides the largest measured KV
capacity, supports a 131,072-token request ceiling, and is the publisher's
recommended text-only daily-driver profile. The selected upstream launcher is
`compose.sm120-tp2-language-only.yaml`. Its key settings are TP2, EP enabled,
DCP2, vision disabled, built-in MTP3, NVFP4 DS-MLA KV, one sequence, chunked
prefill, disabled prefix caching, and a source GPU memory utilization of 0.986.

The v84 capacity receipt reports 1,376,256 allocated KV tokens and 10.50x
maximum concurrency at the 131,072-token request ceiling. The publisher's
container profile reserves 98.6 percent of each 96 GB GPU. The catalog records
a preliminary 94 GiB per GPU, which resolves to the application's 95 percent
utilization cap; it is not a locally binary-searched minimum, and the resulting
KV pool must be validated on this node.

The publisher reports 145.5, 147.2, and 151.5 decode tokens/s at empty, 32K, and
64K contexts respectively. Those values are not copied into the catalog card
because they were not measured through this application's benchmark API.

## Decisions

- Marked the card experimental until download, deployment, capability tests,
  and local VRAM bisection complete on this node.
- Configured exactly two GPUs with TP2, EP2, and DCP2.
- Selected the publisher's language-only MTP3 profile. The card intentionally
  advertises `multimodal: false` even though the checkpoint contains vision
  weights because this deployed profile disables the vision tower.
- Pinned the Hugging Face revision and the publisher-qualified v84 container
  image digest.
- Bundled the exact ShapleyMCG License 1.0 document and retained its custom
  `LicenseRef-ShapleyMCG-1.0` identifier. The license contains attribution
  requirements and a named exclusion and is not an OSI open-source license.
- Listed only files required by the language-only runtime: all 120 weight
  shards, weight index, model and quantization configs, tokenizer, generation
  config, chat template, and EXL3 storage ABI.
- Did not initiate a download. The node currently has only about 25 GiB free,
  far below the 163.56 GiB payload requirement.

## Local Preflight

The selected profile cannot currently be downloaded or validated on this node:

- The node has the required two RTX PRO 6000 Blackwell GPUs, but both are
  occupied by another two-GPU deployment.
- Persistent storage has only about 25 GiB free, so it cannot hold the
  163.56 GiB checkpoint payload.
- No catalog reload, download, deployment, or inference test was attempted.
- The pinned runtime image is declared by this catalog card. The catalog
  download workflow must transfer it into the local registry before an
  air-gapped deployment can use it.

## Sources

- https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw
- https://huggingface.co/api/models/brandonmusic/GLM-5.3-Flash-tr3-4bpw
- https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/blob/792d065e90b3b3a9315cc93b396322ccd169da29/runtime/compose.sm120-tp2-language-only.yaml
- https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/blob/792d065e90b3b3a9315cc93b396322ccd169da29/runtime-results/v84/validation/language-only-capacity.json

## Outcome

Created the schema-v3 experimental catalog card, mandatory journal, and bundled
license. The next operator session must first provide enough local disk space,
then download, deploy on GPUs 0 and 1 through the application APIs, and validate
chat, reasoning, tool calling, memory use, and performance.
