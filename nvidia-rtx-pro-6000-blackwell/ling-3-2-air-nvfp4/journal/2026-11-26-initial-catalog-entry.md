# 2026-11-26 Initial catalog entry (written from sources, nothing run)

**Hardware:** none used. The profile targets one RTX PRO 6000 Blackwell.

## Sources read

- `primitive-ai/Ling-3.2-air-NVFP4` at `a01b9c21680a104ccefcbce5b9e6745c004fb89b`: model card,
  file list with sizes and git blob ids (Hugging Face API), `tokenizer_config.json`. 44
  safetensors files, 77,888,233,296 bytes; 143 downloads, 2 likes.
- `inclusionAI/Ling-3.2-air` at `ff099487f6f57165e6fa9386ad61fddfafc78b76`: model card (SGLang
  and vLLM commands, `--tp 4` on 141 GB GPUs), file list with blob ids, the seven Python files.
- Docker Hub: no `vllm/vllm-openai:nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121`;
  `cu134-nightly-a2bcb565…` exists
  (`sha256:76a533475156ffdb3656ccc389d0c71f5cec16897c0f490ac43e4f379c4b4f0d`). The commit is
  vLLM main of 2026-11-05.
- Hugging Face organisation API: `primitive-ai` is verified, 31 models, 2 members.

## The rule this profile was written under

No Python file from the third-party repository: only safetensors, JSON and template files.

- `main` lists 52 files of `primitive-ai/Ling-3.2-air-NVFP4`, none ending in `.py`; the
  generator asserts it.
- The model's Python files are listed from `inclusionAI/Ling-3.2-air` instead, as the role
  `code`. Blob ids compared between the two repositories: all seven identical, so the third
  party changed none of them, and this profile does not rely on that.
- `tokenizer_config.json` names `PreTrainedTokenizerFast`, so the tokenizer itself needs no
  Python file; `config.json`'s `auto_map` (configuration and model class) and the image
  processor do.

## Decisions

- **One GPU, vLLM,** as primitive-ai measured it. `--gpu-memory-utilization 0.92`: engine
  booking 88 GiB, hardware booking 94 GiB.
- **`--max-model-len 262144`,** above the tested 32,768, because the reported pool is 1.02M
  tokens.
- **Prefix caching with `--mamba-cache-mode align`** from inclusionAI's vLLM command.
- **No MTP,** as primitive-ai ran it.
- **No license file:** neither repository has one.

## Not done

No image pulled, no weights downloaded, no start. primitive-ai's accuracy and speed figures are
not reproduced. The Python files were scanned with grep, not reviewed.

## Checker

Run with the control plane's `check_profile` (platform-alpha `b3643190`): no errors, 26 rules
passed, 1 warning, 3 hints.
