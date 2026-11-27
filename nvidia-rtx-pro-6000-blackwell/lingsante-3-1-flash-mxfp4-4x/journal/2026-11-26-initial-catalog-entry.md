# 2026-11-26 Initial catalog entry (written from sources, nothing run)

**Hardware:** none used. The profile targets four RTX PRO 6000 Blackwell; the authoring host
has two.

## Sources read

- `inclusionAI/LingSante-3.1-flash-MXFP4` at `56229393c30374608ce101f14fa2483b51c2229c`: model
  card, `config.json`, file sizes (Hugging Face API). Safetensors 313.2 GB (291.7 GiB) in 153
  files; 277.8B 8-bit integers (packed MXFP4), 15.0B FP8, 1.5B BF16. The card's SGLang command
  is for 4 × NVIDIA H20-3e, TP4 / EP4 / DP1, `--mem-fraction-static 0.80`, Humming 0.1.12.
- `inclusionAI/LingSante-3.1-flash` (the FP8 source model): names the image
  `lmsysorg/sglang:dev-ling-3.1-flash`, 8 × H200 with TP8 + EP8, 4 × GB300 with TP4 + EP4.
- SGLang's Ling-3.1-flash cookbook: FP8 only; no MXFP4, no Humming, no RTX GPU; "other
  platforms are unverified"; about five minutes of JIT compilation at a first start.
- Docker Hub: `lmsysorg/sglang:dev-ling-3.1-flash` is
  `sha256:001bd5c3d594767a4bb2bfc44dc7a3cf548cc190c17dfdf3d88de3a814f1c10f`. No tag of
  `lmsysorg/sglang` names Humming, MXFP4 or LingSante.

## Decisions

- **Four GPUs:** 291.7 GiB does not fit two.
- **The publisher's command,** with `--mem-fraction-static` 0.80 -> 0.88 and
  `--max-running-requests` 32 -> 8 for 96 GB cards. Engine booking 83.6 GiB (0.88 of 95 GiB),
  hardware booking 94 GiB.
- **Weights' files:** the 153 safetensors files, the tokenizer and template, `config.json`,
  the index, and the publisher's two Python files. Left out: the conversion logs
  (`worker-00N.json`, `conversion_complete.json`, `clean-success-*.txt`), the README and
  assets.
- **`speculative_decoding: mtp`:** SGLang's NEXTN on the model's one MTP layer.
- **No license file:** the repository has none.

## Not done

No image pulled, no weights downloaded, no start. The two Python files were scanned with grep,
not reviewed.

## Checker

Run with the control plane's `check_profile` (platform-alpha `b3643190`): no errors, 25 rules
passed, 1 warning, 2 hints.
