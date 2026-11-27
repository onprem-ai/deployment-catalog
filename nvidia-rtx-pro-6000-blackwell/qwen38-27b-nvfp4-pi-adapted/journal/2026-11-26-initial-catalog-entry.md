# 2026-11-26 Initial catalog entry (written from sources, nothing run)

**Hardware:** none used. The profile targets one RTX PRO 6000 Blackwell.

## Sources read

- `harness-hub/Qwen3.8-27B-pi-adapted` at `41396a9f14d9c1a1bf5bee4fb89e9dddadaeae7e`: model
  card, `adapter_config.json` (PEFT LoRA, r 32, alpha 64, seven projection types, base
  `Qwen/Qwen3.8-27B`), `chat_template.jinja`, file sizes. `adapter_model.safetensors` is
  637,610,624 bytes. No Python files.
- The catalog's `qwen38-27b-nvfp4` profile (the template) and `jev-27b-vl` (the catalog's
  other profile that serves a LoRA with vLLM, on a BF16 base).
- `unsloth/Qwen3.8-27B-NVFP4` `chat_template.jinja` at `16b6615a`, compared with the adapter's:
  Unsloth's merges leading system and developer messages and maps `high` effort to `xhigh`; the
  adapter's is 989 bytes shorter and does neither.

## Decisions

- **NVFP4 base, as the template.** The adapter's card says it was trained on BF16 and that
  quantized bases drift. The BF16 base fits one GPU too (`jev-27b-vl` measures 55.4 GB of
  weights) and is the fallback.
- **Two weights roles:** `main` (the template's, unchanged) and `adapter`, each on a volume of
  its own; the LoRA module is `/models/adapter`.
- **The routing model id is the LoRA module's name,** `qwen3.8-27b-pi`, so routed requests run
  with the adapter. The base model's served name is `qwen3.8-27b-pi-base`.
- **The adapter's evaluated options over the template's** where they differ: thinking off by
  default, its chat template, `qwen3_xml`, `--language-model-only`, `--mamba-cache-mode align`.
- **MTP off.** Unknown with a LoRA; the adapter's recipe does not use it.
- **Bookings as the template's; no measured block:** the compose file changed, so the
  template's measurement does not carry over.
- **Context left at the native 262,144;** the adapter's evaluations ran with 131,072.

## Not done

No deploy, no request, no benchmark. Whether the gateway passes `qwen3.8-27b-pi` through to
the engine as the model name was not checked beyond the checker's rules.

## Checker

Run with the control plane's `check_profile` (platform-alpha `b3643190`): no errors, no
warnings, 28 rules passed, 2 hints.
