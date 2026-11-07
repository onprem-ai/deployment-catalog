# 2026-11-02 - Without --trust-remote-code

## Problem

The profile started SGLang with `--trust-remote-code`. The flag lets Transformers
run Python code that ships with the model. The profile needs no such code, so the
flag only added risk.

## Why the flag is not necessary

- The weights (`RadixArk/Qwen3.8-Flash-Next-NVFP4` at
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`) hold no `.py` files: only
  safetensors, JSON, jinja and text files.
- `config.json` has no `auto_map`. The architecture
  `Qwen4ExpForConditionalGeneration` is in the SGLang image.
- `tokenizer_config.json` names `Qwen2Tokenizer`, and the preprocessor configs
  name `Qwen3VLProcessor`: both are Transformers classes.

## Change

- `compose.yaml`: `--trust-remote-code` removed. Nothing else changed.

## Measured on the deployment

Deployed on two RTX PRO 6000 Blackwell GPUs. The replica started,
became healthy, and answered through the API gateway:

- Text: "What is 17*23?" answered 391, with the reasoning in
  `reasoning_content`.
- Image: a 64x64 PNG, red left and blue right, answered "red on the left and
  blue on the right".
- Tool call: `get_weather` called with `{"city": "Zurich"}`, finish reason
  `tool_calls`.
