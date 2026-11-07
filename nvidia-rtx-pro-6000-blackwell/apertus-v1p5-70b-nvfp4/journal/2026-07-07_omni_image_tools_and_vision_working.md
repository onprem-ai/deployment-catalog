# 2026-07-07 -- vllm-apertus-omni: tool calling AND vision WORKING on the three-tier checkpoint

## Summary

Built `onpremai/vllm-apertus-omni:apertus15-20260707`: the official swiss-ai Apertus 1.5 multimodal
vLLM fork image + the upstream apertus tool parser vendored at commit
4a39b4f5 and loaded via --tool-parser-plugin. Served the three-tier
checkpoint with it and verified BOTH capabilities end to end on the RTX 6000
Blackwell:

- Tool calling: finish_reason=tool_calls with correctly structured
  get_current_weather({"city": "Zurich"}) -- vendored parser + the fixed
  chat template working together.
- Vision: a synthetic test image (red square top-left, blue square
  bottom-right on white) was described EXACTLY: "a red square positioned in
  the upper left corner and a blue square located in the lower right
  corner." The prerelease weights are genuinely vision-trained, and the
  Emu3.5 discrete-token pipeline works on the QUANTIZED (NVFP4+FP8
  mixed) checkpoint.

Footprint with multimodal enabled: 49.14 GiB weights, 74,288 KV tokens at
util 0.80 / 16k max-len (vs 103,872 at util 0.90 text-only -- the lower util
follows the caveat that CPU-driven encoder VRAM is invisible to
vLLM's profiler).

## Serve command that works (recorded verbatim)

```
python3 -m vllm.entrypoints.openai.api_server \
  --model /model --served-model-name apertus-v15-70b-omni \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --max-model-len 16384 --gpu-memory-utilization 0.80 \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image": 2, "audio": 0}' \
  --mm-processor-kwargs '{"apertus_vision_tokenizer_device": "cuda"}' \
  --chat-template /model/chat_template.jinja \
  --tool-parser-plugin /workspace/apertus_tool_parser.py \
  --tool-call-parser apertus --enable-auto-tool-choice
```

## Bring-up failures and fixes (in order hit)

1. Tokenizer load failure without --trust-remote-code. The official
   quickstart uses the flag; required here too.
2. `ValueError: Tokenizer class TokenizersBackend does not exist` -- the
   checkpoint's tokenizer_config.json was written by transformers 5.3.0
   (class name TokenizersBackend); the fork image ships transformers 4.57.6.
   Fix: patch tokenizer_class to PreTrainedTokenizerFast (same
   tokenizer.json backend, accepted by both transformers generations).
   Applied to all three local checkpoints with .orig backups.
3. `KeyError: "Modality 'audio' not found. Available modalities: set()"`
   during engine-core memory profiling. Root cause: the Apertus fork uses a
   token-injection design -- ApertusMultiModalProcessor serializes images/
   audio to discrete token ids in the prompt and returns EMPTY mm_kwargs;
   embed_multimodal() explicitly raises if it receives any kwargs. vLLM's
   encoder-cache profiling is therefore fundamentally incompatible and must
   be skipped: `--skip-mm-profiling` (the guard at
   gpu_model_runner.py:5725 in the fork). Also set
   `--limit-mm-per-prompt '{"image": 2, "audio": 0}'`.
4. Audio remains DISABLED: even before profiling was skipped, the audio
   dummy path produced no processor output. Untested whether real audio
   inputs work with profiling skipped; WavTokenizer path may need
   `apertus_audio_tokenizer_path` in mm_processor_kwargs (the fork example
   passes a local wavtokenizer path). Follow-up item.

## Open items

1. Redeploy gemma-4-31b-it-nvfp4-sglang-mtp (was undeployed for the test).
2. Audio input bring-up (apertus_audio_tokenizer_path + WavTokenizer
   weights; bundling the weights with the model files is proposed upstream).
3. Decide catalog shape: separate `apertus-v15-70b-prerelease-omni` card on
   the omni image (tools + vision, util 0.80) vs replacing the text card.
   Suggest a separate experimental card -- the text card keeps util 0.90 and
   103k KV tokens.
4. Image generation test (<|img_generation_start|> + constrained sampling +
   VQ decode) -- the fork may or may not expose this server-side.
5. Push onpremai/vllm-apertus-omni to Docker Hub (needs docker login).
