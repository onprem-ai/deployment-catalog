
## Without --trust-remote-code

`config.json` names `Qwen3_5ForConditionalGeneration` without an `auto_map`, the weights
list no Python files, and vLLM has the architecture built in, so `--trust-remote-code`
left the command. Deployed once on the same GPU with the round 10 settings: 32 of 32 answer
checks (with an image and a hidden number at 231,398 prompt tokens), 103 tokens/s at 1
request (ITL 9.7 ms) and 3,626 at 128, the KV cache unchanged at 1,831,048 tokens. The
benchmark file of 2026-11-12 stays the profile's measurement: the flag changed nothing
the engine ran.
