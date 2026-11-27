# Qwen3.8 27B pi-adapted (NVFP4 base, LoRA, MTP, thinking off)

> ⚠️ **Experimental.** On 2026-11-27 this profile started on an RTX PRO 6000 Blackwell: vLLM
> applies the LoRA to the NVFP4 base, the adapter changes the answers, and the full benchmark
> ran (`journal/2026-11-27-first-start-tuning-and-benchmark.md`). Not measured: how much of
> the adapter's published gain survives the quantized base (it was trained on BF16).

The [`qwen38-27b-nvfp4`](../qwen38-27b-nvfp4) profile with the LoRA adapter
[`harness-hub/Qwen3.8-27B-pi-adapted`](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted)
loaded on top. Ask for the model `qwen3.8-27b-pi`.

## Measured (1 × RTX PRO 6000 Blackwell Server Edition, 2026-11-27)

| | One request | Plateau |
|---|---|---|
| Real coding prompts (`coding-greedy`, probe) | 155 tokens per second | 2,754 at 32 requests (not the limit) |
| Short prompts, long answers (`max-decode`) | 136 tokens per second | 3,529 at 96 requests |
| Mixed lengths (`mixed-load`) | 109 tokens per second | 1,173–1,226 at 48–80 requests |
| 8,192-token prompts (`max-prefill`) | 6,018 prefill tokens per second | 9,803–10,053 at 12–32 requests |

- No failed request in the benchmark (`benchmarks/2026-11-27-0444Z-llmperfbench.yaml`).
- Ready 3.3 to 3.8 minutes after a deploy.
- The adapter answers differently from the base model on 3 of 3 agent-style prompts.
- Answers: 24 of 24 known answers; thinking off by default and on with `reasoning_effort`
  (`low`, `medium`, `xhigh`; `high` is refused); parallel tool calls and an answer from their
  results; a hidden code found in a 251,760-token prompt (161 s: long prompts prefill slowly).
- GPU memory: 21.71 GiB of weights, a 60.24 GiB FP8 KV cache of 1,759,036 tokens; peak
  92.82 GiB.
- **Host memory:** the container held 15.6 GiB of its 16 GiB limit after long prompts. Raise
  `limits.memory` before serving them.

## The adapter

A rank-32 LoRA (638 MB, on the q, k, v, o, gate, up and down projections) for
`Qwen/Qwen3.8-27B`, trained with reinforcement learning (AsyncGRPO) to work as the model behind
the [pi](https://github.com/earendil-works/pi) coding agent. Apache 2.0. Its publisher reports,
under pi, one attempt per task, thinking off, on the BF16 base:

| Benchmark | Qwen3.8-27B | With the adapter | Qwen3.8-27B, `xhigh` thinking |
|---|---|---|---|
| SWE-bench Pro (hard 51) | 39.2% | 64.7% | 74.5% |
| Terminal-Bench 2.1 | 41.0% | 56.6% | 54.2% |

With about 4.5 times fewer generated tokens per task than the base model thinking. These are
the publisher's figures, not reproduced, and not on this profile's quantized base.

## What differs from `qwen38-27b-nvfp4`

| | `qwen38-27b-nvfp4` | This profile | Why |
|---|---|---|---|
| Weights | Unsloth NVFP4 | the same, plus the adapter | |
| Model to ask for | `qwen3.8-27b` | `qwen3.8-27b-pi` | the LoRA module's name |
| Thinking | on by default | off by default | what the adapter was trained for |
| Chat template | Unsloth's | the adapter repository's | the one it was trained with |
| Tool-call parser | `qwen3_coder` | `qwen3_xml` | the adapter's evaluations |
| Images and video | yes | no (`--language-model-only`) | the adapter's evaluations are text only |
| MTP speculative decoding | 2 draft tokens | 4 draft tokens | it runs with the LoRA; one request decodes 155 tokens per second instead of 62 |
| Measured memory, benchmark | yes | yes | 2026-11-27, one GPU |

Image, GPU memory bookings, FP8 KV cache, batch limits and the 262,144-token context are the
template's.

## Use with pi

`~/.pi/agent/models.json`, with the platform's API gateway as `baseUrl`:

```json
{
  "providers": {
    "onprem": {
      "baseUrl": "https://<gateway>/v1",
      "api": "openai-completions",
      "apiKey": "<api key>",
      "models": [{ "id": "qwen3.8-27b-pi" }]
    }
  }
}
```

`"api": "openai-completions"` matters: without it pi uses the Responses API, which the adapter
was not trained or evaluated with. Its evaluations sampled at temperature 0.8, top-p 1.0.

## Before this profile is more than a draft

1. Raise the container's memory limit (16G now; 15.6 GiB held after long prompts).
2. Run the pi harness against `qwen3.8-27b-pi` and `qwen3.8-27b-pi-base` to see whether the
   adapter's gain survives the quantized base. If it does not, the alternative is the BF16
   base (`Qwen/Qwen3.8-27B`, about 54 GB, the adapter's own recipe), or the adapter merged
   into the base and re-quantized.
3. Try 5 draft tokens; look at the prefill speed of long prompts with and without the LoRA.
4. Check the answers through the API gateway and the Anthropic API.

## Checker

No errors, no warnings. Hint: `engine.gpu-memory-share-written`.

## License

Apache License 2.0 for the base model and the adapter (`licenses/source-0-LICENSE.txt`).
