# 2026-10-12 - First Deployment

**Goal:** Download and deploy the JEV 27B VL card on the RTX PRO 6000 Blackwell dev node
through the control plane's API, and check both systems on text and images.

## Download

Requested through `inference_nodes_model_artifacts_change request_download`. The node
downloaded the Hugging Face source at commit `fee8c1d19d60216f8e9bcbb348141c0aa87ee2cc`
itself: the card names the first shard, and the node took the whole
`-NNNNN-of-00018` set. 56,037,541,924 bytes (52.2 GiB) in 44 minutes, about 21 MB/s,
every file checked against the SHA-256 Hugging Face publishes. No code from the
repository was downloaded. `vllm/vllm-openai:v0.31.0` was already on the node.

To make room on the dev machine, the unused DeepSeek-V4-Flash weights and their image
were removed through the API first.

## Deployment

The JEV 9B service was undeployed to free the GPU. The
deploy carried no warnings. Ready about 3.5 minutes after the deploy.

```text
Actual usage is 51.98 GiB for consumed memory (weights + non-torch), 2.05 GiB for peak activation
Available KV cache memory: 20.52 GiB
GPU KV cache size: 292,181 tokens, Maximum concurrency for 32,768 tokens per request: 8.92x
init engine (profile, create kv cache, warmup model) took 126.20 s (compilation: 31.90 s)
Loaded new LoRA adapter: name 'jev-27b-vl-decision', path '/models/model/adapter_vllm'
```

VRAM in use 78.7 GB of the 80.5 GB booked; the engine holds to its booking. The cache
holds 292K tokens, about nine full 32K contexts, so the 8-sequence limit, not memory,
bounds concurrency. A 128K-context variant fits in the same booking.

## Validation (requests sent to the engine's port on the node)

System 2, `jev-27b-vl`, thinking off:

- Text: "Safety stock is the extra inventory held to protect against uncertainties in
  demand or supply lead time, thereby preventing stockouts and ensuring business
  continuity."
- Image (a generated bar chart of three rising bars): "The bar chart displays three blue
  bars of increasing height from left to right, indicating a steady upward progression
  across the three categories."

System 1, `jev-27b-vl-decision` through `/v1/chat/completions` with the model card's raw
decision template, `top_k: 0`, `top_p: 1.0`, the head bias and per-kind temperature:

| question | result | model card |
|---|---|---|
| text choice: "my card was charged twice" to billing / shipping / tech support | 0.997 / 0.000 / 0.003 | 0.9969 / 0.0000 / 0.0031 |
| text noul: refund request | true 0.978 | |
| image noul: red square, "mostly red?" | true 0.999 | |
| image noul: blue square, same question | true 0.000 | |
| image choice: bar chart vs person, landscape, text screenshot | bar chart 1.000 | |
| image choice: bars increasing / decreasing / equal | increasing 1.000 | |

Concurrency: 32 image decisions from 16 parallel clients took 1.6 s, with identical
probabilities for every red and every blue request; requests beyond the 8-sequence limit
queue as the model card describes.

## Speed

Measured on the engine's port, distinct generated images, one `noul` decision per request:

| workload | latency | throughput |
|---|---|---|
| one 448 px image, one request at a time | median 83 ms | |
| 448 px images, 8 requests in flight | median 320-360 ms | 11-21 decisions/s (two runs) |
| one 1024 px image, one request at a time | median 275 ms | |
| 1024 px images, 8 requests in flight | median 0.9 s | 2.3 decisions/s |
| six 448 px images per decision, 8 in flight | median 1.1 s | 2.0 decisions/s |
| System 2, one 1024 px image, up to 120 tokens | 1.9 s | |

The 8-sequence limit caps throughput; further clients queue. Image size dominates: 1024 px
is 4-5 times slower than 448 px, so clients should downscale to about 448 px, as the model
card advises. The model card's B200 figure (about 13 decisions/s with six images each) is
well above the RTX PRO 6000's 2.0. GPU load during the runs could not be read (`nvidia-smi`
failed on the dev machine), so the spread between the two 448 px runs is unexplained.

## Through the API gateway

Requests sent to the dev stack's API gateway (the inference port, 443) with the
playground's service key:

- `/v1/models` lists `jev-27b-vl`. System 2 text and image chat answer as on the engine.
- `/v1/messages` (Anthropic surface) answers with a thinking block and a text block.
- System 1 through the path-prefix ingress, `POST /jev-27b-vl/v1/chat/completions` with
  `model: jev-27b-vl-decision`: the same probabilities as on the engine (billing 0.997,
  red square 0.999, rising bars 1.000). The body-routed `/v1/...` paths refuse
  `jev-27b-vl-decision` as an unknown model, as designed: the routing name is
  `jev-27b-vl`.
- Tool calling (`qwen3_coder` parser): with and without thinking, `get_order_status`
  came back as a proper tool call with `{"order_id": "A-7731", "include_tracking": true}`
  and `finish_reason: tool_calls`; the round trip with the tool's result answered with
  the carrier and tracking number.
- Smoke test: the default one-token check passes every run (99.4 % over the day, the
  misses during the start-up). It checks only for HTTP 2xx: with thinking on by default,
  its 5 tokens end inside the reasoning and `content` is empty.

## Not validated yet

- Long-context decisions near the 32K limit.

## Outcome

The card downloads and deploys through the control plane and serves both systems on text
and images, matching the model card's published example. It stays experimental until the
gateway path and tool calling are confirmed.
