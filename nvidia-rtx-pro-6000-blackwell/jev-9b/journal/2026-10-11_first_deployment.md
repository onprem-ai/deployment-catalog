# 2026-10-11 - First Deployment

**Goal:** Download and deploy the JEV-9B card on the RTX PRO 6000 Blackwell dev node
through the control plane's API, and check both systems against the model card.

## Download

The weights come from Hugging Face directly (`provider: huggingface`), downloaded by
the inference node's own model weight downloader (issue #157): the file list at commit
`d7bf1c0051f78e3a7bd5bbb90add79bfd769079b`, the 14 selected files, every file checked
against the SHA-256 Hugging Face publishes for it. The request went through
`inference_nodes_model_artifacts_change model_weights_retry`.

- 18,104,344,656 bytes (16.86 GiB) in about 12 minutes, about 27 MB/s.
- The `SHA256SUMS` the node wrote matches the hub's LFS ids, for example
  `model-00005-of-00005.safetensors` `dd63614f...44344`.
- Nothing under `vl/` (the repository's code) was downloaded.
- `vllm/vllm-openai:v0.31.0` was pulled and verified against its pinned digest
  (22.8 GB) through `model_images_retry`.

## Deployment

The Qwen3.8-27B service on the GPU was undeployed to
free the GPU. `propose_replica_placement` then offered that GPU (95.0 GiB free) and
refused the other (10.2 GiB free, the profile allocates 30 GiB). The deploy carried no
warnings.

From the vLLM log:

```text
Initializing a V1 LLM engine (v0.31.0)
Available KV cache memory: 9.79 GiB
GPU KV cache size: 288,263 tokens, Maximum concurrency for 32,768 tokens per request: 8.80x
init engine (profile, create kv cache, warmup model) took 98.87 s (compilation: 17.36 s)
Loaded new LoRA adapter: name 'jev-9b-decision', path '/models/model/adapter_vllm'
```

- Ready about two minutes after the deploy, faster than the 3-8 minutes the model
  card states for its build.
- VRAM in use: 30.1 GB of the 32.2 GB booked; the engine holds to its booking.
- `/v1/models` lists `jev-9b` and `jev-9b-decision`.

## Validation (requests sent to the engine's port on the node)

System 2, `jev-9b` through `/v1/chat/completions` with thinking off:

```text
Safety stock is the extra inventory a company maintains beyond expected demand to act
as a buffer against supply chain disruptions or unforeseen spikes in customer orders,
thereby preventing stockouts.
```

System 1, `jev-9b-decision` through `/v1/completions` with the model card's client
(head bias and per-kind temperature applied from the downloaded `decision_head.json`
and `calibration.json`), against the card's published distributions:

| question | this deployment | model card |
|---|---|---|
| choice: supplier response | issue_warning 0.336, renegotiate 0.138, dual_source 0.525, maintain 0.001 | 0.33, 0.14, 0.53, 0.001 |
| noul: refund request | false 0.009, true 0.991 | 0.009, 0.991 |

Both agree with the card to the precision it publishes. A `score` question returned a
plausible distribution (mode 1 of 0-5 for a terse reply to a detailed bug report).

## Not validated yet

- Requests through the API gateway: on the dev stack the gateway refused every model
  at the time (`tunnel session names no transport endpoint`, unrelated in-progress
  transport work), so the smoke test shows the model as starting and the path-prefix
  route for System 1 (`POST /jev-9b/v1/completions`) is not confirmed.
- Tool calling, long-context reasoning, prefix-cache throughput, and the memory floor.

## Outcome

The card downloads and deploys through the control plane and serves both systems with
the model card's numbers. It stays experimental until the gateway path, tool calling,
and a measured memory floor are confirmed.
