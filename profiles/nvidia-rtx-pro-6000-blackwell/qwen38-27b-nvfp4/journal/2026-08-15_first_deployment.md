# 2026-08-15 - First Deployment

**Goal:** Download, deploy, and smoke-test the new Qwen3.8 27B NVFP4 card on
the RTX PRO 6000 Blackwell node.

## Deployment

The previous `rtx6k--qwen36-35b-a3b-sglang-1m` InferenceService was removed
through the control-plane API. The Qwen3.8 checkpoint and vLLM image were then
downloaded through the model-catalog API. Every selected Hugging Face file
completed checksum verification.

The model was deployed through the model-catalog API on GPU 0 with the single
published memory profile plus 60 GiB of operator-added memory:

```text
vram_model_gi: 24
vram_total_gi: 84
gpu-memory-utilization: 0.88
```

The InferenceService became Ready after approximately four minutes. vLLM
0.27.1 loaded the checkpoint successfully on SM120 and reported:

```text
Model weights plus non-torch memory: 23.03 GiB
Peak activation memory: 2.33 GiB
CUDA graph memory: 0.70 GiB
Available KV cache memory: 58.22 GiB
GPU KV cache capacity: 1,700,147 tokens
Maximum 262,144-token request concurrency: 6.49x
KV cache dtype: FP8 E4M3
```

MTP was active with two speculative tokens. vLLM automatically selected
piecewise CUDA graphs because full CUDA graphs are not supported with
speculative decoding on the selected FlashInfer attention backend.

## Functional Tests

A chat request through `/api/inference-services/test/chat` asked for the exact
text `QWEN38_OK`. The model returned that text and included parsed reasoning.

A tool-use request defined an `add_numbers` function and requested arguments
17 and 25. The response had `finish_reason: tool_calls` and emitted:

```json
{"name":"add_numbers","arguments":"{\"a\": 17, \"b\": 25}"}
```

This confirms text inference, reasoning parsing, bundled chat-template use,
and `qwen3_coder` tool-call parsing on the deployed runtime. Vision input was
not exercised in this session.

## Performance Probe

The internal benchmark API was run at concurrency 1 with a random 150-word
prompt and 200 output tokens. The first request used service port 8080 and
timed out because the KServe Service exposes port 80. Retrying against the
Service without an explicit port completed and reported 106 tokens/s in the
benchmark log.

The benchmark result serializer then failed to emit structured JSON because
one latency statistic was positive infinity:

```text
Error formatting benchmark result: error marshalling JSON: json: unsupported value: +Inf
```

Therefore 106 tokens/s is an operational probe, not yet suitable for the
catalog performance field. A longer benchmark with enough samples is needed
to produce finite latency percentiles and a structured result.

## Capability Test API

The dedicated capability-test API was run against the KServe Service URL in
both non-streaming and streaming modes:

| Suite | Result |
|---|---|
| Vision | 8/8 passed |
| Tools | 12/12 passed |
| Vision + tools | 4/4 passed |
| Chat | 6/6 passed |

Vision covered color quadrants, OCR (`HELLO 42`), a landscape, and an animal.
Tool coverage included single selection, selection among multiple tools,
parallel calls, tool-result round-trip, no-tool restraint, and narration
before a tool call. Combined vision/tool tests confirmed structured image
observations and restraint when a tool was unnecessary. Chat tests confirmed
system, developer, and plain-user messages. Every applicable case passed in
both response modes.

The API advertises an `integrity` suite, but the digest-pinned backend image
used for the Kubernetes Job contains an older runner that rejects it:

```text
ERROR: TEST_TYPE must be one of ('vision', 'tools', 'vision-tools',
'embeddings', 'chat'), got 'integrity'
```

This is a test-runner image/version mismatch, not a model failure. The failed
Job was removed after recording the error.

## Outcome

The model is downloaded, deployed, Ready, and left running. The dedicated
capability suites confirm chat, multimodal vision, reasoning, tool calling,
and combined vision/tool use. Remaining validation includes context-specific
VRAM bisection and a longer benchmark that produces valid structured results.
