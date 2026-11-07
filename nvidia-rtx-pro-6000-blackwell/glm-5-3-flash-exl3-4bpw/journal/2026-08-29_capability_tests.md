# 2026-08-29 - Capability Tests

**Goal:** Validate chat integrity and, especially, OpenAI-compatible tool calling
for the deployed `glm-5.3-flash-exl3-4bpw` service.

## Runtime Observed

The InferenceService was already deployed and Ready when this test session
started. Its deployment annotations and rendered command confirm two GPUs,
TP2, EP2, DCP2, the `glm47` tool-call parser, automatic tool selection, and the
built-in three-token MTP speculator. Both RTX PRO 6000 Blackwell GPUs showed a
vLLM worker using approximately 93,328 MiB.

The engine startup log reported:

- Model load memory: 82.48 GiB per worker
- Available KV cache memory: 1.73 GiB
- GPU KV cache size: 363,426 tokens
- Maximum concurrency at 131,072 tokens per request: 2.77x

This local deployment therefore does not reproduce the upstream 1,376,256-token
KV capacity. It runs with `--gpu-memory-utilization 0.95`, while the upstream
capacity receipt used 0.986. The runtime also reports that CUDA graph memory
profiling reduces the effective 0.95 allocation to the equivalent of 0.9393.
The catalog must not claim the upstream 1.376M capacity as locally validated.

## Tool-Calling Capability Suite

The official capability test was started through:

```text
POST /api/capability-test?name=rtx6k--glm-5-3-flash-exl3-4bpw&namespace=inference
```

with `test_type=tools`, the internal predictor `/v1` URL, and the explicit
served model name. All 12 tests passed: six non-streaming and the same six using
streaming responses.

| Test | Non-stream | Stream | Observed behavior |
|---|---:|---:|---|
| Single tool | Pass | Pass | Selected `get_weather` with Berlin |
| Multiple-tool selection | Pass | Pass | Selected `send_email` with `john@example.com` |
| Parallel tool calls | Pass | Pass | Produced calls for both Berlin and London |
| Tool-result round trip | Pass | Pass | Consumed the tool result and answered with 22 C and sunny conditions |
| No-tool restraint | Pass | Pass | Answered simple arithmetic without calling a tool |
| Tool call after narration | Pass | Pass | Narrated briefly, then emitted a parsed Paris weather call without leaking parser markup |

This validates tool selection, JSON argument serialization, parallel calls,
assistant/tool history handling, restraint, and both parser paths used for
streaming and non-streaming responses.

## Other Capability Suites

The chat suite passed 5 of 6 tests:

- Developer-message chat passed in both modes.
- Plain chat passed in both modes.
- System-message chat passed in streaming mode.
- System-message chat returned an empty response in non-streaming mode.

The integrity suite passed all 5 tests:

- No leaked chat-template markup
- Generation terminated normally
- No degenerate repetition
- No confusable-character substitution
- Reasoning-effort probe completed, but reported that distinct reasoning effort
  levels are not supported because all levels produced 13 tokens

The non-streaming system-message failure remains an open issue. It does not
invalidate the successful tool-calling suite, but the card remains experimental
until that failure and the reduced local KV capacity are understood.

## Outcome

Tool calling is functional: 12 of 12 official tool tests passed across streaming
and non-streaming modes. General integrity passed 5 of 5. Basic chat passed 5 of
6 due to one empty non-streaming system-message response. The deployment remains
Ready, and no cluster mutation was performed during testing.
