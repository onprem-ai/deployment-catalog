# 2026-08-15 - Model ID Alignment

**Goal:** Align the locally served model identity with the public Qwen3.8 27B
name used by OpenRouter, without adopting OpenRouter's provider namespace.

## Change

Both the native-context and 1M catalog cards now declare:

```text
model_id: qwen3.8-27b
```

OpenRouter publishes the full routed ID as `qwen/qwen3.8-27b`. The local ID
intentionally omits the external provider prefix while preserving the exact
model-name portion, `qwen3.8-27b`. Because `${MODEL_ID}` drives vLLM's
`--served-model-name`, this becomes the model value accepted by both local
cards after redeployment.

## Outcome

The two cards now share one explicit model identity. Existing running pods do
not change until redeployed from the updated catalog source.
