# 2026-08-15 - Model ID Alignment

**Goal:** Keep the 1M sibling's served identity aligned with the native-context
Qwen3.8 27B card.

## Change

The card now declares `model_id: qwen3.8-27b`. This matches the model-name
portion of OpenRouter's public `qwen/qwen3.8-27b` ID while omitting the
external provider namespace.

The currently running 1M pod was created before this source change and still
serves the previous ID. The updated ID takes effect when the card is next
redeployed.

## Outcome

Both Qwen3.8 cards now use the same source-defined model ID.
