# Anthropic-endpoint inline system message leak (agentic clients)

**Date:** 2026-07-24
**Basis:** a client (a Claude Code setup) hit this while driving `deepseek-v4-flash` through the ai-gateway's native
Anthropic passthrough. Documenting for the record; no code/config change made.

## Symptom

Model echoed raw framework/system-prompt content (skill list, hook/cron
notification text) into its visible reply instead of just answering the
user's message.

## Root cause

Not a catalog misconfiguration and not a missing chat template -- the model
card is explicit that DeepSeek-V4 ships no Jinja template by design, using
the vendored `encoding_dsv4.py` instead (already correctly shipped in this
catalog dir, see `inference.yaml` file list).

The bug is upstream in vLLM's *Anthropic-endpoint* (`/v1/messages`)
chat-template resolution:

- Since vLLM PR #46025, inline (mid-conversation) `role: "system"` messages
  default to "preserved in place" rather than merged to the front, for any
  model whose template doesn't explicitly reject that structure.
- DeepSeek-V4's `encoding_dsv4.py` renders system content with
  `system_msg_template: str = "{content}"` -- no delimiter marks it as
  hidden/framework text. When a client (here: Claude Code, which puts
  `role: "system"` entries inside `messages[]` for skill lists / hook
  context -- a relatively new Anthropic Messages API behavior) injects one
  mid-conversation, the model has no boundary and starts completing/echoing
  it.
- Confirmed via live upstream report reproduced with Claude Code directly:
  `vllm-project/vllm#46710`. Fix PR
  `vllm-project/vllm#47681` ("[Bugfix][Anthropic API] Default to merging
  inline system messages", adds `--inline-system-messages merge|preserve|auto`)
  is **open/unmerged** as of 2026-07-24.

This is specific to vLLM's Anthropic API surface, not a general DeepSeek-V4
defect -- GLM5.x/MiniMax/Kimi handle the same preserve-in-place path fine
per the upstream issue thread.

## Potential workaround (untested, NOT applied)

The ai-gateway already has a translation path for models
with `ai_gateway.anthropic.native: false` (+ `openai.native: true`):
it translates the Anthropic-shaped request and
forwards to vLLM's OpenAI-compatible `/v1/chat/completions` instead of
`/v1/messages` -- a different vLLM code path that should not carry the
same Anthropic-endpoint chat-template auto-detect bug.

Caveats before ever flipping this for real:
- The gateway's own translation does NOT merge
  inline system messages either -- it passes them through positionally.
  Any fix would come purely from landing on vLLM's OpenAI endpoint, not
  from gateway-side normalization.
- Zero catalog entries (of 53) currently set `anthropic.native: false` --
  this path exists and is unit-tested but has never been exercised
  end-to-end in this repo. Would be first-of-its-kind for this catalog.
- Unverified whether `--tool-call-parser deepseek_v4` /
  `--reasoning-parser deepseek_v4` behave identically against vLLM's
  OpenAI endpoint (should be endpoint-agnostic since they parse completion
  output, not request templating, but not empirically confirmed here).

## Disposition

Not changing `inference.yaml` maturity or `ai_gateway.anthropic.native`.
Model remains `stable` for this repo's own (non-agentic, completion-style)
traffic, which is what it was actually validated against. If this leak
starts affecting agentic use of this model more broadly, revisit the `native: false` route above, or re-pin the
forked vLLM image once vllm#47681 merges.
