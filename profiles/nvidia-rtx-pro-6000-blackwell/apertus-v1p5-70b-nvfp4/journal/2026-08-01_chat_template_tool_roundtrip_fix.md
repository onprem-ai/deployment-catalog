# 2026-08-01 -- Chat template fix for tool result roundtrip

## Problem

The tool result roundtrip test (`tool_result_roundtrip`) failed with
`HTTP 400: Invalid assistant content`. The chain-of-thought test had to
send an assistant message with `tool_calls` after getting the first response,
then provide the tool result as a tool message. vLLM's
`_postprocess_messages` transforms the assistant message so that
`content: ""` becomes `content: [{"type": "text", "text": ""}]` (a list),
and the official Apertus v1.5 chat template's content-type check only
handles strings and mappings, not lists. The `{%- else -%}` branch then
raised `raise_exception("Invalid assistant content")`.

Additionally, the assistant message handling had a second `{%- else -%}`
that fired when `"content"` was absent from the message (even if
`tool_calls` were present). This bare `{%- else -%}` caught the case
where an assistant message only has `tool_calls` and no `content` key
at all.

## Root cause

Two fixes were needed:

1. **Line 313 (content as list)**: The template section for
   `{%- if "content" in message -%}` checked:
   - `message.content is string` -> render as text
   - `message.content is mapping and "blocks" in message.content` -> render blocks
   - `{%- else -%}` -> `raise_exception("Invalid assistant content")`
   
   When vLLM transforms `content: ""` into a list of content parts
   (`[{"type": "text", "text": ""}]`), none of the positive branches
   matched, and it fell through to the error.

2. **Line 317 (content missing with tool_calls)**: After the content check
   block, the template had:
   - `{%- else -%}` -> `raise_exception("Invalid assistant message")`
   
   This caught messages that had `tool_calls` but no `content` key.
   Needed to check for `tool_calls` before raising.

## Fix applied

### Fix 1: Handle content as a list (OpenAI format)

Changed the bare `{%- else -%}` at the assistant content type check to:

```
{%- elif message.content is iterable and message.content is not string and message.content is not mapping -%}
    {#- Content as list (OpenAI format) -- join text parts -#}
    {%- for part in message.content -%}
        {%- if part.type == "text" -%}
            {{ part.text }}
        {%- endif -%}
    {%- endfor -%}
{%- else -%}
    {{- raise_exception("Invalid assistant content") -}}
{%- endif -%}
```

This joins text parts from the OpenAI content parts list format.

### Fix 2: Handle tool_calls when content is absent

Changed the bare `{%- else -%}` to check for tool_calls:

```
{%- elif "tool_calls" in message and message.tool_calls -%}
    {#- tool calls without content -- valid, continue -#}
{%- else -%}
    {{- raise_exception("Invalid assistant message") -}}
{%- endif -%}
```

## Files modified

- `/persistent/models/swiss-ai--Apertus-v1.5-70B--model-apertus-model/chat_template.jinja`
  (slug path -- live deployment mount)
- `/persistent/models/swiss-ai/Apertus-v1.5-70B-NVFP4/chat_template.jinja`
  (quant output -- source of truth)

Both locations patched identically.

## Test results after fix

| Test | Before | After |
|------|--------|-------|
| tool_result_roundtrip | FAIL (400) | **PASS** |
| single_tool | PASS | PASS |
| multi_tool_selection | PASS | PASS |
| parallel_tool_calls | FAIL (model behavior) | FAIL (model behavior) |
| no_tool_restraint | PASS | PASS |

The `parallel_tool_calls` failure is a model behavior limitation (70B model
calls 1 tool when asked about 2 cities) -- not a deployment issue.

## Notes

- The fix is on the model checkpoint disk, NOT in the container image.
  `chat_template.jinja` is loaded from the volume mount at `/model/`.
  The fix persists across image rebuilds since the model path and volume
  mount remain the same.
- vLLM caches the tokenizer's chat_template at model load time, so the
  pod must be restarted for the change to take effect. The model does NOT
  re-read `chat_template.jinja` from disk on each request.
- If the model is re-downloaded from HuggingFace, the template fix must
  be re-applied.
