# 2026-07-08 -- Agentic capability finding: prerelease parrots examples under a heavy agent prompt (stack verified correct)

## Symptom

In a DevOps agent chat client, the user asked
"what model is deployed right now?" with model apertus-v15-70b:

- Expected (qwen36-35b-a3b, gemma-4-31b behavior): call `control_plane_api`,
  answer from the response.
- Observed: the model NARRATED a fake tool session in prose ("Now I'll query
  the model catalog... Let me render a chart showing the token usage by
  model...") and fabricated results: `llama-3-8b-instruct: 4,235,890 tokens`,
  `gpt-4o-mini: 3,120,450`, `mistral-8x7b: 1,890,230` -- then repeated the
  same answer block 6+ times.

## Diagnosis

**The serving stack is correct; the model itself degrades under the full
agent prompt.**

1. The fabricated content is a PARROT of the agent's system prompt: the
   prompt (~4.7 KB block) contains
   worked examples, including
   `render_chart(chart_type="donut", query="sum by (routing_model_id) (increase(onprem_ai_api_gateway_tokens_total[1h]))", title="Tokens by model (1h)")`.
   The hallucination is exactly a "tokens by model" narrative. The invented
   model names/token counts appear NOWHERE in the repo (verified by grep) --
   pretraining priors.
2. The repetition loop is the downstream effect: LangGraph expects a tool
   call, gets prose, and the conversation state degenerates.
3. The agent's temperature is 0.1 --
   not a sampling misconfiguration.

## Verification matrix (all PASSED -- stack exonerated)

Serving: onpremai/vllm-apertus-infomaniak:apertus15-79efecfb-p5, three-tier
checkpoint, merged chat template v3, 32k profile. Payload skeleton:

```
{"model": "apertus-v15-70b",
 "messages": [{"role": "system", "content": "You are a DevOps assistant for an AI node. Use the available tools to answer questions about the cluster. Always verify with tools instead of guessing."},
              {"role": "user", "content": "what model is deployed right now?"}],
 "tools": [control_plane_api, query_metrics, render_chart, get_current_time,
           wait_seconds, get_api_schema  -- six OpenAI-style function defs],
 "max_tokens": 200}
```

| # | Path | stream | temp | Result |
|---|------|--------|------|--------|
| A | engine-direct :8080 | no | 0.2 | finish=tool_calls, control_plane_api GET /api/models |
| B | engine-direct | yes | 0.2 | correct tool-call deltas (7), clean finish |
| C | engine-direct | yes | 0.7 | correct call (+ short leading text) |
| D | API gateway :80/v1 (Bearer key) | no | 0.0 | identical to direct at temp 0 |

Repro commands: POST to the engine's
:8080/v1/chat/completions with the payload above; for the gateway
path, POST to http://<api-gateway-host>/v1/chat/completions with an
Authorization Bearer key.

Note: a single gateway probe at temp 0.2 once produced continuation PAST the
tool call (hallucinated tool response + raw <|tools_prefix|> in content).
Not reproducible at temp 0 with identical payloads on both paths -- sampling
variance, i.e. the model sometimes fails to stop after the call even in the
simple scenario. Same underlying robustness issue, milder form.

## What makes the agent context different (untested individually)

The real agent request = 4.7 KB system prompt with inline worked examples +
cluster metrics map + 7 richly-annotated tools + LangChain schema conversion
+ multi-turn state. The model handles the simple version of the same
scenario perfectly; it breaks somewhere in that stack-up. The inline
examples are the prime suspect since the hallucination mirrors them.

## Assessment

Apertus-v1.5-70B-Prerelease-2607 is a PRERELEASE (base-flavored, no
"-Instruct" suffix). Simple tool use: reliable (verified repeatedly).
Heavy agentic prompting: fabricates tool sessions by imitating prompt
examples, and can fail to stop after tool calls. qwen36/gemma-4 are
agent-tuned; this checkpoint is not (yet).

## Reproduction of the FAILURE (for later retesting)

1. Deploy the card, set a DevOps agent chat client to model apertus-v15-70b
   (endpoint: the API gateway's /v1).
2. Ask: "what model is deployed right now?"
3. Failure signature: prose narration of tool usage, fabricated model names
   and token counts echoing the "Tokens by model" prompt example, repeated
   answer blocks, no real control_plane_api call in the thread.
4. Retest worthwhile when: (a) final v1.5 Instruct release lands, (b) the
   agent prompt gets an apertus variant without inline examples, or (c)
   repetition_penalty/frequency_penalty is set for this model.

## Recommendation

Keep qwen/gemma as the agent model; use Apertus 70B for chat, multilingual,
and vision (all verified good -- see infomaniak migration journal; one
arithmetic slip and the vision bottom-half issue are the known caveats).
Optional experiments, cheapest first: frequency_penalty for the loop
symptom; example-free agent prompt variant; final-release retest.
