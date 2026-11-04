# Client-reported degenerate loop on 0731: investigation, one confirmed defect, symptom NOT reproduced

**Date:** 2026-08-08
**Reported by:** a client driving `deepseek-v4-flash` (0731 entry) from a
Claude Code setup through the ai-gateway. Same client/route as the earlier
preview-entry finding in
`../deepseek-v4-flash-dspark/journal/2026-07-24_anthropic_inline_system_message_leak.md`.

## The report

Intermittent ("sometimes gets stuck") degenerate loop in agentic coding use.
The pasted transcript shows three distinct symptoms together:

1. An **invented reasoning delimiter** `<reason>...</reason>`. This is NOT this
   model's format -- DeepSeek-V4 uses `<think>`/`</think>` and `｜DSML｜` tool
   markup. `<reason>` appears nowhere in this repo either, so it is neither a
   vLLM nor an ai-node tooling marker.
2. **Non-termination**: the model narrates "I'll output the tool call now"
   repeatedly, never emits a tool call, and says "I can't seem to stop."
3. **Character-level corruption**: confusable code points substituted for ASCII
   (`›` U+203A and `»` U+00BB where `>` belongs), a stray space inside a closing
   tag (`</ reason>`), dropped punctuation (`test_api_apps py`).

## What was done

Preview (`rtx6000bw--deepseek-v4-flash-dspark`) was undeployed and 0731 deployed
in its place on memory profile 0 (256k) for the session, then preview restored.
Preview was characterised first as a control before teardown.

## CONFIRMED defect: reasoning_effort levels are shifted one step, `high` is inert

This is a real, reproducible misconfiguration of the 0731 entry. It is **not**
the reported loop (see below), but it is worth fixing on its own.

DeepSeek implements reasoning effort as a **prompt prefix**. The 0731 checkpoint
ships a three-level table in `encoding/encoding_dsv4.py`:

| level | injected text |
|---|---|
| `low` | `""` (empty; this is `DEFAULT_REASONING_EFFORT`) |
| `high` | `"Reasoning Effort: Absolute maximum with no shortcuts permitted..."` |
| `max` | `"Reasoning Effort: Beyond maximum -- exhaustive, relentless..."` |

vLLM keeps its **own** copy of that encoder at
`vllm/tokenizers/deepseek_v4_encoding.py`. The copy in our 0731 image is the
pre-0731 **two-level** version: a single `REASONING_EFFORT_MAX` constant that
actually holds the reference's **`high`** text, injected only when the level is
`max`:

```python
assert reasoning_effort in ['max', None, 'high']
if index == 0 and thinking_mode == "thinking" and reasoning_effort == 'max':
    prompt += REASONING_EFFORT_MAX
```

The API-value mapping in `vllm/tokenizers/deepseek_v4.py` compounds it -- any
unrecognised string falls through to `"high"`:

```python
elif reasoning_effort in ("max", "xhigh"): reasoning_effort = "max"
else:                                      reasoning_effort = "high"
```

### Measured on this node, live, against the deployed 0731 (`/tokenize`, user msg "hi")

| requested | rendered prompt tokens | effectively equals |
|---|---:|---|
| omitted | 5 | vendor `low` |
| `low` | 5 | vendor `low` (accidentally correct) |
| `high` | **5** | vendor `low` -- **silent no-op** |
| `max` / `xhigh` | **84** | vendor **`high`** |
| vendor `max` | -- | **unreachable** |

These numbers match, exactly, an independent reproduction posted on the upstream
fix PR (5 / 5 / 84 on 2x DGX Spark GB10).

### Why this matters specifically for 0731

- The preview checkpoint documented only the two-level scheme, so the stale
  encoder was adequate for it. 0731 introduced the third level and renamed the
  texts -- **the encoder file is byte-identical between our preview image and our
  0731 image** (verified by diff), so this is a pre-existing vLLM gap that only
  starts mattering with 0731.
- 0731's published Code Agent benchmark numbers were measured at the `max`
  effort level. That level cannot be selected on this stack, so those numbers
  are not reproducible here.
- An independent SGLang report of the identical bug measured the downstream cost
  on a multi-turn tool-calling benchmark at roughly **-19 points** with the
  shifted mapping, silently: no error, no warning, thinking still produced, just
  much shallower.

### Upstream status (as of 2026-08-08)

- vLLM PR #50684 "Fix DeepSeek-V4 reasoning_effort 'high' tier..." -- **open, unmerged**.
  It splits the constant into `REASONING_EFFORT_HIGH` + a real `REASONING_EFFORT_MAX`.
- A reviewer notes the PR does **not** fix `low`, because the `low -> high`
  fallback lives in `deepseek_v4.py`, which the PR does not touch. So even once
  it merges, `low` would begin injecting the `high` prefix.
- SGLang carries the same bug (sgl-project/sglang#33185).

Nothing to fix in `inference.yaml` -- the defect is inside the pinned image.
Recorded here and in the YAML header so the next person does not re-derive it.

## NOT reproduced: the degenerate loop itself

Roughly 30 generations across the conditions below. **Every one was clean**:
correct tool call, `finish_reason=tool_calls`, no leaked markup, no repetition,
no confusable substitution.

| condition | result |
|---|---|
| Single-turn agentic edit request, tools, thinking on | clean |
| Effort levels none / high / max | clean |
| Multi-turn round-trip (assistant tool_call -> tool result -> "now edit") | clean |
| Same, with prior-turn `reasoning_content` fed back | clean |
| Cold chunked prefill: ~9.7k-token prompt, unique salt (no prefix-cache hit) | clean |
| 6-way concurrent cold burst | clean |
| Anthropic `/v1/messages` with inline mid-conversation `role: "system"` blocks | clean |

The last row is the important negative. The 2026-07-24 preview journal traced a
similar leak to vLLM's Anthropic endpoint preserving inline system messages in
place with no boundary (DeepSeek's `system_msg_template` is bare `"{content}"`).
That mechanism did **not** reproduce here on 0731.

Two observations worth carrying forward:

- With `thinking: true`, 0731 produced **zero** `reasoning_content` in every
  single test, at every effort level -- including `max`, whose injected prefix
  explicitly asks it to "write out your entire deliberation process". Requests
  finished in 100-145 output tokens. Whether that is correct for these short
  tasks or a symptom of the shifted effort mapping is unresolved.
- The upstream DSpark "garble" reports describe a *closely matching* symptom set
  (leaked tool/schema text, replies starting mid-word, leading special tokens,
  cold-only, recovering when warm) root-caused to spec-decode placeholders being
  attached to requests mid **chunked prefill**. Our config has all the
  ingredients (`--enable-chunked-prefill`, `--async-scheduling`, dspark k=5), and
  that report insists `num_speculative_tokens` is *not* the variable. This
  remains the best-supported hypothesis for the reported symptom, but it did
  not fire in the conditions tested here.

Because the mechanism is unproven, **no speculative config change was made**.
Changing `--speculative-config` or adding sampling overrides on a hunch would
have been unfalsifiable and, per the DSpark garble write-up, a documented
crash risk in the case of `repetition_penalty`.

## Round 2: leak / contamination hypothesis, also NOT reproduced

The first round varied *serving* parameters (effort, cold/warm, concurrency,
endpoint) but barely varied *request content* -- and a leak bug lives in the
content. Round 2 attacked that directly, on a freshly redeployed 0731.

The hypothesis was two-stage:

- **Genesis** -- the model invents a `<reason>` delimiter that appears nowhere
  in its input, because DeepSeek's bare `"{content}"` system template gives
  inline framework text no boundary.
- **Propagation** -- once one malformed `<reason>` exists in the history, the
  model sees it in its own context and keeps the pattern going, which would
  explain a stuck session arising from a single rare seed event.

Scored separately, because continuing a seeded pattern is NOT evidence of
genesis. 9 cases x 3 repeats = 27 generations:

| case | what it tests |
|---|---|
| A1/A2 | assistant turn pre-seeded with the reported verbatim corrupted `<reason>` text (batch + **streaming**) |
| B1/B2 | 6 heavy `<system-reminder>` blocks inline, OpenAI path (batch + **streaming**) |
| C1 | `role: "developer"` scaffolding inline |
| D1 | tool RESULT containing `<think>`, DSML markup, and a stale `<reason>` |
| E1 | assistant turn carrying framework text verbatim |
| F1/F2 | Anthropic path, heavy inline system, with and without contamination |

**Result: 27/27 clean.**

```
GENESIS (invented <reason> with no seed): 0
PROPAGATION (continued a seeded <reason>): 0
scaffold echoed verbatim:                 0
native markup leaked:                     0
runaway/looping:                          0
```

The propagation result is the informative one. Directly planting the corrupted
pattern in the assistant turn -- the easiest possible way to trigger a
self-reinforcing loop -- did not make the model continue it. It ignored the
contamination and did the task. That substantially weakens the
contamination-feedback theory.

A follow-up on self-closing invoke tags (a known dsv4 tool-parser bug class,
reachable via no-argument tool calls) was also clean: 6/6 correct tool calls
for a zero-parameter tool and an all-optional-parameter tool.

### The strongest remaining signal is a regime mismatch

Across ~60 generations in both rounds, 0731 emitted **a tool call with zero
prose content** (`clen=0`) in nearly every single agentic case, and **zero
`reasoning_content` at every effort level**. The reported transcript is the
exact opposite regime: all prose, repeated narration, no tool call ever emitted.

Nothing in this synthetic matrix produces that regime. Something about the real
session -- accumulated depth, the real client's tool schemas, or a failed
tool-call parse that returns markup as text and invites a retry loop -- moves
the model into a prose regime that these tests never reach. Chasing it further
with synthetic requests looks like poor value; the next move should be a
capture from the real client (see Open items).

## Capability tests: ran them, and they cannot catch this class

Existing suites against deployed 0731:

- `tools`: **5/5 PASS** (single, multi-select, parallel, round-trip, restraint)
- `chat`: **3/3 PASS**

So the existing suite would have called this model healthy. It is structurally
blind to the reported symptom:

- No suite inspects visible content for leaked `<think>` / `｜DSML｜` / BOS markup.
- No suite checks `finish_reason == "length"` (runaway generation).
- No suite checks for repetition; several cap `max_tokens` at 50-512, so a loop
  is silently truncated and still "passes" if a keyword happens to appear.
- `LlmApiClient.extract_content()` falls back to `reasoning_content` when
  `content` is empty, which *masks* reasoning/content split failures.
- Nothing exercises `reasoning_effort` at all.

### Added: `integrity` suite

New `OutputIntegrityTestSuite` in the capability test runner:

| check | catches |
|---|---|
| `no_leaked_markup_in_content` | prompt-format / parser markup in visible content |
| `generation_terminates` | runaway generation (`finish_reason == "length"`) |
| `no_degenerate_repetition` | repeated-8-gram loops |
| `no_confusable_substitution` | U+203A / U+00BB / U+2039 / U+00AB for ASCII |
| `reasoning_effort_levels_distinct` | accepted-but-inert effort levels |

Live result on 0731: **4/5, with `reasoning_effort_levels_distinct` FAILING**:

```
prompt_tokens omitted=5, low=5, high=5, max=84 -- COLLIDING LEVELS: ['low==high']
```

That is the confirmed defect above, caught automatically with an actionable
message. The check measures `prompt_tokens` because effort is a prompt prefix;
it reports "not supported" rather than failing when a model has no effort
support at all.

The first four checks all PASS on 0731 -- consistent with the loop not
reproducing. They are the regression net for it, not evidence of it.

Also fixed: `chat` and `integrity` were implemented in the runner but missing
from `VALID_TEST_TYPES` in the capability test API, so `chat`
was unreachable through the API. A test now asserts the allow-list does not
drift from the runner's dispatch table.

Unit tests: 20 tests, all passing; full backend suite 2715 passed / 1 xfailed; ruff clean.
Note the suite source uses `\uXXXX` escapes rather than literal confusables --
ruff RUF001 rejects the literals under this repo's ASCII rule.

## Open items

1. **Loop mechanism still unknown.** The next reproduction attempt should come
   from the real client, not a synthetic harness: capture the exact
   failing request (full `messages[]`, `chat_template_kwargs`, sampling params)
   plus `/v1/chat/completions/render` output for it. This vLLM build exposes
   `render` / `derender`, which dumps the rendered prompt -- that will show
   directly whether the prompt fed to the model was corrupted, which would
   confirm or kill the chunked-prefill hypothesis in one shot.
2. **Effort mapping.** Re-check when vLLM #50684 merges; a base-image bump is
   the only fix path. Until then treat `max` as vendor-`high` and do not quote
   0731's `max`-level benchmark numbers for this deployment.
3. **Zero reasoning_content** at all effort levels deserves its own look; it may
   be the same encoder gap or something separate.
4. **max-model-len vs effort.** The vLLM recipe states Think Max wants
   `--max-model-len >= 393216`; memory profile 0 is 262144. Profile 1 (1M) is
   above it. Not a factor while vendor `max` is unreachable, but it becomes one
   the moment the encoder is fixed.
