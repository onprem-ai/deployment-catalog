# 2026-08-15 - Capability Validation

**Goal:** Run the official capability-test API suites against the 1M deployment
after redeploying it with the final `qwen3.8-27b` served model ID.

## Deployment Identity

The card was redeployed through the model-catalog API so the running vLLM
process now uses:

```text
--served-model-name qwen3.8-27b
```

The InferenceService annotation `ai-node.io/model-id` also reports
`qwen3.8-27b`.

## Official Capability Results

Tests ran as Kubernetes Jobs through `/api/capability-test` against the KServe
Service, in both non-streaming and streaming modes.

| Suite | Result |
|---|---|
| Vision | 8/8 passed |
| Tools | 12/12 passed |
| Vision + tools | 4/4 passed |
| Chat, first run | 5/6 passed |
| Chat, immediate retry | 6/6 passed |

Vision covered color quadrants, OCR (`HELLO 42`), landscape recognition, and
animal recognition. Tools covered single selection, selection among multiple
tools, parallel calls, result round-trip, no-tool restraint, and narration
before a tool call. Combined tests confirmed image-derived structured tool
arguments and restraint when tools were unnecessary.

The first chat run returned an empty non-streaming developer-message response;
the same case passed on immediate full-suite retry, and all other chat cases
passed in both runs. This is recorded as a transient response rather than
hidden by reporting only the retry.

The earlier vLLM startup warning that the model would run text-only did not
match observed behavior: all eight official vision checks and all four
combined vision/tool checks passed after startup.

## Integrity Runner Mismatch

The API accepts `test_type=integrity`, but the digest-pinned backend image used
for capability Jobs contains an older runner that rejects it:

```text
ERROR: TEST_TYPE must be one of ('vision', 'tools', 'vision-tools',
'embeddings', 'chat'), got 'integrity'
```

This is a capability-runner image/version mismatch, not a model result. The
failed Job was deleted after the error was captured.

## Outcome

The official applicable suites pass: vision 8/8, tools 12/12, vision-tools
4/4, and chat 6/6 on retry. The final served ID is validated across those
suites. The transient empty developer-message response and unavailable
integrity runner remain explicitly documented.
