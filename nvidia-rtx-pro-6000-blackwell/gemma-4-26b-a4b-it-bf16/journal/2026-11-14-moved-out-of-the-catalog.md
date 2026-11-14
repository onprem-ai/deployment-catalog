# 2026-11-14 - Moved Out of the Catalog: Its Weights Are Not in Model Storage

**Goal:** Benchmark the profile once on RTX PRO 6000 Blackwell (inference node `ain-xtg6o`
on platform-a4ec), as part of benchmarking every RTX PRO 6000 chat profile without one.

## What happened

Requesting the profile's downloads failed on the model weights, and the inference node
said a retry cannot help:

```
remote model directory does not exist or is empty: google--gemma-4-26B-A4B-it--model
```

Error report `01a1d988-e1b7-78c6-8553-1894dd3cf3c2` on platform-a4ec.

No inference node can download these weights, so no deployment of this profile can start.
Nothing was deployed.

## Outcome

The profile cannot be deployed as published, so it moved from `main` to the branch
`drafts/undeployable`. It returns to `main` by pull request once its weights are uploaded to OnPrem AI model storage under that name and a deployment answers.
