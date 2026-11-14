# 2026-11-14 - Moved Out of the Catalog: Its Start-Up Installs a Closed vLLM Pull Request

**Goal:** Benchmark the profile once on RTX PRO 6000 Blackwell (inference node `ain-xtg6o`
on platform-a4ec), as part of benchmarking every RTX PRO 6000 chat profile without one.

## What happened

The command installs vLLM at every start from
`git+https://github.com/vllm-project/vllm.git@refs/pull/38135/head` (with
`VLLM_USE_PRECOMPILED=1`) on top of `vllm/vllm-openai:v0.17.1`. vllm-project/vllm#38135
("Support for nvidia/gpt-oss-puzzle-88B") is closed and was not merged. The install leaves
vLLM without its compiled extension, the engine stops while loading, and the replica restarts
into the same install:

```
ModuleNotFoundError: No module named 'vllm._C'
```

Error report `01a1da96-52b2-737d-ad57-d746d613eb55` on platform-a4ec. The replica was removed.

## Outcome

The profile cannot be deployed as published, so it moved from `main` to the branch
`drafts/undeployable`. It returns to `main` by pull request once it runs from a pinned image (a vLLM release that supports the model, or one built once from the pull request's last commit) with no installation at start.
