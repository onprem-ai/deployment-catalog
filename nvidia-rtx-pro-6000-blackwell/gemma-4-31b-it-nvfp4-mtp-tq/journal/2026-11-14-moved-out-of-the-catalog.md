# 2026-11-14 - Moved Out of the Catalog: Its Image Does Not Exist

**Goal:** Benchmark the profile once on RTX PRO 6000 Blackwell (inference node `ain-xtg6o`
on platform-a4ec), as part of benchmarking every RTX PRO 6000 chat profile without one.

## What happened

The profile's image `onpremai/sglang-blackwell:gemma4-mtp-turboquant-swa` is referenced by
tag only, without a digest, and Docker Hub does not have it:

```
Docker API POST /images/create returned HTTP 404: failed to resolve reference
"docker.io/onpremai/sglang-blackwell:gemma4-mtp-turboquant-swa": not found
```

Nothing was deployed.

## Outcome

The profile cannot be deployed as published, so it moved from `main` to the branch
`drafts/undeployable`. It returns to `main` by pull request once the image is published and pinned by digest in `compose.yaml`.
