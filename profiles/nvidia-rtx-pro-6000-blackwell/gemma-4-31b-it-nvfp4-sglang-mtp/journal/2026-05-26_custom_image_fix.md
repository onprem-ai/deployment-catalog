# 2026-05-26 — Replace broken upstream image with custom build

Upstream `lmsysorg/sglang:gemma4-mtp` was overwritten with a build containing
sglang#24912 regression (FrozenKVMTPVerifyInput.merge_batch AttributeError).
The original working digest is gone from Docker Hub.

## Fix

Built custom image `onpremai/sglang-mtp:bcf8d100`:
- Base: `lmsysorg/sglang:v0.5.12` (compiled C/Rust extensions, CUDA 13.0)
- Python overlay from commit `bcf8d100` (pre-EAGLE refactor, working FrozenKV MTP)
- Patches: `gemma4_assistant` model type registration in transformers CONFIG_MAPPING

## Smoke test

Deployed via catalog API, verified:
- 0 restarts, 131K context allocated
- MTP speculative decoding active (NEXTN algorithm)
- Chat completions working

## References

- Maturity changed: `broken` → `experimental` (pending further testing)
