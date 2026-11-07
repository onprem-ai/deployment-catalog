# Maturity stable

## Change

Set `maturity: stable` in `ai-deploy.yaml` at the operator's direction.

## Checks

- The one image, `lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`,
  is in the tag-plus-full-digest form stable profiles require.
- Nothing else in the profile changed: hardware, memory, runtime, and weights
  are as recorded in `2026-08-28_initial_catalog_entry.md`.
