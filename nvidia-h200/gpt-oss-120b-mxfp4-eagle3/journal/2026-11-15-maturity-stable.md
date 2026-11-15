# 2026-11-15 - Maturity: Stable

**Decision:** The profile moves from `experimental` to `stable`, by the operator's decision.

**Basis:** every check the catalog review of 2026-11-15 asks of a stable profile holds:

- a full benchmark with no failed request: `2026-11-11-2301Z-llmperfbench.yaml` (max-prefill/1.0.0, max-decode/1.0.0, mixed-load/1.0.0; no failed request);
- measured GPU memory in `ai-deploy.yaml`;
- the image pinned by digest, a release build, not a nightly: `vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e`;
- no `--trust-remote-code` in the engine command.
