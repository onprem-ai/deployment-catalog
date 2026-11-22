# 2026-11-15 - Maturity: Stable

**Decision:** The profile moves from `experimental` to `stable`, by the operator's decision.

**Basis:** every check the catalog review of 2026-11-15 asks of a stable profile holds:

- a full benchmark with no failed request: `2026-11-13-1835Z-embedperfbench.yaml` (embed-text-query/1.0.0, embed-text-indexing/1.0.0; no failed request);
- an embedding model: its benchmark measures the embeddings endpoint;
- the image pinned by digest, a release build, not a nightly: `vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e`;
- no `--trust-remote-code` in the engine command.
