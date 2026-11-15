# 2026-11-15 - Maturity: Stable

**Decision:** The profile moves from `experimental` to `stable`, by the operator's decision.

**Basis:** every check the catalog review of 2026-11-15 asks of a stable profile holds:

- a full benchmark with no failed request: `2026-11-13-2243Z-embedperfbench.yaml` (embed-text-query/1.0.0; no failed request);
- an embedding model: its benchmark measures the embeddings endpoint;
- the image pinned by digest, a release build, not a nightly: `ghcr.io/ggml-org/llama.cpp:dig-e8d36f4dc2a72a1d@sha256:e8d36f4dc2a72a1df323748f6219c9dd11f662f7cb3b06a6b2916c9bf3866d89`;
- no `--trust-remote-code` in the engine command.
