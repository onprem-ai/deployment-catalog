# 2026-11-15 - Maturity: Stable

**Decision:** The profile moves from `experimental` to `stable`, by the operator's decision.

**Basis:** every check the catalog review of 2026-11-15 asks of a stable profile holds:

- a full benchmark with no failed request: `2026-11-13-1813Z-embedperfbench.yaml` (embed-text-query/1.0.0, embed-text-indexing/1.0.0; no failed request);
- an embedding model: its benchmark measures the embeddings endpoint;
- the image pinned by digest, a release build, not a nightly: `nvcr.io/nvidia/vllm:26.02-py3@sha256:1bec659df62970ce856523b67868ab9f92cf947946ee5e89cc6a9cabe9d783da`;
- no `--trust-remote-code` in the engine command.
