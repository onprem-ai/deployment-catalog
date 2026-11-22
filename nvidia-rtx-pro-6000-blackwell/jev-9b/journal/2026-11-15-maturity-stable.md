# 2026-11-15 - Maturity: Stable

**Decision:** The profile moves from `experimental` to `stable`, by the operator's decision.

**Basis:** every check the catalog review of 2026-11-15 asks of a stable profile holds:

- a full benchmark with no failed request: `2026-11-14-0753Z-llmperfbench.yaml` (max-prefill/1.0.0, max-decode/1.1.0, mixed-load/1.1.0; no failed request);
- measured GPU memory in `ai-deploy.yaml`;
- the image pinned by digest, a release build, not a nightly: `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`;
- no `--trust-remote-code` in the engine command.

**Known limits:** the replica serves at most 8 requests at once (`--max-num-seqs 8`): with more,
the decision module's LoRA path returns wrong System 1 probabilities (model card). AutoTrust
names the model itself experimental; this label is about the profile running as measured.
