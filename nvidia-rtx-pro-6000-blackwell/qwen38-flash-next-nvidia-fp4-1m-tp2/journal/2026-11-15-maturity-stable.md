# 2026-11-15 - Maturity: Stable

**Decision:** The profile moves from `experimental` to `stable`, by the operator's decision.

**Basis:** NVIDIA's official checkpoint pinned by revision; the image NVIDIA's model card names,
pinned by digest; answers and parallel tool calls checked; a hidden number found 21 of 21 up to
1,038,713 tokens; a full replica benchmark with no failed request; peak GPU memory within the
hardware booking; a clean profile check (journals of 2026-11-15).

**Known limits:** the image is a community build (`ghcr.io/local-inference-lab/vllm`) until
upstream vLLM reads this checkpoint's NVFP4 scale format; context beyond the trained 262,144
tokens comes from YaRN rope scaling, checked by retrieval, not by reasoning tasks at 1M.
