# Stable Promotion

**Date:** 2026-08-16
**Status:** Stable

## Decision

Promoted the RTX PRO 6000 Blackwell entry to `stable` after successful catalog
download, deployment, readiness, authenticated chat and tool validation, and a
measured single-user decode result of 181.18 tokens per second.

The target checkpoint, DFlash assistant, and SGLang image remain pinned. The
known parallel multi-call limitation remains documented; single tool calls and
most tool scenarios passed. The first working 75 GiB profile is retained as a
known-working allocation rather than represented as a measured minimum.

VRAM-floor bisection, valid prefill measurement, concurrency characterization,
long-context testing, and DFlash acceptance measurement remain follow-up
performance characterization tasks.

## License provenance

Both pinned Hugging Face repositories contain the same Apache License 2.0 text.
The card now records independent source-specific metadata and checksum-verified
offline copies for the target checkpoint and DFlash assistant.
