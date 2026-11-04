# Stable Promotion

**Date:** 2026-08-16
**Status:** Stable

## Decision

Promoted the RTX PRO 6000 Blackwell entry to `stable` after successful catalog
download, deployment, readiness, and authenticated application-path validation.

Validated inputs include a base64 image, an image URL with returned boxes, and
a two-page PDF. The deployment remained Ready with zero restarts. Both model
revisions and the runtime image remain pinned.

Broader multilingual, industrial, latency, and throughput testing remains useful
future characterization work rather than a blocker to the validated OCR paths.

## License provenance

Both PaddlePaddle source repositories declare Apache-2.0 in their model metadata
but do not bundle separate license files at the pinned revisions. Each stable
source record therefore points to and bundles the canonical Apache License 2.0
text from apache.org with an independently verified SHA-256 checksum.
