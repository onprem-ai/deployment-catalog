# Stable Promotion

**Date:** 2026-08-16
**Status:** Stable

## Decision

Promoted the RTX PRO 6000 Blackwell entry to `stable` after successful catalog
download, deployment, readiness, and authenticated capability validation.

The deployed card passed chat (6/6), vision (8/8), tools (12/12), and combined
vision plus tools (4/4), including streaming and non-streaming requests. The
checkpoint revision and runtime image remain pinned.

The current 24 GiB memory value remains the vendor-documented floor rather than
an on-node binary-searched minimum. This limitation remains documented and does
not change the successful deployment and capability evidence used for this
promotion.

## License provenance

The source repository declares Apache-2.0 in its model metadata but does not
bundle a separate license file at the pinned revision. The stable card therefore
bundles the canonical Apache License 2.0 text from apache.org and records its
SHA-256 checksum without inferring additional permissions.
