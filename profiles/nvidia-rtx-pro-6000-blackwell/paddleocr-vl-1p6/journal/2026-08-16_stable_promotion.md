# Stable Promotion

**Date:** 2026-08-16
**Status:** Stable

## Decision

Promoted the NVIDIA RTX 6000 Blackwell hardware entry from `experimental` to
`stable` after the first deployment and authenticated API validation documented
in the preceding journal entry.

The stable card preserves the validated pinned model revision and container
image digests. Source-specific Apache-2.0 provenance was also added using the
exact `LICENSE` file from the pinned Hugging Face revision, with its SHA-256
checksum verified locally.

## Stable source provenance

- Repository: `PaddlePaddle/PaddleOCR-VL-1.6`
- Revision: `c5630abae1d940eafe0697512a0325494b02ab42`
- License expression: `Apache-2.0`
- Bundled document: `licenses/source-0-LICENSE.txt`
- SHA-256: `b8c4d7deccd236af023af1c88c4d4e8f0fc2f41914e0fb23a3ec9678fb5a8456`
