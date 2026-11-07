# 2026-05-26 — Pin image to digest

Pinned container image from floating tag `nightly` to immutable `@sha256:` digest.

Floating tags can be silently overwritten by upstream maintainers — this happened with
`lmsysorg/sglang:gemma4-mtp` causing 23h+ of crash-loops.

No functional change — same image content, now immutable.
