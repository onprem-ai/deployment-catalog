# 2026-07-04 — Pin sources images to match deploy-spec digests

Pinned the `sources.container.images` entries to the same `@sha256:` digests
already used by the deploy-spec `image:` fields, instead of the floating
`latest-nvidia-gpu-sm120-offline` tag.

The new deploy-time local-registry rewrite matches sources entries against
deploy-spec image strings by exact equality. A sources entry using a floating
tag while the deploy-spec uses a digest no longer matches, so the rewrite
would silently skip this catalog entry. Keeping both strings byte-identical
is now a hard precondition, not just good hygiene.

No functional change — same image content, sources and deploy-spec now refer
to it identically.
