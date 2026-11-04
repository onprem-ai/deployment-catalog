# 2026-07-09 -- Template default-value gap found + CI render test; NVFP4 repo prepared

Preparing the NVFP4 (MLP-only W4A16) checkpoint for upload to
`onprem-ai/Apertus-v1.5-70B-Prerelease-2607-NVFP4` surfaced two things: the
checkpoint was a stale Jul 7 snapshot missing every Jul 8 fix, and the Jul 8
merged template itself had a gap.

## 1. The 2026-07-08 merged template missed three PR-1 fixes

The merge journal (2026-07-08_infomaniak_gpu_native_migration.md) claims the
checkpoint template is "PR-4 base + all PR-1 fixes". Diffing against PR-1
verbatim (kept at bf16-assembled/) shows the tool-parameter DEFAULT-VALUE
block (render_tools macro, ~line 117) kept upstream's '+' concatenation in
all three branches (enum / oneOf / else). Consequence: a tool schema with a
oneOf parameter and a non-string default (or a numeric enum default) crashes
every tools request with the familiar
`can only concatenate str (not "dict") to str`.

Nobody noticed because the tool requests tested so far never sent that schema
shape. Verified by rendering in jinja2's ImmutableSandboxedEnvironment (the
engine transformers/vLLM use): the Jul 8 template and upstream main both
crash on a hostile toolset; the corrected template renders it.

Fix: PR-1's exact edits ('~' plus the mapping/sequence tojson guard),
annotated `{# Edits: ... #}` like the rest.

## 2. Templates now live in git with a CI render test

Root cause of the recurring template incidents (4th this week): the unified
template existed only as hand-patched copies inside checkpoint directories,
no git, no tests. Fixed in commit `151f1598`:

- canonical template: `catalog/rtx6k/apertus-v15-70b-nvfp4/chat_template.jinja`
- regression test: `tests/unit/test_catalog_chat_templates.py` -- renders in
  ImmutableSandboxedEnvironment against non-string defaults, OpenAI-wrapped
  tools, dict arguments in assistant history, multimodal parts. TDD: fails
  against the Jul 8 template with the observed TypeError, passes against
  the corrected one.
- jinja2 added as backend dev dependency (test-only).

Synced to all local checkpoints (all md5 9f851113...): ThreeTier,
FP8-Dynamic, NVFP4 staging, bf16-assembled. Previous versions kept as
`chat_template.jinja.jul08-merge`. Deployed ThreeTier service restarted
to load it.

## 3. NVFP4 checkpoint brought to Jul 8 parity for upload

The nvfp4-mlp-only output (Jul 7 00:33 build) predated all Jul 8 fixes.
Applied before upload:

- corrected chat template (above)
- `tokenizer_class`: TokenizersBackend -> PreTrainedTokenizerFast
  (transformers-v5 name fails on older stacks; same fix ThreeTier got)
- multimodal bundle from ThreeTier: emu35_vison_tokenizer.safetensors,
  wavtokenizer_large_unify_600_24k.safetensors, config.json
  vision/audio/dummy-token blocks

Backups moved OUT of the upload folder (hf upload-large-folder has no
exclude filter) to a separate template-backups/ directory.

## 4. Upload gotcha: hf_xet OOM

`hf upload-large-folder` (huggingface_hub >= 1.22 with hf_xet) buffers Xet
chunks natively and OOMs hard on constrained memory:

- in a container with a 1Gi memory limit: the process was OOM-killed after
  ~16 min -- do NOT run uploads in a memory-constrained container
- on the host in a systemd scope (MemoryMax=4G): oom-kill in 34 s
  (MemoryPeak 4.0G, journalctl Result=oom-kill)

Working recipe (host, with a memory cap):

    systemd-run --user --scope -p MemoryMax=8G \
      sh -c 'HF_HUB_DISABLE_XET=1 hf upload-large-folder <repo> <dir> \
        --repo-type=model --num-workers=2'

HF_HUB_DISABLE_XET=1 forces the classic streamed LFS multipart path
(memory-flat). A small uv venv (29 MB) holds the hf CLI for this.

## Open items

1. Upstream PR: ship the corrected union template + multimodal bundle to
   the 70B repo; report the default-value gap in PR-1's template upstream.
2. Consider a catalog-mounted `--chat-template` for this card (like the 8B
   FP8 card) so serving cannot drift from git even if the model dir copy is
   replaced by a re-download.
3. The 8B repo's merged multimodal template (PR-4) still lacks ALL PR-1
   fixes upstream -- same crash class for anyone using it with tools.
