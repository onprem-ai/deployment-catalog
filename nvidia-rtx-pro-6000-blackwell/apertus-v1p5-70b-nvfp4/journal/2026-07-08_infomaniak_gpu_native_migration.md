# 2026-07-08 -- Migration to the Infomaniak GPU-native multimodal image

## Outcome

The card now runs `onpremai/vllm-apertus-infomaniak:apertus15-79efecfb-p5`:
Infomaniak/vllm branch apertus-1-5
(upstream main @ 08a8a4af, post-v0.24.0, + 6 python-only apertus files) with
our patch stack. VERIFIED on the three-tier checkpoint, single RTX 6000
Blackwell, 32k profile:

- Tool calling: structured tool_calls via the NATIVE apertus parser (no
  plugin file) -- get_current_weather({"city": "Bern"}).
- Vision: image described correctly (shapes + colors) with encoders running
  in the GPU WORKER (profiler-visible; no --skip-mm-profiling, no OOM
  kills, memory limit unchanged at 16Gi).
- Boot: ~3m15s to ready, zero restarts.

## Why we migrated off the swiss-ai omni image

The swiss-ai fork (v0.19 base) loads the Emu3.5 encoder CPU-side in the API
process: the process was OOM-killed at 8-16Gi RAM limits, encoder VRAM invisible to
the profiler (needed util 0.80 + --skip-mm-profiling), no native tool parser.
The Infomaniak refactor (code at
github.com/Infomaniak/vllm@apertus-1-5) moves encoders into the GPU worker
and sits on a modern vLLM with the apertus parser built in.

## The branch was never run against its own base -- 5 patches + template merge

All in patches/0001-mm-input-required-args.diff (to be reported upstream to the Infomaniak branch):

1. `mm_input()` now requires mm_hashes + mm_placeholders (engine crashed at
   startup profiling). Added standard hashes + per-item PlaceholderRange.
2. `MultiModalFieldConfig.batched()` was called with FIELD names
   ("pixel_values") instead of MODALITY names ("image"/"audio"); and
   MultiModalKwargsItems built from a flat dict instead of from_hf_inputs
   -> "Modality 'audio' not found" crash.
3. Typo: `num_images=mm_options.get("image", 0)` (the options OBJECT)
   instead of `mm_counts.get(...)` -> "can't multiply sequence" crash.
4. PlaceholderRange location: our first approach (re-encode the layout
   string, subsequence-match) fails on BPE boundary effects; final approach
   anchors on the ATOMIC special tokens (<|img_start|>/<|img_end|>,
   <|audio_start|>/<|audio_end|>) with is_embed masks over dummy tokens.
5. THE VISION ROOT CAUSE: the model class does not define
   `get_placeholder_str`, so vLLM's chat serving silently inserted NOTHING
   for image content parts -- the image vanished from the prompt with no
   error. Added the classmethod ("<|image|>"/"<|audio|>").
   Also: audio waveform was fed float32 into bf16 conv weights (crash);
   added dtype cast mirroring the vision branch.

## Chat template: the two upstream template variants each fix half

- PR-1 (70B template variant): tool-calling fixes ('~' concat, dict-safe args) --
  but drops image/audio content parts entirely.
- PR-4 (8B template variant): multimodal template (<|image|>/<|audio|> emission,
  "Apertus 1.5 Omni" system prompt) -- but still has the '+' tool crash AND
  lacks the tool.function unwrap, so OpenAI-style tool declarations render
  empty (model never sees the tools; silent no-tool-call regressions).

The checkpoint now ships a MERGED template: PR-4 base + all PR-1 fixes +
tool.function unwrap + dict-safe rendering in the assistant-history path
(one more unsafe '+' PR-1 never covered). Backups in the checkpoint dir:
chat_template.jinja.orig (upstream main), .pr1-text-only (previous).

Lesson repeated for the third time this week: chat templates are code;
every vendored template needs the same review and testing as code.

## Checkpoint additions (from the 8B PR-4 variant, tokenizer-level = size-agnostic)

- emu35_vison_tokenizer.safetensors (1.8 GB; upstream filename typo intact)
- wavtokenizer_large_unify_600_24k.safetensors (848 MB)
- config.json: vision_config + audio_config + dummy token blocks merged
  (backup: config.json.pre-multimodal)

## Open items

1. RESOLVED (patch v6, image tag -p6): vision "bottom-half blindness" /
   vertical position collapse. Symptom evolution: bottom-right shapes were
   first mislocated to the top, then (3-quadrant image) reported missing,
   plus a hallucinated shape in an empty quadrant; an inverted-color probe
   showed bottom content RELOCATED to the top -- content perception intact,
   vertical geometry destroyed. ROOT CAUSE (one character class of bug):
   the Infomaniak layout wrote the image size header in PIXELS
   (f"{h_tok * 16}*{w_tok * 16}" -> "256*256") while the model's training
   convention -- and the swiss-ai reference (apertus_utils
   build_apertus_image_prompt) -- uses TOKEN GRID dims ("16*16"). The model
   believed its 16 rows were the top sliver of a 256-row image and mapped
   every shape to "top". Fix: header = f"{h_tok}*{w_tok}". VERIFIED after
   fix: 3-quadrant image (red TL, green TR, blue BR) and inverted image
   (blue TL, red BR) both described with exact positions; tools unaffected.
   Test images: test-image-3q.png
   and test-image-inverted.png. Headline item for the upstream report:
   silent, no error anywhere, destroys spatial grounding completely.
2. Audio still disabled (--limit-mm-per-prompt audio: 0). The dtype fix is
   in patch v5 but untested end to end -- next session: enable and test
   with a real wav.
3. Report the full patch stack + template findings upstream to the
   Infomaniak branch.
4. Image housekeeping: omni + p1-p5 tags accumulate ~28 GB layers each in
   docker + the local image registry; prune superseded tags.
5. Same-tag pitfall hit once: pushing a rebuilt image under an unchanged tag
   does NOT repull on the node (IfNotPresent) -- always bump the tag.
