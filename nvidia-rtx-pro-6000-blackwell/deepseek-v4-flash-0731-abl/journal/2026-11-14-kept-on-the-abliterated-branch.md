# 2026-11-14 - Kept on the `abliterated` Branch, Not in `main`

**Goal:** Decide where an abliterated model belongs in the catalog.

## Why not `main`

The official catalog on `main` is what operators deploy with one click, and every profile
there reads as recommended for general use. This model is not meant for that:

- Abliteration removes the direction in the model's activations that makes it refuse. It
  answers requests the original DeepSeek V4 Flash 0731 refuses, including clearly harmful
  ones, and nothing else in its safety training replaces that.
- The weights come from an individual uploader (`cebeuq`), not from DeepSeek. What else
  changed is not checked; the answer check on 2026-11-14 covered one arithmetic question.
- The profile runs with `--trust-remote-code` and downloads `encoding/encoding_dsv4.py` from
  the same uploader (the checker warns: `model.weights-executable-files`). The original 0731
  profile runs without both.

## Who it is for

Experienced users who want a model without refusals on purpose: security research and red
teaming (for example writing and analysing exploit code), and teams that put their own
filter in front of the model. It works on two RTX PRO 6000 Blackwell GPUs, with its full
replica benchmark of 2026-11-14 (journal `2026-11-14-full-benchmark.md`).

## How to use it

Add this branch as a catalog of its own (the repository with the branch `abliterated`).
Profiles here are kept to the same standard as `main`: benchmarked, measured memory, a
clean profile check apart from the warning above.
