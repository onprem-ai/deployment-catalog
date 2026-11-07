# 1M profile kv-cache-memory re-bisected on the rebuilt -v2 image -- 9.75 GiB -> 7.6 GiB

**Date:** 2026-08-04
**Basis:** the 9.75 GiB value from 2026-07-23 (see
journal/2026-07-23_kv_headroom_expansion.md) was carried over to the rebuilt
`onpremai/vllm-ds4f:0731-prebaked-v2` image (base bumped to
`ormandj/vllm-deepseek-v4-flash-sm120:v20`, DSpark TOPK=256 fix, vendored
vLLM PR #49059 empty-prefill-chunk patch) without re-validating. The replica crashed
under load at ~2026-08-04 with a genuine CUDA OOM inside FlashInfer's
`cutlass_fused_moe` -> `FusedMoeRunner::getWorkspaceInfo` (a transient
per-forward-call scratch workspace, separate from the reserved KV pool --
a different code path from the earlier-fixed allreduce-workspace bug).
Re-bisected per the 2026-07-23 methodology, on this node,
against the new image.

## Why the old value stopped being safe

Model architecture is unchanged (same weights, same TP=2 layout). What
changed is the software stack baked into the image: the v20 base image's
DSpark TOPK=256 fix alters speculative-decoding draft-token shapes, and the
FlashInfer/vLLM build's transient MoE workspace sizing depends on those
shapes and on the exact library build, not on the model. Confirmed the
rebuild shifted memory bookkeeping even at identical settings: at the
unchanged 9.75 GiB kv-cache-memory, the new image's actual KV pool measured
1,330,310 tokens vs the old image's 1,358,228 (~2% smaller for the same
byte budget).

## Bisection table (this node, -v2 image)

| kv-cache-memory | GiB | pool (tokens) | result |
|---|---|---|---|
| 10468982784 (prior value) | 9.75 | 1,330,310 | **OOM under load** (needed 348 MiB, 331.5 MiB free) |
| 9395240960 | 8.75 | 1,193,844 | booted, correct needle answer, but only 6 MiB free at peak -- unsafe |
| 8321499136 | 7.75 | 1,057,377 | booted, correct needle answer, 822 MiB free at peak |
| 7918845952 | 7.375 | n/a | **would not boot at all** -- vLLM startup `ValueError`: needs >= ~7.56 GiB KV reservation for `--max-model-len 1032192` |
| **8160437862 (chosen)** | **7.6** | **1,036,982** | **PASS -- 982 MiB free at peak, 0 restarts** |

## New finding: hard vLLM boot floor (not in the 2026-07-23 journal)

Below ~7.56 GiB, vLLM refuses to start at all for `--max-model-len 1032192`
(a clean `ValueError` at engine-core init, not a mid-inference OOM). This
means the real safe window on this image is only about [7.56, 7.75] GiB
wide -- 7.6 GiB sits ~40 MiB above that floor and was judged the best
stopping point: pushing lower risks the boot-time rejection for at most
~20 MiB of extra headroom.

## Validation

Two back-to-back full-depth (~978K token) needle retrievals on the same
live pod, no redeploy between them:
- `needle1m.json` (978,408 prompt tokens): correct answer, 982 MiB free
  at peak, 0 restarts.
- `needle1m-b.json` (978,212 prompt tokens), fired immediately after:
  correct answer, memory held flat at 982 MiB free, 0 restarts.

## Cost

Max context length is unchanged (`--max-model-len 1032192`, still ~1M
tokens per request). What shrank is concurrency headroom: the pool went
from ~1.29x max-model-len (room for overlap) to ~1.005x (room for
essentially one full-length request at a time).

## Rule of thumb (updated)

Same as 2026-07-23: leave >= 1 GiB per GPU free at peak prefill depth,
validate with a full-depth (~978K token) needle request. Now additionally:
check for the vLLM hard boot-floor `ValueError` first when probing lower
candidates -- it is a distinct failure mode from the mid-inference OOM and
appears abruptly rather than degrading gradually.

## Known constraints

- Value is specific to this image/config. Any base-image bump, patch
  change, or kernel-cache rebuild invalidates it -- re-bisect, do not
  carry the number forward blindly (this is exactly what went wrong here).
- The engine self-recovers from a prefill OOM (worker restart, ~2 min
  unavailable); the boot-floor failure crash-loops until config is fixed --
  bounded churn, not a brick, but should be caught quickly via pod status.
