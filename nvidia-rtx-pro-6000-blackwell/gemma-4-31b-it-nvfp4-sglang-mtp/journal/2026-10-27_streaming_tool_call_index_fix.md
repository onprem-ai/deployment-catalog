# 2026-10-27 - Fix streamed tool_calls[].index

## Problem

A client using the official openai SDK showed
that with `"stream": true` the `index` of streamed tool calls equals the tool's
position in the request's `tools` list instead of the call's position in the
response. Non-streaming responses were correct.

A reproduction script against this entry:

- Case A, `get_weather` + `get_time` (tools[0], tools[4]): streamed index `0, 4`;
  SDK stream helper crashed with `IndexError`.
- Case B, `get_weather` twice: streamed index `0, 0`; SDK merged both calls into
  `get_weatherget_weather` with concatenated arguments.

## Root cause

Upstream SGLang bug in `sglang/srt/function_call/gemma4_detector.py`: `tool_index`
comes from `self._tool_indices.get(name)` (tools-list position) instead of the
detector's call counter `self.current_tool_id`. Issue
[sglang#25073](https://github.com/sgl-project/sglang/issues/25073) (closed
inactive), fix PRs #23081 (closed stale) and #38198 (open). Present in our
`bcf8d100` overlay (byte-identical to v0.5.12) and still on SGLang `main` on
2026-10-27. Other parsers are unaffected; qwen3.8-27b (vLLM) passed the same
reproduction script.

## Fix

New pure-Python overlay image `onpremai/sglang-gemma4-mtp:bcf8d100-toolidx`
(`_docker/sglang-mtp-tool-index-fix/`) on top of the pinned `bcf8d100` digest. It
patches only `gemma4_detector.py` (same change as upstream PR #23081); the build
fails if the source does not match exactly.

Pinned digest: `sha256:9caf91e8424f2efa1c66ff7e7b725dd6587c48c270bb679eace751943543440f`.

## Verification

- Unit tests: `tests/control_plane/test_sglang_gemma4_tool_index_patch.py` runs the
  vendored upstream detector before (reproduces `0, 4` / `0, 0`) and after the
  patch (`0, 1`), for chunk sizes 1, 3, 7 and whole-text, plus non-streaming.
- Inside the built image (CPU, real sglang install): case A `0, 1`, case B
  `0, 1` with Berlin/Paris arguments intact; unpatched base image reproduced
  `0, 4` and `0, 0` with the Berlin arguments lost.
- Not yet verified on a GPU node with a full deployment. Next step: deploy this
  entry, stream cases A and B, and check that the streamed indices are `0, 1`.

## Not covered

`gemma-4-31b-it-nvfp4-mtp-tq` (experimental) uses
`onpremai/sglang-blackwell:gemma4-mtp-turboquant-swa`, which is not on Docker Hub
or this machine; it needs a GPU rebuild before the same patch can be applied.
