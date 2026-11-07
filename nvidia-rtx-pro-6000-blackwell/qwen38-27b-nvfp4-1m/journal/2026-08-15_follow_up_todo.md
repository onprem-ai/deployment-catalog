# 2026-08-15 - Follow-up TODO

**Goal:** Record the remaining validation work that is intentionally deferred
before promoting the 1M card beyond experimental maturity.

## TODO: Measure Minimum VRAM

The current `min_gi: 84` is the explicit allocation used for successful 1M
bring-up, capability validation, and performance benchmarking. It is not the
minimum working allocation.

Run the documented VRAM binary-search procedure through the model-catalog API:

1. Start from the known-working 84 GiB allocation.
2. Reduce the allocation and redeploy until startup or a full-context request
   fails.
3. Narrow the boundary to the smallest allocation that starts successfully and
   can serve a 1,010,000-token request.
4. Record vLLM memory accounting, actual `nvidia-smi` usage, KV token capacity,
   and the first failing allocation.
5. Replace `min_gi: 84` with that measured minimum and update the profile name
   so it no longer says `validation allocation`.

Do not mark the card stable until this bisection and a genuine near-1M-token
request have passed.

## Outcome

The remaining VRAM and full-context validation work is explicitly tracked in
the model journal. The card stays experimental in the meantime.
