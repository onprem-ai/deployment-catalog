# DeepSeek V4 Flash 0731 cache warm startup hang

**Date:** 2026-08-03 (root cause confirmed 2026-08-04)
**Status:** Root cause confirmed; fix in progress (see Update 2026-08-04)
**Scope:** Upgrade from `DeepSeek-V4-Flash-DSpark` preview checkpoint to
`DeepSeek-V4-Flash-0731` official checkpoint on 2x RTX PRO 6000 Blackwell

## Goal

Warm and bake the SM120 kernel/JIT/autotune cache for the new 0731 checkpoint,
then build:

```
onpremai/vllm-dsv4-dspark:2026-08-03-prebaked
```

The checkpoint is documented by DeepSeek as having the same model structure as
`DeepSeek-V4-Flash-DSpark`; the official model card recommends DSpark with:

```json
{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}
```

The old validated preview configuration used 5 speculative tokens with
probabilistic draft sampling.

## Environment

- Host RAM after upgrade: 60 GiB total, ~50 GiB initially available
- Temporary swap: 30 GiB at `/persistent/swapfile-temp` (XFS), manually enabled
- GPUs: 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB each
- Driver: 595.71.05
- Base image: `onpremai/vllm-dsv4-dspark:2026-07-20-prebaked`
- vLLM: `0.23.1rc1.dev1263+gc233d90aa.d20260718`
- FlashInfer: `0.6.15.dev20260718`
- New model path:
  `/persistent/models/deepseek-ai--DeepSeek-V4-Flash-0731--model`
- Container limits: `--memory=45g --memory-swap=75g --ipc=host`
- k3s stopped during cache warming

## Procedure

The base image was started with an empty `/cache` bind mount and the catalog's
1M profile arguments, modified for the official model-card recommendation:

- `num_speculative_tokens=7`
- `draft_sample_method=greedy`
- `VLLM_ALLREDUCE_USE_FLASHINFER=0` (the workspace undersize workaround from
  `2026-08-03_flashinfer_allreduce_workspace_undersize.md`)

The cache directories were redirected under the empty `/cache` mount:

- `VLLM_CACHE_ROOT=/cache/vllm`
- `TRITON_CACHE_DIR=/cache/triton`
- `TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor`
- `TORCH_EXTENSIONS_DIR=/cache/torch_extensions`
- `TILELANG_CACHE_DIR=/cache/tilelang`
- `TVM_CACHE_DIR=/cache/tvm`
- `FLASHINFER_WORKSPACE_BASE=/cache/flashinfer`
- `CUDA_CACHE_PATH=/cache/jit/nv-compute`
- `XDG_CACHE_HOME=/cache`

## Observed progress

Startup successfully completed these stages:

1. Resolved `DeepseekV4ForCausalLM` and `DeepSeekV4MTPModel`
2. Loaded all target weights (79.29 GiB per TP worker)
3. Loaded the DSpark draft model (`96 params` in vLLM's loader summary)
4. Completed initial TileLang compilations:
   - `mhc_pre_big_fuse_broadcast_with_norm_tilelang`
   - `mhc_post_tilelang`
   - `mhc_pre_big_fuse_with_norm_tilelang`
5. Wrote 203 cache files

Last meaningful worker log:

```
(Worker_TP1 pid=602) ... TileLang completes to compile kernel
`mhc_pre_big_fuse_with_norm_tilelang`
```

After that, EngineCore printed only:

```
No available shared memory broadcast block found in 60 seconds. This typically
happens when some processes are hanging or doing some time-consuming work
(e.g. compilation, weight/kv cache quantization).
```

The message repeated for more than 80 minutes. The cache file count remained
at 203, so no compilation progress was occurring. The container remained
running, but `docker exec ... ps` itself did not return reliably, consistent
with the workers or container runtime being stuck.

## What this is not

### Not insufficient RAM in the final run

The earlier attempt with a 20 GiB docker memory cap was genuinely OOM-killed:

```
Memory cgroup out of memory: Killed process ... (VLLM::Worker_TP)
```

After the host RAM upgrade, the final attempt used 45 GiB plus 30 GiB swap.
There was no new OOM event in `dmesg`. It passed the prior OOM point and stayed
alive for more than an hour.

### Not weight loading

Both workers completed model loading successfully. The hang begins afterward,
during startup profiling / KV-cache initialization / dummy model execution.

### Not active compilation

The cache file count remained fixed at 203 for an extended period, and no
compiler, TileLang, Triton, DeepGEMM, or FlashInfer progress logs appeared.

### Not primarily a shared-memory queue defect

The `shm_broadcast` warning is a symptom: EngineCore is waiting for a TP worker
that has not returned. The underlying problem is in the worker-side startup
operation.

## Most likely root causes

### 1. Pinned vLLM predates the 0731 checkpoint

The image contains vLLM commit `c233d90aa` from 2026-07-18. The 0731
checkpoint and its recommended DSpark configuration were released 2026-07-31.
The architecture is unchanged, but that does not prove the older DSpark
implementation supports the new draft configuration and warmup behavior.

### 2. Cudagraph capture sizes are tailored to the old 5-token DSpark setup

The catalog's old validated capture sizes are:

```
5,6,10,12,15,18,20,24,25,30,35,36,40,42,48
```

Five speculative tokens produce six target verification positions. The new
seven-token setting produces eight verification positions. Expected batch
shape multiples include:

```
8,16,24,32,40,48
```

The existing list is missing `8`, `16`, and `32`. The pinned vLLM may therefore
enter an unsupported warmup/capture path for seven-token DSpark.

### 3. `draft_sample_method=greedy` may require newer vLLM behavior

The old deployment was validated with `5/probabilistic`; the official model
card recommends `7/greedy`, likely against a newer vLLM revision. The old fork
accepts the arguments, but acceptance does not establish startup correctness.

### 4. Worker-side GPU/collective deadlock in dummy forward

One TP worker may be blocked in a FlashInfer/DeepGEMM kernel, autotuning path,
or collective while EngineCore waits on shared memory. The absence of a Python
exception and lack of progress are compatible with a GPU kernel or NCCL wait.

## Next diagnostic

Retry the 0731 weights with the previous validated DSpark configuration:

```json
{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}
```

Keep the old validated cudagraph capture-size list unchanged.

Interpretation:

- **If 5/probabilistic boots:** the 0731 weights are compatible with the pinned
  image, and the failure is specific to unvalidated 7/greedy DSpark and/or its
  missing capture sizes. Then validate 7/greedy separately with corrected
  capture sizes or a newer pinned vLLM image.
- **If 5/probabilistic also hangs:** the 0731 checkpoint is not compatible with
  the pinned July 18 image despite its unchanged architecture. Update/rebuild
  the vLLM base image against a revision that explicitly supports 0731.

For a deeper worker-side diagnosis, run with `py-spy`/CUDA diagnostics against
each TP worker and identify whether it is blocked in host compilation, CUDA
synchronization, or NCCL. Do not treat repeating `shm_broadcast` warnings as
proof of ongoing compilation; verify cache file growth and worker stacks.

## Update 2026-08-04: Root cause confirmed

Root cause is an upstream FlashInfer kernel-dispatch gap, not RAM, not weight
loading, not the cudagraph capture-size list, and not specific to
`num_speculative_tokens=7`.

### Root cause

FlashInfer's SM120 sparse-MLA "DSV4" decode kernel is only instantiated for
`topk in {128, 512, 1024}` (`_DECODE_DSV4_DISPATCH` in
`flashinfer/mla/_sparse_mla_sm120.py`). DeepSeek-V4-Flash-0731's DSpark draft
issues sparse-MLA calls with `topk=256` (`dspark_markov_rank`). A `topk=256`
call misses the dispatch table and falls through into the prefill
orchestrator (`csrc/sparse_mla_sm120.cu`), which asserts `num_tokens > 64` and
rejects any decode-shaped call outright:

```
Check failed: num_tokens > 64 (7 vs. 64) : Decode (num_tokens <= 64) must go
through sparse_mla_sm120_decode_dsv4; got num_tokens=7
```

In our run this assertion appears to fire inside an async CUDA-graph-capture
context rather than surfacing as a clean Python exception, wedging the worker
without tripping `monitor_workers()` -- which is why EngineCore sees no
exception and just loops on "No available shared memory broadcast block
found" forever instead of crash-looping visibly.

**Confirmed via three independent, directly-fetched sources (not just search
snippets):**

- [flashinfer-ai/flashinfer#3828](https://github.com/flashinfer-ai/flashinfer/issues/3828)
  (2026-07-04) -- exact bug, exact assert text, exact hardware (SM120, TP=2,
  DeepSeek V4 + DSpark draft sliding-window layers).
- [flashinfer-ai/flashinfer#3989](https://github.com/flashinfer-ai/flashinfer/pull/3989)
  -- the fix (adds `topk=256` decode + prefill instantiations). Open/unmerged
  as of 2026-08-04. Equivalent split PRs:
  [#3817](https://github.com/flashinfer-ai/flashinfer/pull/3817) (decode) +
  [#3834](https://github.com/flashinfer-ai/flashinfer/pull/3834) (prefill).
- [vllm-project/vllm#50720](https://github.com/vllm-project/vllm/issues/50720)
  (open) -- another user hit this on the **exact same checkpoint**
  (`deepseek-ai/DeepSeek-V4-Flash-0731`) and **exact same hardware** (2x RTX
  PRO 6000 Blackwell, SM120). Explicitly reproduces with both
  `num_speculative_tokens=5` and `=7`.

This directly answers this journal's originally-proposed "next diagnostic"
(retry with the old validated 5/probabilistic config): **it would not have
booted either.** The bug is driven by the checkpoint's `topk=256`, not by
token count. Reverting to 5/probabilistic alone will not fix this.

The cudagraph-capture-size theory (root-cause candidate #2 above) is a dead
end -- verified against actual vLLM source (`config/compilation.py`,
`v1/worker/gpu/cudagraph_utils.py`): capture sizes auto-round up to the
nearest multiple of `num_speculative_tokens + 1` and gracefully fall back to
eager execution for any uncaptured batch shape. A missing capture size
produces a crash (`IndexError`, see vLLM PR #29091/#29102) or a slower path,
never a hang.

### Fix path

`ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120` -- the same upstream repo our
Dockerfile already builds `FROM` -- has since shipped a `:v20` tag
documented as the "DeepSeek-V4-Flash-0731 v20 validation image" for 2x SM120
GPUs. Their
[`UPSTREAM.md`](https://github.com/ormandj/vllm-deepseek-v4-flash-sm120/blob/main/UPSTREAM.md)
explicitly tracks "DSpark TOPK=256" (citing FlashInfer #3817+#3834) as one of
the patches this fork carries ahead of upstream merge.

Separately, [vllm-project/vllm#41834](https://github.com/vllm-project/vllm/pull/41834)
(SM12x enablement PR, validated 2026-08-02 specifically against 0731) states
that 0731 folded DSpark's draft into the main checkpoint and the **validated
config uses `num_speculative_tokens=5`**, not 7 -- the official model card's
"7/greedy" recommendation does not appear to be field-validated anywhere
found.

**Plan:** re-point the Dockerfile `FROM` digest at `:v20`, revert
`num_speculative_tokens` to 5 / `draft_sample_method` to `probabilistic`
(matching the only validated configuration found in the wild for this
checkpoint), keep the existing cudagraph capture-size list (already tuned
for 5-token DSpark), and re-run the cache-warm bake.

### Separate issue noted for later (not the hang)

A community recipe
([tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark))
documents that vLLM's DSpark draft weight loader silently drops tensors
specifically for the 0731 checkpoint, halving decode throughput with no
error and no quality drop ("Patch 4" in that repo). This would not cause a
startup hang, but if the cache-warm boots clean and measured throughput
looks roughly half of the ~479 tok/s previously seen, check this next.

## Update 2026-08-04 (continued): second bug found on first real generation

The `:v20` base image fixes the topk=256 hang confirmed above -- cache-warm-v2
booted clean (274s engine init, successful CUDA + DSpark graph capture,
`Application startup complete`, cache growing to 777 files). A healthy boot is
not proof of a working decode path, so a real chat completion request was sent
anyway. It crashed EngineCore immediately:

```
RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1]
```

Traceback bottomed out in FlashInfer's SM120 sparse-MLA decode path:
`flashinfer_trtllm_batch_decode_sparse_mla_dsv4` ->
`_trtllm_batch_decode_sparse_mla_dsv4_sm120` ->
`_trtllm_batch_decode_sparse_mla_sm120` -> `_normalize_sparse_mla_segments` ->
`_normalize_sparse_mla_indices_and_lens`, called from
`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`'s
`DeepseekV4FlashInferSM120Attention._forward_prefill`.

### Root cause

`_forward_prefill` chunks prefill work for the SM120 sparse-MLA kernel. Under
`FULL_AND_PIECEWISE` cudagraph padding, the trailing chunk can have
`query_start == query_end` (zero length). The chunk is not skipped, so
FlashInfer's `.reshape(batch_size * q_len_per_request, -1)` is called on a
0-element tensor and raises.

Researched from the full stack trace and version strings, then every
load-bearing claim was independently re-verified with direct `gh api` calls
before touching any code:

- [vllm-project/vllm#49059](https://github.com/vllm-project/vllm/pull/49059)
  "[Bugfix][DSv4][SM120] Skip empty sparse-MLA prefill chunks" -- confirmed
  open/unmerged via `gh api repos/vllm-project/vllm/pulls/49059`. Exact fix:
  skip the chunk when `query_start == query_end`.
- `ormandj/vllm-deepseek-v4-flash-sm120` carries this same patch at
  `patches/vllm/vllm-49059-sm120-empty-prefill.patch` (confirmed via
  `gh api repos/ormandj/vllm-deepseek-v4-flash-sm120/contents/patches/vllm`)
  but `Containerfile.v20` does **not** apply it -- confirmed by fetching and
  reading `Containerfile.v20` directly; it only applies
  `vllm-b12x-compressed-mla-workspace.patch`. The `:v20` tag fixed the
  topk=256 hang but never picked up this second, independent fix.

### Fix

Vendored the patch into this repo at
`_docker/vllm-dsv4-dspark/patches/vllm-49059-sm120-empty-prefill.patch`,
trimmed to only the `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`
hunk (a two-line `if query_start == query_end: continue` guard). The full
upstream patch also touches
`tests/kernels/attention/test_flashmla_sparse.py`, which does not exist under
`/opt/venv/lib/python3.12/site-packages` in the runtime image -- applying it
unmodified would have made `patch --batch` fail on the missing-file hunk.

Dockerfile now applies the patch between the base image and the cache COPY:

```dockerfile
FROM ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120@sha256:dc19ea2f75d8217cbb0fbc01d8734af79b02e84f0c9cda49e1c137db2c5c902b
COPY patches/vllm-49059-sm120-empty-prefill.patch /tmp/vllm-49059-sm120-empty-prefill.patch
RUN patch --batch --fuzz=0 -p1 -d /opt/venv/lib/python3.12/site-packages < /tmp/vllm-49059-sm120-empty-prefill.patch \
    && rm /tmp/vllm-49059-sm120-empty-prefill.patch
COPY vllm-dspark-cache/ /cache/
```

Layer ordering (patch before cache COPY) has no functional dependency --
patch touches `site-packages/vllm/...`, cache COPY touches `/cache/...`,
disjoint paths -- but is correct for Docker build-cache efficiency: the patch
layer is small and effectively static, the cache-copy layer is large and
changes on every re-warm, so keeping the static layer first lets Docker reuse
it across rebuilds. Confirmed empirically: the final build showed
`#7 CACHED` for the patch-apply layer, reused from an earlier throwaway
validation build (`localhost/vllm-dsv4-dspark-warm-stage:49059`) that applied
the same patch against the same base with no cache.

### Validation (cache-warm-v3, same reused cache dir)

- Patch applied cleanly against the `:v20` base in a throwaway verification
  image first: `patching file vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`,
  `Hunk #1 succeeded at 867 (offset 1 line)`.
- Re-launched cache-warm with the patched image, same cache dir (continuing
  from 777 files, not restarting from empty).
- Short completion: success.
- Long completion (768 completion tokens): success, `finish_reason: stop`.
- Concurrent (8-parallel) completions: success.
- Cache file count grew as previously-uncompiled code paths were exercised
  for the first time: 777 -> 801 -> 817 -> 1089 -> 1153, then stabilized at
  1153 across two further rounds of test requests (no further growth).

Final warmed cache (1153 files, ~42M) copied into the build context per the
README procedure (`sudo cp -a` + `chmod -R a+rX`), file counts verified to
match exactly between source and destination (1153/1153), guarding against
a partial copy.

Final image built: `onpremai/vllm-dsv4-dspark:2026-08-04-prebaked`.

### Verification (round 1, 1153-file cache)

Swapless (`--memory=20g --memory-swap=20g`), 20Gi-capped verification of the
first build ran as container `vllm-final-verify`. One invocation
snag along the way: the base image's default `ENTRYPOINT`
(`nvidia_entrypoint.sh`) and default `CMD` (`/usr/local/bin/run-kimi26-vllm`)
is a non-functional stub (`echo "Kimi launcher is not available in this vLLM
ref" >&2; exit 2`) -- running the container with vLLM flags as plain args
(no entrypoint override) fed those flags to the NVIDIA entrypoint's `exec
"$@"` and bash misparsed the leading `--model` flag. Fixed by launching with
`--entrypoint python3` and `-m vllm.entrypoints.openai.api_server <flags>`
explicitly, matching the invocation already used in the cache-warm scripts.
Startup succeeded (`STARTUP_SUCCESS mem=10.1GiB / 20GiB`), `/health` returned
200, and a chat completion succeeded (`finish_reason: stop`).

### Round 2: diverse code-sample prewarm surfaces 5 more first-time kernels

To prewarm with code samples across various programming languages before
treating the bake as final, a much wider battery
of requests was run against `vllm-final-verify`: Python/Rust code generation,
tool-calling (`get_weather` function), a reasoning-style word problem, JSON
mode, a long (single ~700-token) prompt, an 8-way and then a 16-way
concurrent batch, and a 10-language code-generation sweep (Go, Java, C++,
JavaScript, TypeScript, Ruby, Swift, Haskell, SQL, Bash).

This surfaced 5 Triton kernels that the 1153-file cache had never compiled,
each logged by vLLM's own `jit_monitor.py` as a "latency spike" warning
recommending the warmup be extended to cover the shape:

- `_build_c128a_topk_metadata_kernel`
- `_build_prefill_chunk_metadata_kernel`
- `_compute_prefill_metadata_kernel`
- `_prepare_dflash_inputs_kernel`
- `_compute_global_topk_indices_and_lens_kernel`

Running the same battery a second time (another 16-way concurrent round +
the 10-language sweep again) produced **zero** further new-kernel warnings --
confirming this 5-kernel set is the genuine ceiling for this request mix, not
an open-ended tail.

The container's `/cache` (grown from the baked 1153 files to 2084 files) was
extracted with `docker cp`, folded into the repo per the README procedure
(`sudo cp -a` + `chmod -R a+rX`, file counts verified 2084/2084), and the
image was rebuilt as `onpremai/vllm-dsv4-dspark:2026-08-04-prebaked-v2`. The
patch-apply layer was reused from Docker's build cache (`#7 CACHED`), as
expected from the layer-ordering decision above.

### Round 3: fresh-boot re-verification of the expanded cache

A brand-new container (`vllm-final-verify-v2`, same swapless 20Gi command,
same `--entrypoint python3` invocation) was booted from the `-v2` image and
run through the **entire** battery from round 2 again. Two findings:

1. **The same 5 kernels re-triggered their "JIT compilation during
   inference" warning on this fresh process**, despite the compiled
   artifacts already being present in the baked-in cache. Log timestamps show
   these warnings firing in the same wall-clock second as a request that
   completed with `200 OK` and immediately-normal `SpecDecoding` throughput
   (e.g. 20.72 tokens/s) right after -- i.e. no multi-second stall is visible.
   This indicates the warning is logged whenever a kernel takes the
   reactive/inference-time compile path (as opposed to the deterministic
   pre-`Application startup complete` warmup routine), regardless of whether
   the disk cache makes that path fast or slow. Disk caching is doing its
   job -- these are near-instant, not the tens-of-seconds-to-minutes
   first-compile events seen during the original cache-warm run -- but the
   warning itself cannot be eliminated by baking a cache; it would require
   changing vLLM's own deterministic startup warmup routine (`kernel_warmup.py`)
   to explicitly exercise these shapes, which is out of scope for a
   cache-bake.
2. **No kernel outside that same 5-item set appeared**, across the full
   repeat battery (8-way concurrent, 16-way concurrent, 10-language sweep,
   tool-calling, reasoning, JSON mode, long prompt). This confirms the
   -v2 cache is a genuine superset of what round 1 shipped, and that the
   5-kernel gap is a process-lifetime characteristic of this deployment, not
   evidence the bake is incomplete.

`/health` returned 200 and all 26 completion/tool-call/code-gen requests in
the fresh-boot battery succeeded (`finish_reason` of `stop` or `length` as
expected, one valid `tool_calls` response). Memory stayed at 11.2GiB / 20GiB
peak, comfortably under the swapless cap.

### Round 4: decode/prefill throughput measurement on the 0731 checkpoint

With the `-v2` image verified functionally, single-stream decode/prefill
throughput was measured against the running `vllm-final-verify-v2` container
(same 1M-context profile as the 2026-08-03 `~479 tok/s` measurement, but this
is the **first** clean throughput measurement of the 0731 checkpoint itself --
that number was on the older preview checkpoint before this upgrade, so it is
not a direct regression baseline).

**First attempt was wrong -- methodology bug, not a performance bug.** A naive
streaming client that increments a token counter per SSE chunk measured only
~60 tok/s decode. Investigating why led to
[`tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark)
(already referenced above), which documents exactly this trap: under
speculative decoding, vLLM emits at most one SSE chunk per decode *step*, not
per token, so counting chunks measures steps/s and under-reports by roughly
the mean acceptance length. Fixed by requesting
`stream_options: {"include_usage": true}` and dividing the server-reported
`usage.completion_tokens` by wall-clock decode time (first-token to last-token),
per that repo's explicit guidance ("Read `usage.completion_tokens`, or divide
server-side `vllm:generation_tokens_total` by wall time").

**Corrected single-stream measurements** (3 runs each, `bench_v2.py`):

| Test | num_words / max_tokens | Result |
|---|---|---|
| Decode | 50 / 500 | 162.5, 202.0, 318.4 tok/s (avg 227.6) |
| Prefill | 1500 / 100 | 8644, 8711, 9334 tok/s (avg 8896) |

**Checked the same repo's other finding (shared-expert draft-weight-loader
bug) against our own source -- found not applicable here.** That repo
documents a real upstream bug: their vLLM fork's DSpark
`_STACKED_PARAM_NAME_MAPPING` only remaps the two attention shards, so the
draft's always-on shared-expert `gate_up_proj` (fed by checkpoint tensors
`w1`/`w3`) silently drops 12 tensors per draft stage (`logger.debug`, invisible
at INFO), collapsing acceptance from ~60% to ~25.7% specifically on 0731.
Confirmed via direct read of our container's
`/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/dspark.py`
that our fork's equivalent table (`stacked_params_mapping`, line ~446) is
structurally different -- it maps generically (`("gate_up_proj", "w1", 0)`,
`("gate_up_proj", "w3", 1)`) rather than restricting to attention-only shards,
so it resolves `shared_experts.w1`/`w3` correctly as a side effect of the
broader match (the `.experts.` routed-expert path already `continue`s earlier,
so there is no collision). Also confirmed via the checkpoint's
`*.index.json` weight map that our 0731 copy carries the same 18
`mtp.{0,1,2}.ffn.shared_experts.w{1,2,3}.{weight,scale}` tensors that repo
describes -- so the precondition for the bug is present, but our loader
handles it correctly. **Our observed 25.8%-64.7% acceptance-rate swing across
requests is consistent with that repo's own documented content-driven
variance in a correctly-loaded drafter (33.7% prose reasoning to 78.3%
structured/repetitive), not evidence of the same bug.**

### Round 5: 6th JIT kernel surfaces from the benchmark script; re-verification battery re-run

Running `bench_v2.py` (Round 4) against the same long-lived
`vllm-final-verify-v2` container triggered a **6th** first-time
"JIT compilation during inference" warning, on top of the 5 already
documented in Round 2/3:

```
mhc_pre_big_fuse_broadcast_with_norm_tilelang   (TileLang JIT, not Triton --
  logged by the same jit_monitor.py:135 but a different backend)
```

This means the "5 kernels, converged" claim in the prior round's "Current
state" section was **not actually converged** -- it was only true for the
request shapes exercised up to that point. The num_words=1500/max_tokens=100
prefill shape in `bench_v2.py` (or possibly `temperature=0.0`, untested in
isolation) hit a code path the earlier battery never touched. The correct
fix is to add this shape to the warmup surface before the cache is called
final.

Re-ran the diverse battery against the same container to check for further
drift:

- Short chat, code-gen (Python), tool-calling, reasoning: all 4 succeeded
  functionally (`finish_reason` stop/length/tool_calls as expected per test;
  tool-calling produced a correctly structured `get_weather` call).
- Multi-language sweep (Go, Java, C++, JavaScript, Rust, 250-token cap each):
  all 5 produced valid, on-topic code, all hit `finish_reason: length` as
  expected at the token cap.
- JIT tally after this full re-run: **still 6**, same 6 kernel names as
  above -- no 7th kernel from either the battery re-run or the language
  sweep. First plausible "converged" reading, but only for the request
  shapes exercised across Rounds 2-5 combined; not a guarantee no other
  shape exists.

## Current state

- 0731 weights are present on disk.
- Catalog and Dockerfile changes: base image digest bumped to `:v20`, spec-decode config reverted to 5/probabilistic, and a
  vendored patch for the second (empty-prefill-chunk) bug added.
- `onpremai/vllm-dsv4-dspark:2026-08-04-prebaked-v2` has been built with the
  expanded (2084-file) warmed cache and the vLLM #49059 patch applied; a
  from-scratch swapless 20Gi verification of this exact image passed
  (startup, `/health`, and a 26-request diverse battery including
  tool-calling, reasoning, JSON mode, and 10 programming languages), plus a
  later re-verification round (Round 5) covering a 5-request battery and a
  5-language sweep against the same long-lived container.
- **6 kernels** (not 5) show a one-time "JIT compilation during inference"
  (or, for one, "TileLang JIT compilation during inference") warning on a
  fresh pod: `_build_c128a_topk_metadata_kernel`,
  `_build_prefill_chunk_metadata_kernel`, `_compute_prefill_metadata_kernel`,
  `_prepare_dflash_inputs_kernel`, `_compute_global_topk_indices_and_lens_kernel`,
  `mhc_pre_big_fuse_broadcast_with_norm_tilelang`. This is confirmed
  process-lifetime behavior (each fires exactly once per pod, not a caching
  gap), not blocking functionally, but the cache-warm script does not yet
  cover the request shape(s) that trigger the 6th kernel -- a candidate
  follow-up before calling the warmup surface complete.
- Do not deploy the catalog card until the final image tag is settled and
  pushed to the registry.
