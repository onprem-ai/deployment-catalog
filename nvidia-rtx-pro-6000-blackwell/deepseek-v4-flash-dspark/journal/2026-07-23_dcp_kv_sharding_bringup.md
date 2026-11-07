# DCP KV sharding + DSpark -- from an 11-round debugging bring-up

**Date:** 2026-07-23
**Basis:** a request for more KV capacity without giving up DSpark
decode speed. Result: GPU-validated experimental image
`onpremai/vllm-dsv4-dspark-dcp:2026-07-23-rc1`, recipe at
`catalog/rtx6k/_docker/vllm-dsv4-dspark-dcp/` (PATCH-NOTES.md there is
the full engineering log). NOT yet a catalog memory
profile (open items below).

## Measured numbers (2026-07-23, this node, rc1)

- FULL_AND_PIECEWISE cudagraphs + DSpark 5 + DCP=2, max-num-seqs 8:
  KV pool 1,577,120 tokens at the 1M profile (1.53x at max len);
  ~2,745,592 tokens / 2.66x with spec decode off
- Correctness 9/9 in one run: repeated short battery, 300k needle, and TWO
  ~1M needles through one server (queued through the pool correctly)
- C1 decode ~115 tok/s (vs 219 replicated; ~80 tok/s DCP spec-off). The
  gap vs replicated is per-step DCP collective cost over PCIe P2P
  (no NVLink), not a bug.
- Operating points: 219 tok/s @ 1.12x | 115 tok/s @ 1.53x | ~80 tok/s @ 2.66x

## The journey (11 rounds; each falsified hypothesis listed for the record)

Upstream vllm PR #44573 (DeepSeek-V4 DCP) applied cleanly to our pinned
tree but implements DCP ONLY for the FlashMLA backend; our SM120-only
FLASHINFER_MLA_SPARSE_DSV4 backend accepted DCP flags and silently
corrupted the KV cache (first fresh-server request correct, everything
after garbage -- eager mode too). We ported the full DCP integration
(subagent-driven; ~1,100-line follow-up patch) and then debugged, in order:

1. NaN algebra: a2a decode combine did acc += out*weight unguarded
   (IEEE NaN*0=NaN) fed by an UNINITIALIZED LSE buffer from the kernel
   wrapper; vllm's topk/topp sampler triton kernel spins FOREVER on NaN
   (device-side while loop). Fixed: -inf-prefilled LSE + partial sanitize.
   Lesson: 100% GPU util at ~100 W with no progress can be a spinning
   kernel, an NCCL wait op, or anything queued behind them.
2. CUDA async lies: three successive py-spy stacks blamed the sampler and
   even our own isfinite probe -- all just the first CPU sync point after
   the real offender. CUDA_LAUNCH_BLOCKING=1 is the tool that names the
   actual op (here: cuGraphLaunch of the target FULL graph).
3. Graph-capture audits (capture-safe buffers, rank-divergent collective
   counts via NCCL_DEBUG=TRACE diff -- sequences were IDENTICAL, protocol
   LL128-only, NCCL_GRAPH_MIXING_SUPPORT, async-scheduling off, dedicated
   drafter comm, drafter captured vs eager): ALL falsified. Kept as the
   record of what the deadlock was NOT.
4. Bisection beats deduction: a 4-arm op-swap kit
   (VLLM_DSV4_DCP_BISECT) convicted the CAPTURED
   torch.distributed all_to_all_single decode combine in ONE boot.
   Root cause: ProcessGroupNCCL host-side Work accounting freezes at
   capture; replayed device handshakes wait on state the host never
   advances. vllm's pynccl ops are capture-clean (that is why pynccl
   exists). THE RULE: no torch.distributed device collectives inside
   captured graphs on this stack -- pynccl only.
5. Drafter quality: DSpark drafter wrote draft KV with global slot
   addressing into DCP-sharded caches -> ~5% draft acceptance (rejection
   sampling kept outputs correct and only destroyed the speedup). Fixed by
   mirroring the target's cp-aware slot kernel; acceptance recovered to
   the 45-70% baseline ballpark.
6. Bonus fix: indexer decode did localize(ctx)//ratio instead of
   localize(ctx//ratio), dropping the NEWEST compressed entry per rank per
   token on C4A layers.

## Upstream-worthy findings

- vllm PR #44573: a2a-only dcp_comm_backend + raw FULL capture deadlocks;
  flashmla C4A decode reads a sharded compressed cache with unfiltered
  request-relative positions (confirmed bug)
- vllm sampler: topk/topp triton kernel should fail fast on NaN logits
- DSpark drafter slot mapping is not DCP-aware upstream

## Known constraints / to verify before catalog use

- Validated at max-num-seqs 8 / capture sizes <=48 only; sizes 80-192
  (max-num-seqs 32) untested since the NCCL fix (old hang there is
  expected to be the same root cause, but verify)
- Acceptance-rate windows varied 35-70% across workload phases; longer
  measurement wanted
- Catalog integration needs per-variant speculative-config (spec-off
  2.66x profile), registry mirror, kernel cache re-bake
- DCP variant must be opt-in: the decode-speed trade is real
