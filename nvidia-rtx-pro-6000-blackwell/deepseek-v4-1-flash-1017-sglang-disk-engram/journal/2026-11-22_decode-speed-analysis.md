# 2026-11-22: decode speed on the 4x RTX PRO 6000 test host, and what disk Engram costs

Question: is single-stream decode slow (benchmark 65 tok/s) because the Engram rows are read
from disk?

Short answer: the disk is a real part of it, but not the whole gap. The same stack with the
tables in RAM is reported at about 2x our speed; most of that gap is probably the Triton
attention fallback, the small Engram cache and the SAS RAID latency under a deep queue.

## Setup measured
- Profile as deployed 2026-11-22 01:2x Z: SGLang image c4ca6511, sybil adapter (native AIO,
  64 reads in flight per store), DSV41_CACHE_GIB=8, SGLANG_SM120_FLASHMLA_BACKEND=triton,
  TP4/EP4, DSpark block 5, context 131072, max 8 running requests.
- Host: 4x RTX PRO 6000 Server Edition, 61 GB RAM, 10 GB swap. Weights on /dev/sda, an
  Adaptec SmartRAID 3204-8i logical volume of SSDs, ext4. Not NVMe.
- Disk baseline (Python O_DIRECT, 4 KiB random reads, 32 GiB file): QD1 p50 70 us, p99 87 us,
  13.6k IOPS; QD8 p50 78 us, 98.6k IOPS; sequential write 3.69 GB/s.

## Platform replica benchmark (2026-11-22 01:37Z, no failed requests at any level)
| Workload | c=1 | c=8 (plateau) |
|---|---|---|
| max-decode, total decode tok/s | 64.6 (TTFT 0.33 s) | 158.2 (TTFT 0.75 s) |
| max-prefill (~8k prompts), prefill tok/s | 1,149 (TTFT 5.7 s) | 1,394 |
| mixed-load, decode / prefill tok/s | 43.3 / 346 | 96.2 / 506 |
File: benchmarks/2026-11-22-0137Z-llmperfbench.yaml. Above 8 concurrent requests only the
queue grows (max-running-requests 8).

Ad-hoc gateway requests: prose 38-50 tok/s, code 99 tok/s (DSpark accepts more on code);
29,198-token prompt in 11.6 s (~2,500 tok/s) with the needle recalled.

## Disk during decode (one request, /proc/diskstats on sda, nvidia-smi every 500 ms)
- 853 prose tokens in 21.9 s (39 tok/s): 189,409 reads (8.5k/s), 805 MB, 4 KiB each, about
  222 reads per output token. Average read latency 902 us: 4 ranks x 64 in flight keep about
  256 reads queued on the RAID, against 70 us for a single read.
- GPU utilisation 89%, 204 W mean per GPU. Too coarse to rule out short stalls: a 500 ms sample
  counts as busy if any kernel ran.
- The reads run inside a cudaLaunchHostFunc callback on the CUDA stream, so every miss batch
  sits on the critical path of the decode step.
- Repeat test at temperature 0: run 1 719 tokens at 38.5 tok/s with 199 reads/token; run 2 693
  tokens at 42.0 tok/s with 159 reads/token. Outputs differed (DSpark sampling), so the cache
  helped only partly. 22% fewer reads gave 9% more speed: one noisy pair, but it suggests the
  disk is a sizable share of the step (a third or more), not a few percent.

## Reports from others (4x RTX PRO 6000 unless noted)
| Source | Engram | 1 request | Notes |
|---|---|---|---|
| commdata2338 benchmark, same sybil SGLang recipe, image of 2026-09-10 | locked in host RAM (189 GiB) | prose 112, code 191 tok/s; 146-170 tok/s in the recipe's matrix | 12-14% of table pages still not resident; 50-60 MB/s disk reads during decode |
| sybil-solutions README sweep | NVMe + 64 GiB cache, 128 GB host, FlashInfer path | 197-232 tok/s | 8,192 forced tokens of repeated synthetic text (high DSpark acceptance); not comparable to prose |
| sybil issue #1 (caiovicentino), FlashInfer + mask fix | NVMe | about 100 tok/s, about 140 on code; prefill about 6.4k tok/s at 29k | |
| HF discussion deepseek-ai/DeepSeek-V4.1-Flash #28 (aaron-newsome) | NVMe vs pinned RAM | RAM "only slightly faster", mostly with many sessions | anecdote |
| nktlabs/dsv41-flash-engram-disk-prestage (vLLM, 2 GPUs, EXL3 2.0bpw) | pinned RAM 113.3; disk + prestage + CUDA graphs 80.2; disk in eager mode 26.1 tok/s | disk at about 71% of RAM | remaining gap was per-step overhead, not row latency |
| 0xSero, sybil local-ai-recipe-kit PR #36: ghcr.io/sybil-solutions/vllm-moet-dsv41-engramdisk@sha256:f3d7a175e8ccb89507ca66974386e724734d0e7ed0080d21d5518daf88aca6bd | NVMe, 128 GB host | prose 137.6 tok/s (live traffic, noisy); prefill 8.9-11.1k tok/s | needs >= 32 GiB /dev/shm and about 47 GB host RAM; read mechanism not checked (io_uring would be refused here) |
| hanxiao recipe, LiL vLLM with ENGRAM_TABLE_MEMORY=disk | NVMe RAID | 166-211 tok/s | io_uring (b12x DiskTable); refused here by the default seccomp profile |

## Probable causes of our gap to about 112 tok/s, in order of likely size
1. Triton FlashMLA fallback. Every faster report uses the FlashInfer SM120 path. Here it
   still corrupts prompts > 64 tokens although PR #4's mask is in flash_mla_sm120.py.
2. Engram reads on the critical path: 8 GiB cache (others 64 GiB), SAS RAID latency rising to
   about 0.9 ms under 256 queued reads.
3. Tensor-parallel all-reduce over PCIe: SGLang disables its custom all-reduce for more than
   two PCIe-only GPUs, so every layer uses NCCL.
4. DSpark acceptance is lower on prose than on code (prose 38-50, code 99 tok/s here).

## Next steps
1. A/B with the platform benchmark: DSV41_CACHE_GIB 8 -> 16 (stays inside the 40 GiB limit;
   the container used 11-19 GB). A clear rise in c=1 decode means the disk is a main factor.
   At >= 11.5 GiB the scales stay resident and each miss costs 1 read instead of 2.
2. Find why the FlashInfer path still corrupts prompts > 64 tokens here: check that SGLang
   imports the patched flash_mla_sm120.py, bisect 62-72 tokens as in sybil issue #1.
3. Look at the vllm-moet-dsv41-engramdisk image: how it reads Engram (pread/AIO vs io_uring),
   and whether it runs with the node's shm (min(RAM/4, 32 GiB), about 15 GiB here) and a
   40 GiB memory limit.
4. Only then think about RAM: the reports above say RAM brings a small to moderate gain.
   256 GB of DDR5 RDIMM cost about EUR 10,000-12,000 in Europe in November 2026.

## Update 2026-11-22 morning: A/B results

### Engram cache 8 -> 16 GiB: no difference (correction)
- With 16 GiB the scales stay resident (732 MiB per store), so a miss costs 1 read, not 2.
- Average disk read latency during decode fell from about 900 us to about 200 us (4.5x).
  Reads per token fell only a little (about 199-222 -> 168-192).
- Prose decode did not change: 36.8 and 38.9 tok/s (before 38.5 and 39).
- max-decode benchmark, c=1/2/4/8: 59.2 / 93.2 / 128.5 / 163.1 tok/s
  (8 GiB: 64.6 / 99.0 / 123.2 / 158.2). Same within noise.
- Correction: the estimate above that the disk is "a third or more" of a decode step was
  wrong. A 4.5x faster disk gave no gain, so the disk is not the bottleneck on this host.
- A much larger cache (60 GiB) would not fit this host either (61 GB RAM; the container used
  28.8 GB with 16 GiB), and would attack a factor that does not limit speed.
- Cache set back to 8 GiB for RAM headroom.

### FlashInfer decode + Triton prefill: +45% to +140%
- Found: flash_mla_sm120.py dispatches calls with > 64 query tokens (prefill) to the
  FlashInfer prefill path and calls with <= 64 tokens (decode, DSpark verify) to the
  FlashInfer decode kernel. SGLANG_SM120_FLASHMLA_BACKEND=triton had moved both to Triton.
- Only the FlashInfer prefill path corrupts output here. New switch in the profile's copy of
  flash_mla_sm120.py: DSV41_SM120_PREFILL_BACKEND=triton sends only > 64-token calls to the
  Triton kernel and keeps FlashInfer for decode. The log shows
  "SM120 FlashMLA backend: flashinfer, prefill: triton".
- Correctness: 42 at 74 and 178 prompt tokens, Gotthard answers right, 29k needle recalled.
- Ad-hoc: prose 61.8 and 65.8 tok/s (before 37-39), code 165 tok/s (before 99).
- max-decode benchmark:
  | c | Triton everywhere | hybrid | change |
  |---|---|---|---|
  | 1 | 64.6 | 93.9 | +45% |
  | 2 | 99.0 | 171.3 | +73% |
  | 4 | 123.2 | 263.4 | +114% |
  | 8 | 158.2 | 379.7 | +140% |
- Host memory unchanged (25 GB of 63 GB under load).

### Remaining gap and next steps
- Prefill still uses Triton (about 1.4k tok/s at 8k prompts, 2.5k at 29k). The faster
  reports get 6-7k tok/s from the FlashInfer prefill path. Next: find why that path still
  corrupts here (check _split_kv_pages_to_64 with -1 indices and the extra-cache split; bisect
  62-72 tokens), so prefill can move back to FlashInfer.
- Single-stream decode (94) is now near the ~100 tok/s reported for this stack on NVMe.

## Plan: make FlashInfer prefill correct here (done: results in the next section)

Goal: prefill on FlashInfer instead of Triton. Others report 6-7k tok/s prefill with it; here
Triton gives about 1.4k tok/s at 8k prompts and 2.5k at 29k. Decode is not affected.

Known facts:
- Our log line "SM120 FlashMLA backend: flashinfer, prefill: triton" proves SGLang imports
  the profile's copy of flash_mla_sm120.py, so the PR #4 mask is in the module that runs.
- Calls with > 64 query tokens go to _flash_mla_sm120_prefill (FlashInfer prefill); <= 64 go
  to _flash_mla_flashinfer (decode). Only the prefill path corrupts. Boundary seen: 42 tokens
  correct, 74 wrong (sybil issue #1 saw <= 62 correct, >= 65 wrong).
- The mask only runs when topk_length / extra_topk_length is not None.

Steps:
1. Reproduce: redeploy with DSV41_SM120_PREFILL_BACKEND unset (or "flashinfer"). Test at
   temperature 0 with "Please read carefully." x n + "Answer with just the number: what is 19
   plus 23?" for prompts of 60-72 tokens (expect 42).
2. Compare the kernels on the same input: add a debug switch (for example
   DSV41_SM120_PREFILL_CHECK=1) to the profile's flash_mla_sm120.py that, for the first few
   prefill calls, runs both _flash_mla_sm120_prefill and flash_mla_sparse_decode_triton and
   logs the max absolute difference, whether topk_length / extra_topk_length are None, the
   index dtypes and shapes, and how many indices are -1. Short log lines (the console log
   keeps 500 characters per line).
3. Narrow down: run the check with (a) only the main KV source (extra source off),
   (b) only the extra (SWA) source, (c) without _split_kv_pages_to_64 where the page size
   already is 64. Suspects: the page split with -1 indices (touched_indices), the extra-cache
   split, a None topk_length that skips the mask, an index dtype mismatch.
4. Check the FlashInfer version in the image (0.6.18 per sybil issue #1) against upstream
   fixes flashinfer-ai/flashinfer#5075 and #5095. A newer wheel would have to be installed at
   container start (needs internet in the container; profile files are limited to 1 MiB).
5. Acceptance before switching the default: 42 at 74 and 178 tokens, the Gotthard reading
   test, the 29k needle (7319-ORCHID), then the max-prefill benchmark (target >= 5k tok/s at
   8k prompts) and max-decode (must stay at about 94 / 380 tok/s).

Safety while testing: each redeploy takes about 7 minutes of downtime; ask before taking the
model down. Watch host memory through the control plane metric
onprem_ai_inference_node_memory_used_bytes (it works without SSH access to the host). Do not trust the 40 GiB container limit alone: driver-mapped host memory is
not always charged to the cgroup (platform-alpha #375). vm.panic_on_oom is 0 on this host.
Keep the weights file selection unchanged, or a new 510 GB download starts.

## Result 2026-11-22 08:40-10:00Z: FlashInfer prefill A/B, and the dec64 path

Short version: FlashInfer's prefill kernel computes the same values as Triton, but serving
its output gives garbage in some processes, decided by the first FlashInfer prefill of the
process. Not fixable from the profile. The profile now runs prefill through the FlashInfer
decode kernel in 64-token slices (DSV41_SM120_PREFILL_BACKEND=dec64): correct in every test,
prefill about 2.5-3x faster than Triton. Draft-catalog revisions 0b641021 .. 5a8003d1.

How it was measured: SGLang's runtime output after READY is only readable through the
admin UI's live logs (Socket.IO with a browser session; an API key is refused), and the
deployment log closes at READY. The debug builds therefore ran SGLang on 127.0.0.1:8081,
sent real chat requests from container-files/selftest.py before READY, and only then opened
port 8080 through container-files/tcp_forward.py. A mode file in the container
(DSV41_SM120_PREFILL_MODE_FILE) switched the prefill path at runtime, and /flush_cache
between modes kept KV from one mode out of the next. DSV41_SM120_PREFILL_CHECK (or mode
"check") serves Triton and runs FlashInfer variants on the same inputs, logging the
difference per call (PFCHK lines).

Facts:
- Image: flashinfer 0.6.18, SGLang on its own CUDA stream (not the default stream).
- Kernel comparison (check mode, 560 calls of real requests, T = 120 .. 2048, chunked
  prompts and prefix-cache hits): FlashInfer prefill = Triton within FP8 noise (mean
  relative difference 0.6 %, no row over 5 %). Per call at T = 2048: FlashInfer prefill
  0.9 ms, dec64 7.0 ms, Triton 23.7 ms (rank 0).
- The PR #4 mask changes nothing here: lanes past topk_length were always -1 already.
- The same comparison in a process where served FlashInfer prefill gives garbage: still
  equal (359 calls, none over 5 %). So the kernel result is right; serving it is not.
- Served-output A/B, 18 prompts per mode (fixed and unique ones, up to 8k tokens, plus 4
  concurrent):
  | process start (warmup path) | triton | fi | fi_sync | fi_trilse | fi_fresh | fi_perpbs | dec64 |
  |---|---|---|---|---|---|---|---|
  | Triton warmup (rev e19592f7) | 11/11 | 10/11 then 11/11 | 11/11 | 11/11 | - | - | - |
  | FlashInfer warmup (rev 45ffff6a) | 18/18 | 4/18 | 7/18, 7/18 | - | - | - | - |
  | FlashInfer warmup (rev b384b0f6) | 18/18 | - | 7/18, 6/18 | - | 6/18 | 5/18 | 18/18 |
  fi_sync = device sync before and after each call; fi_trilse = FlashInfer output with
  Triton's LSE; fi_fresh = new split buffers on every call; fi_perpbs = separate persistent
  extra buffers per page size (the extra source has 256- and 128-token pages).
- A clean build with flashinfer_sync from the start (rev b49ff501) gave 6/32 on unique
  prompts. It served wrong answers for about 15 minutes before it was replaced.

Ruled out: the PR #4 mask, the missing LSE (dec64 also returns None and is correct), a
stream race (sync does not help), the wrapper's persistent split buffers (fresh buffers do
not help), shared state in the Triton kernel (its source has no shared buffers).

Probable cause (not proven): a side effect of the FlashInfer SM120 prefill orchestrator on
memory it does not own, set up on its first call. Which tensor it hits depends on the memory
layout at that moment; after a Triton warmup it was harmless by luck. This matches
flashinfer#5095 ("non-deterministic ... stale memory"), where the mask seemed to fix it.
Worth an upstream report with this data; a newer FlashInfer (with #5075) may fix it.

dec64 (profile default now): prefill calls with > 64 tokens go through
_flash_mla_flashinfer (the decode kernel, the same path that serves all decode steps) in
slices of 64 query tokens.
- Cold start (rev 5a8003d1): first request correct; 64 unique prompts at concurrency 4 and
  8: 64/64 correct (the same seed gave 6/32 with flashinfer_sync); 33.6k-token needle correct.
- End to end through the gateway, one request: 7k prompt 3.1-3.6k tok/s, 25k prompt
  4.0-4.3k tok/s (Triton prefill: about 1.4k and 2.5k; broken FlashInfer prefill: 3.2-4.9k).
- Platform benchmark (2026-11-22 ~10:05Z, rev 5a8003d1, measure_while_serving_clients,
  no failed requests), against the Triton-prefill run of 01:37Z / 08:02Z:
  | | Triton prefill | dec64 |
  |---|---|---|
  | max-prefill tok/s, c = 1 / 2 / 4 / 8 | 1149 / 1264 / 1331 / 1394 | 1592 / 1783 / 1883 / 1910 |
  | max-prefill plateau | ~1420 | ~2000 (at c = 12) |
  | max-decode tok/s, c = 1 / 2 / 4 / 8 | 93.9 / 171.3 / 263.4 / 379.7 | 98.4 / 192.6 / 267.1 / 400.7 |
  | TTFT p50 max-decode c = 1 | 0.348 s | 0.294 s |
  Prefill +40 % in the benchmark (parallel prompts, where MoE and Engram disk reads weigh
  more than in a single request); decode unchanged or slightly better. Host RAM peak about
  26 GB of 63 GB.

Next: report to flashinfer with the A/B table; when a newer FlashInfer is available in an
SGLang image, repeat the "FlashInfer warmup" A/B with the debug build (one redeploy).

## GPU booking: 94 GiB per GPU

hardware.gpu.memory_bytes_per_device is 94 GiB (was 95 GiB in the draft), the catalog's target
for the RTX PRO 6000 Blackwell Server Edition (97,887 MiB total, 1 GiB kept for the driver).
Measured with nvidia-smi during the dec64 max-decode benchmark: 84,667-84,685 MiB in use per
GPU (82.7 GiB; engine booking 80.75 GiB, --mem-fraction-static 0.85), so 11 GiB of margin.

## io_uring: not used, on purpose (decision 2026-11-22)

The Engram rows are read with native Linux AIO on O_DIRECT (64 reads in flight), not with
io_uring, and the platform is not asked to allow io_uring.
- Docker's default seccomp profile (Docker 29.10 here) is an allow-list that leaves out
  io_uring_setup, io_uring_enter and io_uring_register. The host allows io_uring
  (kernel.io_uring_disabled = 0). Using it would need a seccomp exception in the platform's
  container admission control (platform-alpha #193 refuses SecurityOpt).
- Docker, containerd and Kubernetes leave io_uring out on purpose: it had several kernel bugs
  that let code escape a container. An exception widens the kernel surface of every container
  that uses it.
- Little speedup is expected: native AIO with O_DIRECT already keeps 64 reads in flight, and
  the disk does not limit decode here (Engram cache 8 -> 16 GiB cut disk read latency from
  about 900 to 200 us with no change in decode tok/s). Prefill with long prompts is not measured.
- So there is no platform ticket. If an io_uring engine (the BRRR draft: vLLM + io_uring) ever
  looks worth it, measure it first on a test machine without admission control, where Docker
  accepts security_opt directly. Only a clear gain there would justify asking for the exception.

## Own image with warmed kernel caches (2026-11-22 afternoon)

The profile no longer carries code. The adapter, the patched flash_mla_sm120.py, the AIO row
reader (now compiled at build time), make_view.py and the debug tools are in the image
onpremai/sglang-dsv41-disk-engram:c4ca651-20261122 (build source: onprem-ai/platform-alpha,
dev/inference-engine-images/sglang-dsv41-disk-engram; the catalog convention is that a deploy
profile never carries build sources). The sections above that name container-files/ describe
the earlier layout; the files are now under /opt/dsv41/ in the image, with their licenses in
/opt/dsv41/licenses/.

build.sh ran the engine once on the test host with the command of compose.yaml and sent 22
checked requests (all correct), then committed the kernel caches into the image. Start of the
profile, request to READY, same host, same weights:

| | upstream image + container-files | own image |
|---|---|---|
| request -> READY | 5 min 51 s | 3 min 40 s |
| loading weights (main + DSpark draft) | 131 s | 134 s |
| FlashInfer autotune | 76 s + 8 s | 4 s + 3 s |
| CUDA graph capture (target + draft verify) | 45 s + 16 s | 4.4 s + 4.1 s |
| engine startup (scheduler_e2e) | 285 s | 163 s |
| kernel compile on the first request after READY | 18 s | none |

The log says the per-rank FlashInfer autotune caches "disagree" and are tuned again; with the
compiled kernels in the cache this takes 4 s. After the change: first request correct, 64/64
unique prompts correct at concurrency 4 and 8, decode unchanged (c = 1: code 148 tok/s, prose
66 tok/s; c = 8: 258 tok/s in total for the same prompts as before, 245 tok/s with
container-files).

## GPU memory per GPU (measured)

From the start of this profile from the catalog (2026-11-22 14:43 UTC) and nvidia-smi during the
max-decode benchmark:

| Part | GiB | Source |
|---|---|---|
| Weights (main model 72.51 + DSpark draft 2.05) | 74.56 | `Load weight end ... mem usage` |
| KV cache, 2,935,552 tokens | 5.12 | free memory 19.35 before, 14.23 after the memory pools |
| CUDA graphs | 0.27 | `Capture target verify CUDA graph end ... mem usage=0.27 GB` (draft: 0.00) |
| Runtime overhead | 2.75 | peak minus the three above |
| Peak | 82.70 | 84,685 MiB in nvidia-smi |

`--kv-cache-dtype fp8_e4m3` is now stated in the command; the dsv4 backend chose it before
already, so nothing changes in operation.
