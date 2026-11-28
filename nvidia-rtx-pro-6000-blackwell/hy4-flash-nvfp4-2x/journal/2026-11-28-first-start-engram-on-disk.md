# 2026-11-28 First start, Engram tables on disk, tuning and benchmark

**Hardware:** 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB each), AMD Ryzen 9
9900X, 30 GB of host memory, NVMe (2 × WD_BLACK SN7100), driver 595.112.07. Nothing else on
the host.
**Engine:** vLLM main at 77766db with the SM120 overlays and this profile's `engram.py`
(`runtime/Dockerfile`). **Weights:** `atakans/Hy4-flash-NVFP4` at 5d9383f3: NVFP4 experts,
FP8 Engram tables with a BF16 scale per row. **KV cache:** FP8 (`fp8_ds_mla`, the model's
default in this build). **Drafter:** the checkpoint's DFly head.

## Why the tables are on disk

This host has 30 GB of memory. The SM120 recipe keeps the Engram tables in host memory and
measured about 77 GiB in its container; that does not fit. With the tables on the GPUs the
model takes 79.35 GiB per GPU and 5 to 6 GiB is left for the KV cache.

The checkpoint holds each table as one contiguous block of 96-byte rows (164,994,380 to
165,008,942 rows, 14.75 GiB each, shards 1, 4, 7 and 11). The disk patch of
ZackSample/hy4-flash-dgx-sparks applies cleanly to this image's `engram.py`: it keeps the
loader's file-backed view, and before each step it hashes the step's tokens as the model
will, reads those rows with `preadv` and leaves them in a GPU buffer, so the CUDA graphs are
unchanged. It supported BF16 tables only; here the staged value is the FP8 row times its
scale, in float32, as `FlashEmbedding.forward` computes it.

## Start (as a plain container, before the profile existed)

| | Tables on the GPUs | Tables on disk |
|---|---|---|
| Model loading, per GPU | 79.35 GiB | 49.35 GiB (51.67 with the DFly head) |
| KV cache per GPU / context | 6 GiB / 65,536 | 34 GiB / 262,144 |
| KV pool | 428,581 tokens | 3,277,817 tokens |
| Decode, one request, no speculation | 47.2 tok/s | 46.1 to 46.4 tok/s |
| 26,003-token prompt | 6.6 s | 7.0 s |
| Container host memory | 15.6 GiB | 15.3 GiB |

Both: 24 of 24 known answers, two parallel tool calls and an answer from their results.

## Is the disk reader exact

- Answers at temperature 0 cannot show it: two runs of one replica differ after 5 to 47
  tokens (mean logprob difference 0.002 to 0.05 on the shared tokens).
- **The rows:** inside the replica, 399,882 randomly chosen rows over the four tables, read
  with the engine's `_RowFile` from the view's file backing, compared with the same rows of
  the checkpoint: 0 differ. The staged value equals `rows.float() * scales` for all of them.
- **DFly draft acceptance** on one probe (decode probe and answer checks), by draft position:
  on disk 0.72, 0.49, 0.32, 0.20, 0.14; on the GPUs (65,536 tokens, 5 GiB KV cache) 0.72,
  0.47, 0.31, 0.21, 0.15. Decode 78 to 87 against 80 to 85 tok/s.

## Cold rows

Prompts of Python source files, 16 output tokens, prefix cache not used. "Cold": the Engram
shards dropped from the page cache first (`posix_fadvise(DONTNEED)`).

| Prompt | Cold | Warm |
|---|---|---|
| 24,598 tokens | 5.0 s, 4,925 tok/s | 4.1 s, 6,032 tok/s |
| 105,496 tokens | 26.3 s, 4,015 tok/s | 18.9 s, 5,584 tok/s |

## Rounds (through the platform)

| Round | Change | Result | Kept |
|---|---|---|---|
| 0 | 8 sequences, no DFly | ready in 5.3 min; decode 46 tok/s for one request | |
| 1 | DFly, 5 draft tokens | decode 78 to 87 tok/s; model 51.67 GiB per GPU; answers correct | yes |
| 2 | full benchmark, 8 sequences | `benchmarks/2026-11-28-0957Z-llmperfbench.yaml`: max-decode 425 tok/s at 32, plateau 12; first token after 14 to 62 s above 8 requests | |
| 3 | `--max-num-seqs 32` (probe at 1, 8, 16, 32, 48) | max-decode 92 / 405 / 657 / 887 / 952; mixed-load 63 / 228 / 294 / 388 / 392; max-prefill 2,988 / 4,356 / 4,663 / 4,669 / 5,038; first token 0.4 s at 32 | yes |

## Full benchmark on the final compose file

`benchmarks/2026-11-28-1227Z-llmperfbench.yaml` (compose `6c4dbf6d16b8…`), 90 minutes,
no failed request at any of 32 levels.

| Workload | 1 request | Plateau concurrency | Best |
|---|---|---|---|
| `max-prefill/1.0.0` | 2,893 prefill tok/s | 12 | 4,633 at 20 |
| `max-decode/1.1.0` | 93.0 decode tok/s | 40 | 969 at 48 |
| `mixed-load/1.1.0` | 61.4 decode tok/s | 40 | 401 at 32 |

- Draft acceptance over the run, by position: 0.60, 0.38, 0.28, 0.23, 0.20 (random-word
  prompts).
- After it: 24 of 24 known answers; tool calls; image input; a hidden code at 224,129 prompt
  tokens in 36.3 s (66 s on the 8-sequence replica, before the rows were cached).
- No engine error in the log.

## GPU and host memory

`nvidia-smi` every 30 s during the benchmark, and after the long prompt.

| Part | Bytes | GiB | Source |
|---|---|---|---|
| `model_weights` | 55,480,240,046 | 51.67 | `Model loading took 51.67 GiB`, the DFly head included, the Engram tables not |
| `kv_cache` | 36,507,222,016 | 34.00 | `--kv-cache-memory-bytes 34G`, 3,277,817 tokens |
| `cuda_graphs` | 1,245,540,516 | 1.16 | `Graph capturing finished … took 1.16 GiB` (1.10 on the other rank) |
| `runtime_overhead` | 7,722,846,126 | 7.19 | the peak minus the parts above |
| **Peak** | 100,955,848,704 | 94.02 | 96,279 MiB on GPU 0 after the 224K-token prompt (95,319 MiB during the benchmark) |

- Hardware booking 96,768 MiB: 489 MiB above the peak, 483 MiB below what an idle GPU has
  free. There is no room to spare; a lower `--kv-cache-memory-bytes` is the way to make some.
- Host memory: the container held 10.3 to 12.9 GiB of its 24G limit.

## Through the API gateway, and the image in the registry

- Through the API gateway (`https://<control plane>/v1`, an API gateway key): image input
  correct; 24 of 24 known answers; reasoning default, `no_think` and `high` correct; two
  parallel tool calls and an answer from both results; a hidden code at 53,030 prompt tokens.
- The image was pushed to Docker Hub as `onpremai/vllm-hy4-flash:77766db-sm120-diskengram1`.
  The registry answers the digest `sha256:355a4b10c1996838e423b55b47e16fe4dd03c149bd634d635056a668c363d214`,
  the one `compose.yaml` pins, so the compose file and its benchmark stay as they are.

## Not done

- A start on another CPU; a comparison with Tencent's FP8 weights.
- The benchmarks ran with `measure_while_serving_clients`: the service had no other caller,
  but the control plane counts every ready replica as serving clients.
