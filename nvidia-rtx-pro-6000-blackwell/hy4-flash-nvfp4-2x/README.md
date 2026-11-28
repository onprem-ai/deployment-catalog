# Hy4 flash (NVFP4, DFly, Engram tables on disk, 2× GPU)

> ⚠️ **Experimental.** Third-party NVFP4 weights, a vLLM development build with community
> SM120 patches, and a disk reader for the Engram tables that this profile adds. Measured on
> one host on 2026-11-28 (`journal/2026-11-28-first-start-engram-on-disk.md`).

[`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash) in the NVFP4 build
[`atakans/Hy4-flash-NVFP4`](https://huggingface.co/atakans/Hy4-flash-NVFP4) on two RTX PRO 6000
Blackwell GPUs with vLLM: 262,144 tokens per request, 32 requests at once, DFly speculative
decoding, text and image input.

## What is different here: the Engram tables stay on disk

The model looks up rows of four n-gram embedding tables ("Engram", 165 million rows of 96
bytes each, 59 GiB together) at four of its layers. The usual ways to hold them:

| Tables | GPU memory per GPU for the model | Host memory | KV cache per GPU |
|---|---|---|---|
| on the GPUs (vLLM's default for this build) | 79.35 GiB | small | 5 to 6 GiB: 65K-token requests |
| in host memory (the SM120 recipe) | 51.67 GiB | about 77 GiB in the recipe's container | 34 GiB |
| **on disk (this profile)** | **51.67 GiB** | **12 GiB measured** | **34 GiB** |

With `HY4_ENGRAM_DISK=1` each step's rows are read from the checkpoint's `.safetensors`
files before the step runs. The kernel's page cache keeps the rows in use and can drop them.

- **The rows are exact:** 399,882 randomly chosen rows of all four tables, read through the
  engine's reader, are bit-identical to the checkpoint's.
- **Decoding costs nothing:** 46.1 to 46.4 tok/s for one request on disk against 47.2 with the
  tables on the GPUs (no speculative decoding), and the same DFly draft acceptance.
- **Long prompts cost something when their rows are cold:** on Python source text, a
  105K-token prompt ran at 4,015 prompt tok/s with the rows evicted from the page cache and
  5,584 with them cached. The benchmark's random-word prompts are the worst case (2,893 for
  one request). The SM120 recipe reports about 6,600 to 6,900 with the tables in host memory.
- **The weights must be on fast local storage** (NVMe here). On a network file system every
  cold row is a remote read.

## Measured (2 × RTX PRO 6000 Blackwell Server Edition, 2026-11-28, nothing else on the host)

| | One request | Plateau |
|---|---|---|
| Short prompts, long answers (`max-decode`) | 93 tokens per second | 916 to 969 at 32 to 48 requests |
| Mixed lengths (`mixed-load`) | 61 tokens per second | 353 to 401 at 32 to 48 requests |
| 8,192-token prompts (`max-prefill`) | 2,893 prefill tokens per second | 4,326 to 4,633 at 8 to 32 requests |

- No failed request in the benchmark (`benchmarks/2026-11-28-1227Z-llmperfbench.yaml`, 32
  levels in 90 minutes). `benchmarks/2026-11-28-0957Z-llmperfbench.yaml` is the same profile
  with 8 sequences instead of 32: it stopped growing at 8 requests.
- Ready 5.1 to 5.4 minutes after a deploy with the weights in the page cache.
- Answers: image input, 24 of 24 known answers, two parallel tool calls and an answer from
  their results, a hidden code found in a 224,129-token prompt (36 s).
- The model always reasons. `reasoning_effort` takes `no_think`, `high` and `max`.
- GPU memory per GPU: 51.67 GiB of model, a 34 GiB KV cache of 3,277,817 tokens (12.5
  requests of the full 262,144 tokens); peak 96,279 MiB, 489 MiB below the booking.
- Host memory: 12.1 to 12.9 GiB of the 24G limit.

## The model

| | |
|---|---|
| Publisher | Tencent, released 2026-11-17, Tencent Hy License 1.0 |
| Parameters | 220B total, about 12B active per token |
| Input / output | text and images / text |
| Context | 1M tokens in the publisher's card; 262,144 per request here |
| Weights | `atakans/Hy4-flash-NVFP4` at `5d9383f3`: a third party's NVFP4 build of the experts, 164.7 GB in 14 shards, FP8 Engram tables with a scale per row. No Python file is downloaded. |

Tencent publishes BF16 and FP8. The NVFP4 build's quality against them is its author's claim
and is not measured here.

## Image

`onpremai/vllm-hy4-flash:77766db-sm120-diskengram1`, built from `runtime/Dockerfile`:

1. `vllm/vllm-openai` at vLLM main `77766db` (2026-11-17), pinned by digest.
2. Seven overlay files of
   [`high-tech-system/Hy4-flash-NVFP4-vLLM-SM120`](https://huggingface.co/high-tech-system/Hy4-flash-NVFP4-vLLM-SM120)
   at `b0067d3f` (Apache-2.0 and MIT): ModelOpt NVFP4 experts and FlashInfer's SM120 sparse
   attention backend. Fetched by revision and checked against their SHA-256. 150 changed lines
   and two small new files; reviewed against the image's originals.
3. `runtime/engram.py`: the image's `hy_v4_flash/nvidia/engram.py` with the disk patch of
   [`ZackSample/hy4-flash-dgx-sparks`](https://huggingface.co/ZackSample/hy4-flash-dgx-sparks)
   (Apache-2.0), extended here from BF16 tables to FP8 tables with a scale per row.

**No kernel cache is baked in.** Triton, FlashInfer and the CUDA compute cache compile
host-side code for the CPU they run on, and a cache warmed on one CPU stops with SIGILL on a
CPU without those instructions. Every node compiles its own at the first start; the build
fails if such a cache is in the image.

The image is built on the measuring host and is **not pushed to a registry yet**: another
node cannot pull it.

## Before this profile is more than a draft

1. Push the image and check that the registry's digest is the one in `compose.yaml`.
2. Start it on a node with another CPU (the first start compiles the kernels there).
3. Compare answers with Tencent's FP8 weights; check the answers through the API gateway.
4. The recipe reports about 124 tok/s for one request with DFly; here it is 78 to 93, with
   the tables on disk and on the GPUs alike. The cause is not known.
5. A faster reader for cold rows (asynchronous direct I/O instead of one read per row).

## Checker

No errors. Warning `engine.kv-cache-size-fixed`: the KV cache's size is set in bytes, as the
recipe does.

## License

Model: Tencent Hy License 1.0 (`licenses/source-0-LICENSE.txt`), derived from Apache 2.0 with
additional commercial terms for services with more than 100 million monthly active users.
Runtime patches: Apache-2.0 and MIT (`licenses/runtime-MIT-atakann.txt`).
