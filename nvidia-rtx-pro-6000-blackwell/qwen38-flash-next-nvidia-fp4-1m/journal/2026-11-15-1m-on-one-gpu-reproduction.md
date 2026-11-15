# 2026-11-15 - NVIDIA's Qwen3.8 Flash Next FP4 at 1M Context on One RTX PRO 6000

**Goal:** Replace the community NVFP4 checkpoint of the `qwen38-flash-next-nvfp4` card with
NVIDIA's official `nvidia/Qwen3.8-Flash-Next-FP4-Local`, and serve it with 1,048,576 tokens of
context on one GPU, in vLLM, without a prepared runtime package. Every step is listed so it
can be repeated.

## Why

- Our card (`qwen38-flash-next-nvfp4`, SGLang, two GPUs, community checkpoint
  `RadixArk/Qwen3.8-Flash-Next-NVFP4`) measured 65 GiB of weights per GPU, a 16-bit KV cache
  and 592,576 KV cache tokens per replica, with 262,144 tokens of context.
- NVIDIA's model card reports about 850K tokens of FP8 KV cache on one RTX PRO 6000 with this
  checkpoint (weights ~96 GiB on disk, with the PLE n-gram table offloaded to host RAM).
- A community package (`JohannesBertens/Qwen3.8-Flash-Next-FLARE-1M`, a prepared Flare runtime
  package, not weights) reports 1M context through YaRN-style rope interpolation, factor 4.
  We use the same interpolation in vLLM instead of the package: plain weights from NVIDIA, our
  usual engine, no third-party compiled kernels.

## Inputs

| Input | Value |
|---|---|
| Machine | `ain-4euk3`: two RTX PRO 6000 Blackwell (96 GiB, SM120), 62 GiB RAM; one GPU used |
| Weights | `nvidia/Qwen3.8-Flash-Next-FP4-Local` at revision `b9ce18a6887e18c089fdea8509a33e63429a3300` (2026-10-15), 33 files, 95.2 GiB |
| Image | `ghcr.io/local-inference-lab/vllm:release-10152026`, pinned `sha256:745feb916ee816f5409fd58386114be4f22fbede7f9c0e0a7b8c4fe8dac0fb29` (vLLM `0.1.dev21830+g8244d1c44`), the image NVIDIA's model card names for RTX PRO 6000 |
| Licence | NVIDIA Open Model Agreement; the base model is under the Qwen Community License 1.0 (model card) |

NVIDIA's card says upstream vLLM cannot read this checkpoint's NVFP4 scale format yet, so the
named community image is required.

## Steps

1. **Download, verified.** List the revision's files and their LFS SHA-256 from
   `https://huggingface.co/api/models/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/<revision>?recursive=true`,
   download each with `curl -C -` from `.../resolve/<revision>/<path>`, 8 at a time, and accept
   a file only when its size and SHA-256 match. Result: 33 of 33 files verified.
2. **Pull the image** by digest (44.3 GB).
3. **Environment and command.** Copied unchanged from the model card's "NVIDIA RTX PRO 6000
   Blackwell" section: 55 environment variables (attached:
   `2026-11-15-nvidia-env.sh.txt`, SHA-256 `cfd2f0f484e7910d...`) and the `vllm serve` command
   (FP8 KV cache, MTP with 3 draft tokens, b12x kernels, `--gpu-memory-utilization 0.975`,
   `--max-num-seqs 16`, `--load-format instanttensor`, `VLLM_PLE_CPU_OFFLOAD=1`), with the model
   path `/models/model` and the name `qwen38-flash-next`.
4. **Run at native context** (`--max-model-len 262144`):
   `docker run --gpus '"device=<GPU>"' --ipc host --shm-size 32g --ulimit memlock=-1
   --ulimit stack=67108864:67108864 -v <weights>:/models/model:ro -v qfn-cache:/root/.cache
   <image> bash -c 'source env.sh; source serve.sh'`. The first start took 645 s (kernel
   compilation, then cached in `/root/.cache`).
5. **Run at 1M.** The same, with two changes (attached: `2026-11-15-serve-1m.sh.txt`):
   `--max-model-len 1048576` and
   `--hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true,
   "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000,
   "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144},
   "max_position_embeddings": 1048576}}'`. The checkpoint's own `rope_parameters` is
   `rope_type: default` with the same theta and M-RoPE sections; only the type, factor and
   anchor change. Second start: 375 s.
6. **Confirm YaRN took effect.** transformers logs "Unrecognized keys in `rope_parameters` for
   'rope_type'='default'", which reads as if the override were ignored. It is not: building the
   rotary embedding with vLLM's own `get_rope` in the image gives `MRotaryEmbedding` with
   `scaling_factor 4.0`, `mscale 1.1386` and a 4,194,304-position table for the YaRN
   parameters, and a different table at position 1,000 than the native parameters.

## Measured (one GPU)

| | 262K (native) | 1M (YaRN factor 4) |
|---|---|---|
| `Model loading took` | 71.33 GiB | 71.71 GiB |
| `Available KV cache memory` | 16.1 GiB | 15.77 GiB |
| `GPU KV cache size` | 874,144 tokens | 1,090,080 tokens |
| CUDA graphs | 0.38 + 0.25 GiB | 0.39 + 0.26 GiB |
| GPU memory used after start (`nvidia-smi`) | 94,039 MiB | 95,045 MiB after the 1M probes |

## Answers

Thinking on: "17 * 23" gave 391 (124 reasoning characters). Thinking off: 391, and "Canberra"
for the capital of Australia. Two parallel tool calls for Paris and Tokyo:
`finish_reason: tool_calls`, both cities.

## Hidden number at long context

A secret six-digit number in varied filler sentences, at 10, 50 and 90 percent depth,
thinking off, temperature 0 (script attached: `2026-11-15-needle.py.txt`).

| Context | Prompt tokens | Result | Time to answer |
|---|---|---|---|
| native, 32K | 31,761 | 3 of 3 | 2.2 s |
| native, 128K | 127,900 | 3 of 3 | 8.8 s |
| native, 255K | 254,600 | 3 of 3 | 18.8 s |
| YaRN, 128K | 127,852 | 3 of 3 | 8.9 s |
| YaRN, 400K | 399,578 | 3 of 3 | 32 s |
| YaRN, 700K | 699,644 | 3 of 3 | 67 s |
| YaRN, 1.04M | 1,038,713 | 3 of 3 | 113 s |

## Open

- Retrieval of one fact is not reasoning quality at 1M; no task-level comparison yet.
- The full replica benchmark through the control plane (next entry).
- TP2 on both GPUs.
