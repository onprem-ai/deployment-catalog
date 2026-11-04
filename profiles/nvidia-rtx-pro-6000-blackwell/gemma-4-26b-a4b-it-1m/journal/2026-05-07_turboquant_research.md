# 2026-05-07 — TurboQuant Research for 1M Context

**Goal:** Research requirements for running Gemma 4 26B MoE with 1M context on RTX PRO 6000 Blackwell using TurboQuant KV cache compression.

## Container Image

No pre-built image exists with TurboQuant + sm_120a support.

The spiritbuun fork only publishes source — no container registry with sm_120a builds. The upstream `ghcr.io/ggml-org/llama.cpp:server-cuda` uses generic CUDA arch targets and lacks TurboQuant support entirely. The charleschen wiki notes the generic Docker image is 66% slower on prefill vs the native sm_120a build.

**Build from source:**

```bash
git clone https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git
cd llama-cpp-turboquant-cuda

# Build natively (recommended — what the wiki uses)
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Or build a Docker image
docker build -t llama-turboquant:server-cuda-sm120a \
  --target server \
  --build-arg CUDA_DOCKER_ARCH=120a-real \
  -f .devops/cuda.Dockerfile .
```

`120a-real` is correct for RTX PRO 6000 — it enables native MXFP4 tensor core instructions (mma.sync.kind::mxf8f6f4), which compute directly in FP4 without dequant-to-BF16.

## Model Files

The MXFP4 GGUF is on HuggingFace. Two sources:

**Unsloth (primary — this is the provider the charleschen wiki references):**
- https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF
- File: `gemma-4-26B-A4B-it-MXFP4_MOE.gguf` (~16.7 GB)
- Standard variant: MXFP4 for MoE expert tensors, Q8 for the rest.

**noctrex (higher quality alternatives):**
- https://huggingface.co/noctrex/gemma-4-26B-A4B-it-MXFP4_MOE-GGUF

| File | Non-MoE tensors | Size |
|------|-----------------|------|
| gemma-4-26B-A4B-it-MXFP4_MOE.gguf | Q8 | 15.4 GB |
| gemma-4-26B-A4B-it-MXFP4_MOE_BF16.gguf | BF16 (highest quality) | 17 GB |
| gemma-4-26B-A4B-it-MXFP4_MOE_F16.gguf | F16 | 17 GB |

MXFP4 weights require Blackwell with FP4 tensor cores — the RTX PRO 6000 (sm_120a) supports this natively. Consumer RTX 5080/5090 (sm_120) do not support MXFP4 — those cards should use IQ4_NL or Q4_K_M quants instead.

## Community Results — 1M+ Context Achieved

### Gemma 4 26B MoE on 3x RTX PRO 6000 Blackwell (96 GB each)

From charleschen.ai wiki:

| Context | KV Type | KV Size | Total VRAM | Decode |
|---------|---------|---------|------------|--------|
| 262K | turbo3 | 1,514 MiB | 29.3 GB | 125 t/s |
| 524K | turbo3 | ~3,028 MiB | 43.8 GB | — |
| 1M | turbo3 | ~6,056 MiB | 72.7 GB | — |
| 2M | turbo2 | 6.4 GB | 98 GB (3 GPUs) | 137 t/s |
| 10M | turbo2 | 32 GB | 222 GB (3 GPUs) | 137 t/s |

They hit 10M tokens with a one-line llama.cpp fix (capping n_kv reservation to 131K during graph sizing).

### Gemma 4 31B Dense on RTX 5090

- Model: gemma-4-31B-it-UD-Q4_K_XL (Unsloth, 17.46 GiB)
- KV: turbo3 (3-bit PolarQuant + Hadamard rotation)
- VRAM at 262K: 27.7 GB / 32 GB
- Decode: 61.5 t/s (constant regardless of context length)
- Prefill at 262K: 900 t/s (down from 3,363 at 4K — O(n^2) attention)

### Gemma 4 26B on RTX 3090 (24 GB) — turbo4 fork

A dedicated fork (llama-cpp-turboquant-gemma4) got turbo4 at zero speed penalty vs f16:

| Config | Speed (32K) | KV VRAM (256K) | Fits 24GB? |
|--------|-------------|----------------|------------|
| f16 KV | 120 t/s | 9.4 GB | No |
| turbo4 KV | 120 t/s | 2.5 GB | Yes |

## Quality Analysis

| Measurement | Source | Result |
|-------------|--------|--------|
| Perplexity (WikiText-2) | Llama 70B, tq3 K+V | +7.6% (4.40 vs 4.09 baseline) |
| Perplexity (turbo3 vs q8_0) | Qwen 35B, llama.cpp | +1.1% (5.471 vs 5.414) |
| Prefill quality (turbo3) | Relative to q8_0 | 98.7–99.5% flat through 32K |
| Subjective output | RTX 3090 turbo4 Gemma 4 | "identical-quality" — verified on math, reasoning, chat |
| Speculative decoding | turbo3 KV | Incompatible — 0% draft acceptance rate |

**Key findings:**
- K-only compression is nearly transparent — adding V compression adds most of the degradation
- Asymmetric is recommended: q8_0 for Keys, turbo3 for Values preserves retrieval quality (Keys pass through softmax where noise gets amplified)
- Quality holds through ~256K but nobody has published rigorous benchmarks at 1M for Gemma 4 specifically (needle-in-haystack, RULER, etc.)
- Compounding with weight quantization: Q4 weights + turbo2 KV = compounding errors on complex reasoning tasks

## Caveats for 1M+ Context

1. **Gemma 4 was only trained on 256K** — beyond that, it's pure RoPE extrapolation. The model has never seen positional encodings for positions >262,144 during training
2. **No YaRN/NTK scaling applied** in most community runs — they're just raw-extending, relying on Gemma 4's RoPE theta being high enough
3. **Attention quality degrades** even if KV fits — the model wasn't trained to attend across 1M positions
4. **Dequant overhead grows** with cache fill: turbo2 is 21.6% slower at 6.4K cached tokens vs empty cache; at 1M the decode speed will be significantly impacted
5. **Attention compute bottleneck** — at 2M+, flash attention workspace memory exceeds single-GPU capacity regardless of KV compression

## Software Status

| Runtime | TurboQuant Support |
|---------|-------------------|
| llama.cpp upstream | Not merged (active discussion: #20969) |
| TheTom/llama-cpp-turboquant fork | turbo2/3/4, Metal + CUDA |
| Python transformers + turboquant pkg | Full support (drop-in cache class) |
| LM Studio / Ollama / vLLM | Not supported yet |

## Single GPU Feasibility (96 GB RTX PRO 6000)

Based on the charleschen data:
- 1M context with turbo3 needs ~72.7 GB total VRAM
- Single RTX PRO 6000 has 96 GB
- **Should fit on single GPU with ~23 GB headroom**

## Sources

### TurboQuant — Papers & Official
- https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- https://arxiv.org/abs/2504.19874 (TurboQuant paper)
- https://arxiv.org/abs/2511.01815 (KVTC paper)

### charleschen.ai Wiki
- https://wiki.charleschen.ai/nvidia/techniques/turboquant-kv-cache-1m-context-mcqueen
- https://wiki.charleschen.ai/nvidia/techniques/kv-config-matrix-gemma4-blackwell
- https://wiki.charleschen.ai/nvidia/techniques/gpt-oss-120b-mxfp4-native-sm120a-mcqueen

### llama.cpp — Upstream & Forks
- https://github.com/ggml-org/llama.cpp/discussions/20969 (TurboQuant integration discussion)
- https://github.com/TheTom/llama-cpp-turboquant (TheTom's fork)
- https://github.com/spiritbuun/llama-cpp-turboquant-cuda (CUDA fork with sm_120a)
- https://github.com/test1111111111111112/llama-cpp-turboquant-gemma4 (RTX 3090 Gemma 4 turbo4 fork)

### Gemma 4 — GGUF Downloads
- https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF
- https://huggingface.co/noctrex/gemma-4-26B-A4B-it-MXFP4_MOE-GGUF

## Outcome

Research complete. Key findings:
1. **Need to build custom Docker image** from spiritbuun/llama-cpp-turboquant-cuda with `CUDA_DOCKER_ARCH=120a-real`
2. **Model file:** unsloth/gemma-4-26B-A4B-it-GGUF → gemma-4-26B-A4B-it-MXFP4_MOE.gguf (~16.7 GB)
3. **1M context should fit** in single RTX PRO 6000 (96 GB) with turbo3 KV (~72.7 GB total)
4. **Quality caveat:** Model only trained to 256K, 1M is extrapolation with ~7% perplexity increase from KV compression

Next steps:
- Build the llama-cpp-turboquant Docker image with sm_120a
- Download the MXFP4 GGUF model
- Deploy and test with increasing context lengths
- Measure actual VRAM usage and quality at 1M
