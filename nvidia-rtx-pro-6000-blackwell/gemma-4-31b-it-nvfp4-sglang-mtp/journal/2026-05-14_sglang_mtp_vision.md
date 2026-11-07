# Gemma 4 31B NVFP4 on SGLang with MTP + Vision

**Date**: 2026-05-14
**Goal**: Run Gemma 4 31B with MTP speculative decoding AND vision on SGLang, which vLLM cannot do (requires `--language-model-only` for MTP, disabling vision).

## Steps

### 1. Pre-flight checks
- [ ] Verify drafter model is already downloaded
- [ ] Check available RAM/VRAM for Docker build
- [ ] Determine correct SGLang base image (nightly vs release)

### 2. Build SGLang nightly image with NVFP4 JIT warmup
- [ ] Build `onpremai/sglang-blackwell:nightly` using `_docker/sglang-blackwell/build.sh`
- [ ] Verify FROZEN_KV_MTP module exists in image

### 3. Create catalog entry
- [ ] Create `gemma-4-31b-it-nvfp4-sglang-mtp/inference.yaml`
- [ ] Verify quantization flag and parser names

### 4. Deploy and test
- [ ] Deploy via API
- [ ] Run vision capability test
- [ ] Run tool calling test
- [ ] Measure VRAM and performance

---

## Log

### 12:00 — Pre-flight
- Drafter model present: `/persistent/models/google--gemma-4-31B-it-assistant--model`
- Main NVFP4 model present: `/persistent/models/nvidia--Gemma-4-31B-IT-NVFP4--model`
- No stable release > v0.5.11 on DockerHub, used `lmsysorg/sglang:gemma4-mtp` instead
  - Built from PR #24436 branch (confirmed working MTP for Gemma 4)
  - SGLang v0.5.12.dev93, CUDA 13.0
  - Has None-check guard in `frozen_kv_mtp_utils.py` (no bug #24912)
  - `modelopt_fp4` quantization available, `NEXTN` algorithm available
  - All Gemma 4 model modules present (gemma4_mm, gemma4_mtp, gemma4_causal)

### 12:08 — Image verified, catalog entry created
- Config ID: `rtx6000bw--gemma-4-31b-it-nvfp4-sglang-mtp`
- API confirms: downloaded=true, can_deploy=true
- Deployed via `POST /api/catalog/deploy` with memory_profile=1 (32k ctx), +20 GiB
- Pod creating, K8s pulling image on the GPU node

### 12:25 — First crash: drafter path validation
- **Error**: SGLang's `_is_already_quantized()` validates the drafter path as
  an HF repo ID, rejecting absolute paths like `/drafter`
- **Fix**: Populate HF cache structure at `/tmp/hf_cache/models--google--gemma-4-31B-it-assistant/`
  with symlinks to `/drafter/*`, pass HF model name instead of absolute path

### 12:25 — Second crash: drafter quantization inheritance
- **Error**: SGLang inherited `modelopt_fp4` from main model → tried to quantize BF16 drafter
  → entered "Standard quantization mode" → `ImportError: accelerate is required`
- **Root cause**: `ServerArgs` line 786-787: if `speculative_draft_model_quantization` is None,
  it defaults to main model's quantization
- **Fix**: Added `--speculative-draft-model-quantization unquant` (line 788-789: sets to None)

### 12:25 — Third crash: `${DRAFT_CACHE}` template variable collision
- **Error**: Backend template engine resolved `${DRAFT_CACHE}` as template variable
- **Fix**: Inlined the path instead of using shell variable

### 12:28 — Redeployed with all three fixes

### 12:29 — Server up and running!
- Main model: `Gemma4ForConditionalGeneration` (multimodal), 33.15 GiB, `modelopt_fp4`
- Drafter: `Gemma4AssistantForCausalLM`, 0.92 GiB (unquantized BF16)
- CUDA graph capture: main 24.2s + drafter 4s
- KV cache: SWA pool 35.63 GiB (swa: 83016 tokens, full: 103771 tokens)
- Available GPU mem after init: 21.37 GiB

### 12:30 — All capability tests passed
| Test | Result |
|------|--------|
| Text completion | "Four" (correct) |
| Vision (1px red PNG) | "Red" (correct) |
| Vision (64x64 gradient) | Correctly described blue-to-red gradient |
| Tool calling | `get_weather(location="Zurich")` (correct) |
| Reasoning | Step-by-step multiplication (correct) |

### 12:31 — Performance measurements
- **VRAM**: 76.8 GiB used / 96 GiB total (nvidia-smi)
- **Decode throughput (text)**: 85-108 tok/s (SGLang gen throughput)
  - MTP acceptance rate: 36-56% (content-dependent)
  - Average accept length: 3.0-3.8 tokens per step (out of 5 steps)
- **Decode throughput (vision)**: ~81 tok/s (shorter generation)
- **Effective tok/s** (pod-internal, end-to-end):
  - 500 tokens text: 103.1 tok/s
  - 1000 tokens text: 84.6 tok/s
  - 114 tokens vision: 81.5 tok/s
  - 291 tokens reasoning: 119.0 tok/s

### 13:00 — Benchmark results (4-test suite, pod-internal)

| Test | Input | Output | Key Metric |
|------|-------|--------|------------|
| Decode | 50 words | 500 tokens × 10 req | 95.7 tok/s generation |
| Prefill | 1500 words (~5400 tokens) | 100 tokens × 5 req | ~4,719 tok/s prompt |
| Balanced | 500 words | 500 tokens × 5 req | P50=8206ms, P95=8518ms |
| Long context | 3000 words (~10k tokens) | 100 tokens × 3 req | prefill ~4069 tok/s, TTFT ~2636ms |

### 13:10 — VRAM binary search (32k context, fp8 kv, NEXTN MTP)

Pass criteria: `max_total_num_tokens >= 32768` (full context must be usable).

| min_gi | max_total_num_tokens | Status |
|--------|---------------------|--------|
| 50 | 36,872 | PASS |
| 47 | 37,850 | PASS |
| 46 | 35,103 | **PASS — minimum** |
| 45 | 24,116 | FAIL (< 32768) |
| 42 | ~20,000 | FAIL |
| 40 | 18,623 | FAIL |
| 38 | 13,129 | FAIL |

**Result: min_gi=46** (max_total_num_tokens=35,103 >= 32,768)
- nvidia-smi: 51,373 MiB (50.2 GiB)
- kubectl top: 7,595 MiB (7.4 GiB RAM)

### 13:30 — VRAM binary search (128k context, fp8 kv, NEXTN MTP)

Pass criteria: `max_total_num_tokens >= 131072`.

| min_gi | max_total_num_tokens | Status |
|--------|---------------------|--------|
| 80 | 133,985 | PASS |
| 79 | 131,239 | **PASS — minimum** |
| 78 | 128,492 | FAIL (< 131072) |
| 77 | 125,745 | FAIL |
| 75 | 117,505 | FAIL |
| 60 | 76,304 | FAIL |

**Result: min_gi=79** (max_total_num_tokens=131,239 >= 131,072)
- nvidia-smi: 87,141 MiB (85.1 GiB)
- kubectl top: 6,769 MiB (6.6 GiB RAM)

### Summary

SGLang with NEXTN (FROZEN_KV_MTP) speculative decoding successfully runs
Gemma 4 31B NVFP4 with **both MTP AND vision** — something vLLM cannot do.
Throughput is ~90 tok/s (vs 140 tok/s on vLLM text-only MTP), but the
multimodal support is the key differentiator.

**Final catalog values:**
- 32k context: min_gi=46
- 128k context: min_gi=79
- tps_decode: 90
- RAM: requests=12Gi, limits=16Gi

**Three SGLang-specific issues found and resolved:**
1. Drafter path validation: HF cache symlink workaround
2. Drafter quantization inheritance: `--speculative-draft-model-quantization unquant`
3. Template variable collision: inline paths instead of shell variables
