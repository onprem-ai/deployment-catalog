# Publishing roadmap: Apertus v1.5 70B NVFP4

## Overview

We have a working NVFP4 quantized checkpoint (48 GiB) of
`swiss-ai/Apertus-v1.5-70B` running on our RTX 6000 Blackwell node via a
patched vLLM image (`vllm-apertus-infomaniak`). Two things to publish:

1. **The checkpoint** on HuggingFace (derived work, Apache-2.0)
2. **The Docker image** in a public GitHub repo (build recipe + patches)

---

## Phase 1: Quant quality validation (MUST DO FIRST)

Before publishing anything, we need confidence that the NVFP4 quant hasn't
degraded quality vs the official BF16. Since we deleted the 135 GiB BF16
download to free disk space, we need a baseline to compare against.

### Plan

**Step 1:** Run the same `model_free_ptq` recipe but with FP8_DYNAMIC on
ALL linear layers (no two-pass, no NVFP4). This produces an FP8 checkpoint
of roughly equal size to the official FP8 release and serves as a high-quality
baseline. FP8_DYNAMIC is the standard vLLM quant format with minimal quality
loss — closer to BF16 than NVFP4 is.

```
input:  /persistent/models/swiss-ai/Apertus-v1.5-70B/     (re-download, 135 GiB)
output: /persistent/models/swiss-ai/Apertus-v1.5-70B-FP8/  (~60 GiB)
scheme: FP8_DYNAMIC on all Linear (single pass)
ignore: lm_head, model.language_model.embed_tokens,
        re:.*vision_tokenizer.*, re:.*audio_tokenizer.*
```

This is the same recipe as the official `swiss-ai/Apertus-v1.5-70B-FP8-Dynamic`
but using the exact same llm-compressor 0.12.0 + PyTorch 2.11 environment
as our NVFP4 quant to eliminate toolchain differences.

**Step 2:** Deploy both checkpoints (FP8 and NVFP4) side by side and run
the full capability test suite on each. Compare results.

Metrics to compare:
- Vision tests: 4/4 pass rate, output quality (subjective)
- Tool tests: 4/5 pass rate (parallel_tool_calls is model behavior)
- Chat tests: 3/3 pass rate
- Response coherence: same prompt, side-by-side output comparison
- Stop token behavior: both should show stop_reason 72/68

**Step 3:** If FP8 and NVFP4 results are comparable, proceed to Phase 2.
If NVFP4 is notably worse, investigate whether the two-pass recipe needs
adjustment (e.g., different scheme assignment, different ignore list).

---

## Phase 2: Publish the checkpoint on HuggingFace

### Repo

- Name: `swiss-ai/Apertus-v1.5-70B-NVFP4` (derivative of the official repo)
- License: Apache-2.0 (same as original)
- Access: Public (not gated)

### Contents

| File | Purpose |
|------|---------|
| `*.safetensors` (5 shards) | Quantized weights (48 GiB) |
| `config.json` | With fixed `architectures`, `quantization_config` (three-tier) |
| `generation_config.json` | With `eos_token_id: [2, 68, 72]` |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer (from original) |
| `chat_template.jinja` | Patched (system/assistant list content + tool_calls fix) |
| `special_tokens_map.json` | From original |
| `README.md` | Quant description, benchmark, usage |
| `model_free_ptq_recipe.json` | Reproducibility: the exact llm-compressor config used |

### Model card (README.md) sections

- Title: "Apertus v1.5 70B NVFP4 (three-tier mixed-precision)"
- Description: Two-pass data-free quantization of swiss-ai/Apertus-v1.5-70B
- Quantization method: llm-compressor 0.12.0, model_free_ptq
- Scheme: NVFP4A16 on MLP up/down_proj + FP8_DYNAMIC on attention q/k/v/o_proj
  + BF16 on embeddings, lm_head, norms
- Compression: 48 GiB vs 135 GiB BF16 (2.8x)
- Requirements: vLLM >= 0.23 (compressed-tensors mixed-precision), 96 GB GPU
- Performance: ~30 TPS decode, ~2200 TPS prefill (measured on RTX PRO 6000 BW)
- Known limitations: SGLang not supported, vLLM only

---

## Phase 3: Publish the Docker image

### GitHub repo: `apertus-vllm-image` (public)

#### Contents

```
README.md
Dockerfile
build.sh
CHANGELOG.md
patches/
  0001-mm-input-required-args.diff
  0002-reasoning-init-registration.diff
  0002a-apertus-reasoning-parser.diff
  0003-bos-fix.diff
  0004-weight-key-transforms.sh
  apply-0004.sh
  patch-wavetokenizer.py
  patch-chat-template.py
  apertus_reasoning_parser.py
.github/workflows/
  build-and-publish.yml
```

#### README sections

1. **What this is** — GPU-native Apertus multimodal on vLLM (Infomaniak/vllm
   branch apertus-1-5 + upstream main patches)
2. **Upstream lineage** — links to vllm-project/vllm, Infomaniak fork, PR #50496
3. **Patch breakdown** — what each patch does, why it's needed, upstream PR refs
4. **Usage** — `docker run` with model mount, required flags
5. **Supported models** — Apertus v1.5 70B/8B, ApertusForConditionalGeneration
6. **Quantization support** — BF16, FP8, NVFP4 (any compressed-tensors format)

#### CI

- GitHub Actions builds and pushes to DockerHub on tag/release
- Tag format: `apertus-v1.5-<commit-short>` 

---

## Phase 4: Announce

- Post in HuggingFace model discussion on the original repo
- Link to the quant repo and the Docker image repo
- Provide the exact `docker run` command
- Note that testing / quality comparisons with FP8 are available

---

## Timeline

| Phase | Depends on | Est. effort |
|-------|-----------|-------------|
| 1. Download BF16 + quant FP8 | Disk space (~200 GiB free needed) | ~15 min download + 2 min quant |
| 2. Compare test results | Phase 1 done | ~30 min |
| 3. Publish HF checkpoint | Phase 2 pass | ~30 min (upload 48 GiB) |
| 4. Create public GitHub repo | Phase 3 done | ~1 hr (docs, CI) |
| 5. Announce | Phase 4 done | ~15 min |

Total: ~2.5 hours if Phase 2 looks good.

## Pre-flight checklist

- [ ] 200+ GiB free on `/persistent` for the BF16 re-download
- [ ] HF token with write access to the target org
- [ ] GitHub token for the new public repo
- [ ] DockerHub account for automated builds
- [ ] All patches cleaned up and documented with upstream references
