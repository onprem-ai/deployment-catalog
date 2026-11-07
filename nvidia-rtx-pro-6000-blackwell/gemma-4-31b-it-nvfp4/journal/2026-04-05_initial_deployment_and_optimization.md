# Session: Initial Deployment and Optimization

**Date:** 2026-04-05
**Model:** Gemma 4 31B IT NVFP4 (`nvidia/Gemma-4-31B-IT-NVFP4`)
**Hardware:** 1x NVIDIA RTX 6000 Blackwell (96GB VRAM)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML frameworks.
> Information 2+ weeks old may be outdated. Verify against current releases.

**Goal:** Deploy Gemma 4 31B IT NVFP4 on single RTX 6000 Blackwell (96GB).

**Outcome:** SUCCESS — Model running with 31.04 GiB VRAM, 128K context, prefix caching enabled.

---

## Final Working Configuration

```bash
vllm serve /models/gemma-4-31b-it-nvfp4 \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --chat-template /models/gemma-4-31b-it-nvfp4/chat_template.jinja
```

**Key settings**:
- `--quantization modelopt` - NVIDIA's modelopt quantization format (NOT compressed-tensors)
- `--tensor-parallel-size 1` - Single 96GB GPU (31B NVFP4 fits in ~31GB VRAM)
- `--chat-template` - Required! Gemma 4 tokenizer doesn't embed chat template
- Image: `vllm/vllm-openai:gemma4-cu130` - Official Gemma 4 image (released 2026-04-02)

---

## What We Tried (Chronological)

### Attempt 1: onpremai/vllm-blackwell:latest
**Status**: FAILED - Image not found

Image doesn't exist on Docker Hub. This is a local build image per `catalog/rtx6k/_docker/vllm-blackwell/README.md`.

**Error**: `failed to resolve reference "docker.io/onprem-ai/vllm-blackwell:latest": not found`

---

### Attempt 2: vllm/vllm-openai:v0.17.1-cu130
**Status**: FAILED - Architecture not supported

Image pulled successfully but vLLM v0.17.1 doesn't recognize Gemma 4 architecture.

**Error**:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, The checkpoint you are trying to load has model type `gemma4` 
  but Transformers does not recognize this architecture.
```

**Lesson**: HuggingFace model card says "vLLM version: 0.17.2rc1+" - need newer version.

---

### Attempt 3: Config files in wrong directory
**Status**: FIXED

Download system stored config files in hash-based directory, model weights in download_id directory:
- Config files: `/persistent/models/0b912f919bc6199373e0b1abac07a066/`
- Model weights: `/persistent/models/nvidia--Gemma-4-31B-IT-NVFP4--model/`

vLLM couldn't find `config.json` because it was in different folder.

**Fix**: Copied config files to model weights directory:
```bash
sudo cp /persistent/models/0b912f919bc6199373e0b1abac07a066/*.json \
  /persistent/models/nvidia--Gemma-4-31B-IT-NVFP4--model/
```

**Note**: This is a bug in the download system - should be fixed upstream.

---

### Attempt 4: vllm/vllm-openai:v0.19.0
**Status**: FAILED - transformers too old

Same error as v0.17.1 - the bundled transformers library doesn't have Gemma 4 support yet.

**Error**: `The checkpoint you are trying to load has model type 'gemma4' but Transformers does not recognize this architecture`

**Lesson**: vLLM Docker images bundle a specific transformers version. Even v0.19.0 doesn't have Gemma 4 support yet. Need either:
1. Custom image with `pip install git+https://github.com/huggingface/transformers.git`
2. NVIDIA NGC vLLM image (nvcr.io/nvidia/vllm) which may have newer transformers
3. Wait for official vLLM release with Gemma 4 support

---

### Attempt 5: nvcr.io/nvidia/vllm:25.12-py3
**Status**: FAILED - same transformers issue

NGC December 2025 image also has outdated transformers that doesn't recognize `gemma4` architecture.

**Conclusion**: No publicly available vLLM Docker image supports Gemma 4 yet. Must build custom image.

---

### Attempt 6: onpremai/vllm-blackwell:vllm0.17.1-cuda13.0
**Status**: FAILED - same transformers issue

Custom Blackwell build is based on vLLM 0.17.1, which predates Gemma 4 architecture support.

**All tested images**:
- `vllm/vllm-openai:v0.17.1-cu130` - no gemma4
- `vllm/vllm-openai:v0.19.0` - no gemma4
- `nvcr.io/nvidia/vllm:25.12-py3` - no gemma4
- `onpremai/vllm-blackwell:vllm0.17.1-cuda13.0` - no gemma4

**Root cause**: Gemma 4 model was released after all these images were built. The `gemma4` architecture needs to be in the transformers library bundled with vLLM.

**Solution**: Build new `onpremai/vllm-blackwell` image with latest transformers:
```dockerfile
# Add to Dockerfile before pip install vllm:
RUN pip install git+https://github.com/huggingface/transformers.git
```

---

### Attempt 7: vllm/vllm-openai:gemma4-cu130
**Status**: PARTIAL SUCCESS

Official vLLM Gemma 4 image with CUDA 13.0 support, released 2026-04-02.
Found via Docker Hub API - confirmed tags exist:
- `gemma4-cu130` - 2026-04-02T23:25:21Z

This image includes transformers 5.5.0 with Gemma 4 architecture support.

**First attempt**: Failed with "World size (8) is larger than the number of available GPUs (1)".
- HuggingFace README says `--tensor-parallel-size 8` but we only have 1x 96GB GPU
- Changed to `--tensor-parallel-size 1` since 31B NVFP4 (~31GB) fits in 96GB

**Second attempt**: Model loaded (31.04 GiB VRAM) but chat completions failed:
```
As of transformers v4.44, default chat template is no longer allowed, 
so you must provide a chat template if the tokenizer does not define one.
```

---

### Attempt 8: Chat template fix
**Status**: SUCCESS

Gemma 4 tokenizer doesn't embed the chat_template in tokenizer_config.json (unlike Gemma 2/3).
The official template is in a separate `chat_template.jinja` file in the HuggingFace repo.

**Solution**: Created simplified chat template using official Gemma 4 format:
- Uses `<|turn>role\n...<turn|>` tokens (NOT `<start_of_turn>`/`<end_of_turn>`)
- Maps OpenAI `assistant` role to Gemma `model` role
- Disables thinking mode by default with empty `<|channel>thought\n<channel|>` block

**Key insight**: The official Gemma 4 chat template format changed from Gemma 2/3:
```
Old format: <start_of_turn>user\nHello<end_of_turn>
New format: <|turn>user\nHello<turn|>
```

**Final working command**:
```bash
vllm serve /model \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --chat-template /model/chat_template.jinja \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**Test results**:
- Chat completions: Working
- System messages: Working
- VRAM usage: 31.04 GiB
- Context: 128K configured

---

## VRAM Measurements

### Methodology

vLLM pre-allocates KV cache to fill `--gpu-memory-utilization` setting. To measure actual minimum VRAM requirements:

1. Deploy with a specific `--max-model-len` and low `--gpu-memory-utilization`
2. If pod crashes/restarts, utilization too low
3. If pod starts successfully, check actual VRAM with `nvidia-smi`
4. Increase utilization until stable, record minimum working value

### Measurement Commands

```bash
# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Check pod status (RESTARTS > 0 means OOM)
kubectl get pods -n <namespace> -l serving.kserve.io/inferenceservice=<inference-service>
```

### Results (RTX 6000 Blackwell 96GB, default FP8 KV cache)

| Context Length | GPU Util Setting | VRAM Used | Status |
|----------------|------------------|-----------|--------|
| 32K | 50% | 51 GiB | Working |
| 64K | 70% | 70 GiB | Working |
| 128K | 40% | - | **FAILED** (OOM) |
| 128K | 80% | 82 GiB | Working |
| 128K | 95% | 95 GiB | Working (chosen config) |

### Memory Breakdown

- **Model weights**: 31 GiB (fixed, NVFP4 quantized)
- **KV cache (FP8)**: Scales with context length
  - 32K: ~20 GiB
  - 64K: ~39 GiB
  - 128K: ~51 GiB

### KV Cache Quantization Options

vLLM supports different KV cache data types via `--kv-cache-dtype`:

| Option | Description | Memory | Quality |
|--------|-------------|--------|---------|
| `fp8_e4m3` | 8-bit float (default) | 1x | Good |
| `auto` | Model dtype (FP16/BF16) | 2x | Best |

**Note**: vLLM auto-selects FP8 KV cache for this model. All measurements above use FP8.

**FP16 KV cache estimates** (not measured):
- 32K context: ~71 GiB (31 + 40 GiB KV)
- 16K context: ~51 GiB (31 + 20 GiB KV)
- 64K+ context: Won't fit on 96GB GPU with FP16 KV

---

## Performance Optimization Attempts

### Baseline
- Decode speed: ~40 tps
- Prefill speed: ~6600 tps

### N-gram Speculative Decoding
**Config**: `--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'`

**Result**: 0% acceptance rate, slightly slower than baseline (~37 tps)

**Conclusion**: N-gram speculation doesn't help for open-ended generation. It's designed for "input-grounded" tasks (summarization, translation) where output contains patterns from input.

### EAGLE/P-EAGLE Speculative Decoding
**Status**: Not tested - requires:
1. EAGLE3 draft model trained for Gemma 4 (not yet available per [Issue #38893](https://github.com/vllm-project/vllm/issues/38893))
2. HuggingFace authentication for gated draft models

### Prefix Caching
**Config**: `--enable-prefix-caching`

**Results** (150 tokens):
| Request | Time | TPS | Notes |
|---------|------|-----|-------|
| Cold | 4.9s | 30 | First request |
| Cached | 3.8s | 39 | Same prefix, **22% faster** |

**Conclusion**: Prefix caching helps for repeated system prompts or conversation continuations. Recommended for production.

### Recommended Production Config
```bash
--enable-prefix-caching  # 22% faster for repeated prefixes
# Note: n-gram and EAGLE not beneficial for Gemma 4 yet
```

---

## Known Issues

1. **Download system bug**: Config files stored in separate hash directory from model weights. Manual copy required.

2. **Chat template not embedded**: Unlike Gemma 2/3, Gemma 4 tokenizer doesn't include chat_template in tokenizer_config.json. Must provide via `--chat-template` argument.

3. **vLLM version requirement**: Gemma 4 architecture requires `vllm/vllm-openai:gemma4-cu130` or newer image with transformers 5.5.0+.

## Constraints

- **vLLM image**: `vllm/vllm-openai:gemma4-cu130` (or newer with Gemma 4 support)
- **Quantization**: `--quantization modelopt` (NOT compressed-tensors)
- **Chat template**: Must provide via `--chat-template` argument
- **GPUs**: 1x 96GB for single-GPU, or 8x for tensor parallelism
- **Context**: 256K native, tested with 128K

---

## Key Learnings

### 1. Gemma 4 Chat Template Format Changed
Gemma 4 uses completely different control tokens from Gemma 2/3:

| Version | Turn Start | Turn End | Assistant Role |
|---------|------------|----------|----------------|
| Gemma 2/3 | `<start_of_turn>` | `<end_of_turn>` | `model` |
| Gemma 4 | `<|turn>` | `<turn|>` | `model` |

### 2. Chat Template Not Embedded
Unlike previous Gemma versions, Gemma 4's `tokenizer_config.json` does NOT contain a `chat_template` field. The template is in a separate `chat_template.jinja` file. This causes vLLM to fail with:
```
As of transformers v4.44, default chat template is no longer allowed
```

**Fix**: Always use `--chat-template /path/to/chat_template.jinja`

### 3. Thinking Mode Control
Gemma 4 has built-in thinking/reasoning mode controlled via:
- Enable: Add `<|think|>` to system prompt
- Disable: Add empty `<|channel>thought\n<channel|>` before model response

Our chat template disables thinking by default for cleaner API responses.

### 4. Single GPU Works Fine
Despite HuggingFace README suggesting 8-GPU tensor parallelism, the 31B NVFP4 model fits comfortably in a single 96GB GPU:
- Model weights: ~31 GiB
- 128K context: 82 GiB total (measured)

### 5. Image Selection Critical
Standard vLLM images (even v0.19.0) don't have Gemma 4 support. Must use the special `gemma4-cu130` tag which bundles transformers 5.5.0+.

---

## References

- [Gemma 4 vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- [Gemma 4 Chat Template Discussion](https://huggingface.co/google/gemma-4-E2B-it/discussions/8)
- [Google Gemma Prompt Structure](https://ai.google.dev/gemma/docs/core/prompt-structure)
- [NVIDIA Gemma 4 31B IT NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)
