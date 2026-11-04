> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML frameworks.
> Information 2+ weeks old may be outdated. Before relying on findings:
> 1. Check the session date
> 2. Search for updated information
> 3. Verify against current framework releases

# Session: Qwen3.6-35B-A3B NVFP4 — Initial Deployment and Optimization

**Date**: 2026-04-24

Multimodal VLM (vision + text) from Alibaba, NVFP4 quantized by RedHatAI.
This is a "preliminary version" per RedHatAI README — subject to change.

## Model Info
- **Model**: RedHatAI/Qwen3.6-35B-A3B-NVFP4
- **Base model**: Qwen/Qwen3.6-35B-A3B
- **Architecture**: Qwen3_5MoeForConditionalGeneration (hybrid DeltaNet + attention, MoE)
- **Parameters**: 35B total, 3B active (256 experts, 8 routed + 1 shared)
- **Quantization**: NVFP4 (compressed-tensors, FP4 weights + FP8 scales)
- **Hardware**: NVIDIA RTX 6000 Blackwell (96GB VRAM)

## Final Working Configuration

```
python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name qwen36-35b-a3b-nvfp4 \
  --host 0.0.0.0 --port 8080 \
  --trust-remote-code \
  --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
  ${MEMORY_PARAMS} \
  --quantization compressed-tensors \
  --reasoning-parser qwen3 \
  --moe-backend marlin \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --chat-template /model/chat_template.jinja \
  --max-cudagraph-capture-size 128
```

**Key settings**:
- `--moe-backend marlin`: 23% faster decode than flashinfer_cutlass (195 vs 159 TPS)
- `--reasoning-parser qwen3`: Enables thinking/reasoning mode
- `--tool-call-parser qwen3_coder`: Qwen-native tool call format (`<function=name><parameter=key>value</parameter></function>`)
- `--quantization compressed-tensors`: Required for NVFP4 format
- `--chat-template /model/chat_template.jinja`: Model's custom template for tool calling

---

## What We Tried (Chronological)

### Attempt 1: Initial catalog entry creation
**Status**: COMPLETED

Created inference.yaml based on:
- RedHatAI README deploy command: `vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 --reasoning-parser qwen3 --moe_backend flashinfer_cutlass`
- Reference from qwen3-coder-next-nvfp4-redhat catalog entry (same architecture family)

### Attempt 2: vLLM v0.19.0 fails with TokenizersBackend
**Status**: FAILED → FIXED

- vLLM v0.19.0 ships transformers 4.57.6
- Model's tokenizer_config.json uses `"tokenizer_class": "TokenizersBackend"` (transformers 5.x only)
- Error: `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported`
- **Fix**: Upgraded to `vllm/vllm-openai:v0.19.1-cu130` (ships transformers 5.5.3)
- Sources: vLLM Issues #36443, #35998, #38024; vLLM Release v0.19.1

### Attempt 3: flashinfer_cutlass MoE backend
**Status**: WORKING — 159 TPS

- Deployed with `--moe-backend flashinfer_cutlass` (README recommendation)
- Model loaded: 21.88 GiB memory, 14.5s load time
- Text inference works, reasoning mode works
- Benchmark: **159 tokens/s** decode at concurrency 1
- Vision weight warnings: "Parameter visual.blocks.X not found in params_dict, skip loading" — visual encoder may not be loading correctly

### Attempt 4: marlin MoE backend
**Status**: WORKING — 195 TPS (WINNER)

- Switched to `--moe-backend marlin`
- Warning: "Your GPU does not have native support for FP4 computation but FP4 quantization is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel."
- Despite the warning, marlin is **23% faster**: **195 tokens/s** vs 159 tokens/s
- Source: vLLM Issue #38971 predicted ~20% improvement for single-user scenarios

### Attempt 5: Tool call parser testing
**Status**: COMPLETED

- `hermes` parser: Did NOT parse tool calls (wrong format)
- `pythonic` parser: Did NOT parse tool calls (wrong format)
- `qwen3_coder` parser: **WORKS** — properly parses `<function=name><parameter=key>value</parameter></function>` format
- The chat_template.jinja defines this Qwen-native format, same as Qwen3-Coder-Next

### Attempt 6: VRAM measurement — minimum working memory
**Status**: COMPLETED

Binary search for minimum `min_gi` for 32k context:
- 28 GiB (0.29 util): **FAILS** — OOM during engine core initialization/profiling
- 30 GiB (0.31 util): **WORKS** — model loads, inference confirmed
- 42 GiB (0.44 util): **WORKS** — 262k context, measured ~42.5 GiB actual VRAM

**VRAM breakdown** (from vLLM logs):
- Model weights: 21.88 GiB
- Minimum overhead (CUDA context, profiling, encoder cache): ~8 GiB
- KV cache for 262k context: ~12 GiB (hybrid DeltaNet = minimal KV cache)

### Attempt 7: Prefill TPS measurement
**Status**: COMPLETED

Methodology: 10 sequential requests with ~5000 random words (~5016 tokens) each, max_tokens=5 to minimize decode.
Prefill time isolated by subtracting estimated decode time (5 tokens / 195 TPS).

```
Req 1:  5016 tok, 0.203s prefill → 24,663 tok/s (warmup)
Req 2:  5015 tok, 0.140s prefill → 35,889 tok/s
Req 3:  5016 tok, 0.407s prefill → 12,328 tok/s (outlier — likely GC/jitter)
Req 4:  5016 tok, 0.270s prefill → 18,593 tok/s
Req 5:  5016 tok, 0.179s prefill → 27,956 tok/s
Req 6:  5015 tok, 0.140s prefill → 35,898 tok/s
Req 7:  5016 tok, 0.144s prefill → 34,787 tok/s
Req 8:  5016 tok, 0.142s prefill → 35,392 tok/s
Req 9:  5016 tok, 0.141s prefill → 35,495 tok/s
Req 10: 5015 tok, 0.145s prefill → 34,493 tok/s

Average (excl warmup): 30,092 tok/s
Steady-state (req 6-10): ~35,000 tok/s
```

**Note**: vLLM engine logs show "Avg prompt throughput: 510 tokens/s" but this is a 10-second rolling average — the request completes in <0.2s so it's diluted across the window. Client-side measurement with decode subtraction is more accurate for burst prefill.

---

## TODO: Before Marking Stable

1. ~~**Deploy and verify startup**~~ — vLLM v0.19.1 works
2. ~~**Download missing files**~~ — All files downloaded
3. ~~**Measure VRAM**~~ — Binary search: 28 fails, 30 works, 42 works for 262k
4. ~~**Test text inference**~~ — Works with reasoning mode
5. **Test vision inference** — Visual weight warnings suggest encoder may not load properly
6. ~~**Test reasoning mode**~~ — `--reasoning-parser qwen3` works
7. ~~**Test tool calling**~~ — `--tool-call-parser qwen3_coder` works
8. ~~**Benchmark TPS**~~ — marlin: 195 TPS, flashinfer_cutlass: 159 TPS
9. ~~**Update memory profiles**~~ — Measured and updated

## Optimization Summary

### Performance Results

| Test | Result |
|------|--------|
| **MoE backend: marlin** | **195 TPS** decode @ concurrency 1 |
| **MoE backend: flashinfer_cutlass** | 159 TPS decode @ concurrency 1 |
| **marlin vs flashinfer_cutlass** | **+23% faster** (marlin wins) |
| **Prefill throughput** | **~30,000 TPS** avg, ~35,000 peak (5k token prompts) |
| **Text inference** | Working |
| **Reasoning mode** | Working (`--reasoning-parser qwen3`) |
| **Tool calling** | Working (`--tool-call-parser qwen3_coder`) |
| **Vision inference** | Untested (weight loading warnings) |

### Optimal Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `--moe-backend` | `marlin` | 23% faster than README-recommended flashinfer_cutlass |
| `--tool-call-parser` | `qwen3_coder` | Only parser that works (hermes, pythonic both fail) |
| `--reasoning-parser` | `qwen3` | Enables thinking/reasoning tokens |
| `--chat-template` | `/model/chat_template.jinja` | Required for Qwen-native tool call format |
| `--quantization` | `compressed-tensors` | Required for NVFP4 |
| vLLM image | `v0.19.1-cu130` | v0.19.0 fails (missing TokenizersBackend in transformers 4.57.6) |

### VRAM Measurements

| Profile | min_gi | GPU Util | Status |
|---------|----------|----------|--------|
| 262k context (fp8 kv) | 42 GiB | 0.44 | Measured: ~42.5 GiB actual |
| 128k context (fp8 kv) | 36 GiB | 0.38 | Interpolated |
| 32k context (fp8 kv) | 30 GiB | 0.31 | Measured: 28 fails, 30 works |

**VRAM breakdown**: weights 21.88 GiB + overhead ~8 GiB + KV cache varies (hybrid DeltaNet = minimal)

### End-to-End Verification (final deployment)

```
=== TEXT + REASONING ===
Content: Paris
Has reasoning: True
Finish: stop

=== TOOL CALLING ===
Tool calls: 1
  Function: get_stock_price
  Args: {"symbol": "AAPL"}
Finish: tool_calls

=== VRAM ===
63146 MiB used (includes chandra-ocr-2 co-deployed)
```

---

## Current Status

**Experimental** — text inference, reasoning, and tool calling work. Vision untested (weight loading warnings).

## Constraints
- Vision encoder: warnings during loading ("Parameter visual.blocks.X not found in params_dict") — may not work
- Marlin warning about non-native FP4 (but still faster than flashinfer_cutlass)
- README says "preliminary version" — upstream model may change
- Model uses heavy reasoning tokens — even simple queries use 100+ thinking tokens
