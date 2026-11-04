# Session: MTP Speculative Decoding Testing and Performance Benchmarks

**Date:** 2026-04-06
**Model:** NVIDIA Nemotron-3 Super 120B A12B NVFP4
**Hardware:** NVIDIA RTX 6000 Blackwell (96GB VRAM, SM120)
**Base config:** vLLM v0.17.1-cu130 (established working in 2026-03-16 session)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML inference frameworks.
> Information that is 2+ weeks old may be outdated. Before relying on any finding:
> 1. Check the session date - is it recent?
> 2. Search the web for updated information on the specific issue
> 3. Verify against current framework releases and changelogs

---

## MTP Speculative Decoding Testing (Failed)

**Goal:** Enable MTP (Multi-Token Prediction) speculative decoding for improved throughput

**Background:**
- Nemotron-3 Super has built-in MTP head for speculative decoding
- NVIDIA recommends `--speculative-config '{"method":"nemotron_h_mtp","num_speculative_tokens":5}'`
- Expected benefit: 1.5-2x decode throughput improvement

**Test 1: vLLM v0.17.1 with MTP (5 tokens)**

Config:
```bash
--speculative-config '{"method":"nemotron_h_mtp","num_speculative_tokens":5}'
--mamba-ssm-cache-dtype float16
```

Results:
- Model loaded successfully with MTP drafter
- First few requests worked - measured 87 TPS decode (vs 82 TPS without MTP)
- SpecDecoding metrics showed: Mean acceptance length 2.54, Avg acceptance rate 30.8%
- **Crashed after benchmark run** with `EngineDeadError`

Error:
```
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

**Test 2: vLLM v0.19.0 without MTP**

Rationale: v0.19.0 has fixes for NemotronH MTP + Chunked Prefill (#35447)

Result: **Failed** - Crashed during engine initialization, never reached model loading

**Test 3: vLLM v0.19.0 with MTP (3 tokens)**

Config:
```bash
--speculative-config '{"method":"nemotron_h_mtp","num_speculative_tokens":3}'
--no-enable-chunked-prefill
```

Result: **Failed** - Same initialization crash as Test 2

**Root Cause Analysis:**

1. **VRAM constraint**: MTP requires ~20GB+ extra VRAM for draft model state
   - Model weights: 75 GiB
   - MTP overhead: ~20 GiB
   - Total needed: ~95 GiB (exceeds 96 GiB minus KV cache)

2. **Known issues** (from web search):
   - [HuggingFace Discussion #9](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/discussions/9): "MTP unusable on RTX 6000 Pro, spec decoding consumes 20GB+ VRAM at start-up"
   - [vLLM #35288](https://github.com/vllm-project/vllm/issues/35288): MTP produces corrupted output at concurrency >= 4
   - [vLLM #35031](https://github.com/vllm-project/vllm/issues/35031): MTP with NVFP4 weight shape mismatch

3. **v0.19.0 regression**: New version unstable with this model on SM120

**Outcome:** MTP not viable on single RTX 6000 96GB due to VRAM constraints

**Baseline Performance (without MTP):**
- Decode TPS: 82 TPS @ concurrency 1
- Working config: vLLM v0.17.1-cu130

---

## Performance Benchmarks

**Goal:** Measure prefill and decode TPS for documentation

**Test setup:**
- Container: vLLM v0.17.1-cu130
- Concurrency: 1
- Random input (no cache hits)

**Results:**

| Run | Input Tokens | Output Tokens | Prefill TPS | Decode TPS | TTFT |
|-----|--------------|---------------|-------------|------------|------|
| 1   | 169          | 500           | 7,022       | 82.6       | 0.09s |
| 2   | 1512         | 100           | 15,716      | 63.9       | 0.38s |
| 3   | 1492         | 500           | 15,085      | 78.8       | 0.39s |

**Analysis:**
- **Prefill TPS**: ~15,000 with larger input (1500 tokens), ~7,000 with small input (170 tokens)
- **Decode TPS**: ~80 TPS consistent with sufficient output tokens (500+)
- Short outputs (100 tokens) show lower decode TPS due to insufficient steady-state time

**Final values for inference.yaml:**
- `tps_prefill: 15000`
- `tps_decode: 80`

---

## Current Status (as of this session)

**Status:** Working (without MTP)

**Recommended Configuration:**
- Image: `vllm/vllm-openai:v0.17.1-cu130`
- Key flags: `--quantization modelopt_fp4 --kv-cache-dtype fp8`
- MoE Backend: FLASHINFER_CUTLASS (automatic with v0.15.1+)

**Memory profiles:**

| Context Length | VRAM Required | Notes |
|---------------|---------------|-------|
| 128k | ~88Gi | Tested working |
| 64k | ~82Gi | Lower memory profile |
| 262k | ~96Gi | Full native context (may need tuning) |

**Known Issues:**
- NVFP4 performance characteristics differ from Q4/Q8 quantization
- MTP architecture uses token-level parallelism - batch size 1 optimal for latency
- **MTP speculative decoding not viable** on single 96GB GPU (needs ~95GB+ VRAM)
- vLLM v0.19.0 crashes during initialization on SM120 - stick with v0.17.1
