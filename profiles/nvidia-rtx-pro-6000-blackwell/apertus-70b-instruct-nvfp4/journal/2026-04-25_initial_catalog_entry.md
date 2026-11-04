# 2026-04-25 — Initial Catalog Entry

**Goal:** Add Apertus-70B-Instruct-2509-NVFP4 to the RTX 6000 Blackwell catalog.

## Model Research

- **Source:** bg-digitalservices/Apertus-70B-Instruct-2509-NVFP4
- **Base model:** swiss-ai/Apertus-70B-Instruct-2509
- **Architecture:** 70B dense transformer with xIELU activation
- **Quantization:** NVFP4 W4A4 (FP4 weights + FP4 activations), NVIDIA ModelOpt v0.43
- **Calibration:** 4096 samples from CNN/DailyMail, batch size 32, seq_len 1024
- **Quantized on:** NVIDIA H200
- **Disk size:** ~35 GiB (5 safetensors shards)
- **Context:** 65,536 tokens
- **Languages:** 1,811 (multilingual)
- **License:** Apache 2.0
- **Tool use:** Yes (instruct model)

## README Findings

README recommends vLLM with:
```bash
vllm serve bg-digitalservices/Apertus-70B-Instruct-2509-NVFP4 \
  --quantization modelopt \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 65536 \
  --trust-remote-code
```

Key flags: `--quantization modelopt`, `--dtype auto`, `--trust-remote-code`.

## Configuration Decisions

- **Image:** `vllm/vllm-openai:v0.19.1-cu130` — latest stable cu130 for Blackwell
- **Quantization:** `--quantization modelopt` per README
- **KV cache:** FP8 (`--kv-cache-dtype fp8_e5m2`) to save VRAM on 70B dense model
- **Tool parser:** `hermes` — standard for instruct models without specific parser
- **Memory estimates:** Initial guesses, need VRAM binary search after deployment

## Outcome

Catalog entry created with initial VRAM estimates. Next steps:
1. Download model via catalog API
2. Deploy and verify inference works
3. VRAM binary search for accurate memory profiles
4. Performance benchmarks (TPS)
