# Session: Container Upgrade and TPS Benchmark

**Date:** 2026-04-06
**Model:** GPT-OSS-120B (`openai/gpt-oss-120b`)
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)

> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML frameworks.
> Information 2+ weeks old may be outdated. Verify against current releases.

## Model Info

- **Architecture**: MoE (120B total, 5B active per token)
- **Weights**: ~92 GiB
- **Container**: `nvcr.io/nvidia/vllm:26.02-py3` (vLLM 0.15.1)

---

## Container Upgrade: 25.12-py3 to 26.02-py3

Tested container upgrade from `nvcr.io/nvidia/vllm:25.12-py3` (vLLM 0.11.1) to `26.02-py3` (vLLM 0.15.1).

**Decode TPS improvement**: 186 -> 200 TPS (~7.5% improvement)

## TPS Benchmark Session

Measured using app benchmark API with random input (no cache effects).

**Test setup:**
- Concurrency: 1
- Random input: 1400-1900 tokens (num_words: 1500-2000)
- Output: 100-500 tokens

| Run | Input Tokens | Output Tokens | Prefill TPS | Decode TPS | TTFT |
|-----|--------------|---------------|-------------|------------|------|
| 1   | 1886         | 100           | 14,675      | 116.86     | 0.49s |
| 2   | 1886         | 100           | 9,744       | 116.48     | 0.74s |
| 3   | 1377         | 500           | 8,281       | 174.83     | 0.66s |
| 4   | 1408         | 500           | 7,499       | 174.50     | 0.73s |
| 5   | 199          | 450           | 1,400       | **200.11** | -     |

**Observations:**
- **Prefill TPS varies significantly** (7.5k - 15k) depending on cache state and input size
- **Decode TPS depends on input context size:**
  - ~200 input tokens → 200 TPS (small KV cache)
  - ~1500 input tokens → 175 TPS (larger KV cache overhead)
- Short outputs (100 tokens) show lower decode TPS (~117) due to insufficient steady-state time
- Previous 80k prefill measurement was from warm cache with tiny input - not representative

**Final values for inference.yaml:**
- `tps_prefill: 10000` (conservative middle of measured range with 1400+ input tokens)
- `tps_decode: 200` (measured with typical small-context usage)

## Benchmark Commands

```bash
# Start benchmark with random input
curl -s -X POST "http://localhost:8437/api/inference-services/benchmark?name=gpt-oss-120b&namespace=<namespace>" \
  -H "Content-Type: application/json" \
  -d '{
    "base_url": "http://<inference-service-address>/v1",
    "model": "gpt-oss-120b",
    "concurrency": "1",
    "max_tokens": 500,
    "num_words": 1500
  }'

# Check results
curl -s "http://localhost:8437/api/inference-services/benchmark?name=gpt-oss-120b&namespace=<namespace>" | jq '.results'

# Delete benchmark job
curl -s -X DELETE "http://localhost:8437/api/inference-services/benchmark?name=gpt-oss-120b&namespace=<namespace>"
```
