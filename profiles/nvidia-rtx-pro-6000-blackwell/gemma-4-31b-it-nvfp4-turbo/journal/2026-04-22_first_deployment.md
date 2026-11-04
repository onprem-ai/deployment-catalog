# 2026-04-22 — First Deployment and Benchmarks

**Goal:** Deploy model and measure actual VRAM and TPS.

## Setup

- Image: `vllm/vllm-openai:cu130-nightly` (8GB, pulled in ~2m47s)
- Config files had to be downloaded separately (known download system bug)
- Memory profile: 128k context, fp8_e4m3 KV cache, max-num-seqs 128

## Working Configuration

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name gemma-4-31b-it-nvfp4-turbo \
  --host 0.0.0.0 --port 8080 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --max-model-len 131072 --kv-cache-dtype fp8_e4m3 --max-num-seqs 128 \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

Key settings:
- `--quantization modelopt` — NVIDIA modelopt format, activates Blackwell FP4 tensor cores
- `--kv-cache-dtype fp8_e4m3` — Halves KV cache memory on Blackwell
- `--max-num-batched-tokens 8192` — README-recommended for throughput
- `--enable-prefix-caching` — README-recommended, 22% faster for repeated prefixes

## VRAM Measurement

- Total GPU: 97,887 MiB (95.6 GiB)
- Used: 71,413 MiB (69.7 GiB) at `gpu-memory-utilization 0.95`
- Estimate was 70 GiB — appeared to confirm, but this was just vLLM filling the allocation (see 2026-04-24 session)

## Benchmark Results

**Decode workload** (154 input tokens, 200 output tokens, 10 requests):
- Decode: 51.22 tok/s
- Prefill: 11,080 tok/s (short input)
- TTFT: 50ms
- Duration: 3.91s total

**Prefill workload** (1,350 input tokens, 10 output tokens, 5 requests):
- Prefill: 12,448 tok/s
- TTFT: 420ms

README claims 15,359 tok/s prefill at 1K/200 workload. Our 12,448 is lower — possibly due to different input size and no CUDA graph warmup.

## Outcome

Model deployed and serving. Decode TPS matches README exactly. Prefill slightly lower than claimed.
