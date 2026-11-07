# 2026-07-07 -- FP8 KV cache verified, memory profiles measured, benchmarks

Cheap-optimization session on the three-tier checkpoint (vLLM
v0.19.1-cu130, RTX 6000 Blackwell 96 GB). All measured, not estimated.

## 1. FP8 KV cache (fp8_e4m3) WORKS on the mixed checkpoint

The v1 70B ModelOpt entry had to fall back to BF16 KV (fp8_e5m2 rejected).
The compressed-tensors mixed checkpoint accepts `--kv-cache-dtype fp8_e4m3`
cleanly:

| Config | KV tokens | VRAM used | Note |
|--------|-----------|-----------|------|
| fp8 KV, util 0.90, 16k max-len | 207,760 | 90.5 GiB | exactly 2x the BF16-KV 103,872 |
| fp8 KV, util 0.90, 131k max-len | 207,760 | 90.5 GiB | comfortable |
| fp8 KV, util 0.90, 192k max-len | 207,760 | 90.5 GiB | comfortable |
| fp8 KV, util 0.95, 229k max-len | 238,880 | 94.97 GiB | works; whole card, no headroom |

Full 262,144 (the model's max_position_embeddings) is NOT reachable: needs
~40 GiB KV, max pool at util 0.95 is ~36.5 GiB.

Generation sanity under fp8 KV: French translation token-identical to the
BF16-KV run, arithmetic correct.

## 2. Memory profiles (BF16 KV, card verification)

| Profile | util | KV tokens | VRAM used | Card estimate |
|---------|------|-----------|-----------|---------------|
| 32k | 0.68 | 35,408 | 67.5 GiB | 65 Gi (close) |
| 65k | 0.79 | 69,648 | 78.5 GiB | 75 Gi (close) |

## 3. Throughput benchmarks (fp8 KV, 16k, util 0.90)

| Metric | Value |
|--------|-------|
| Prefill (8,471-token prompt) | 2,202 tok/s |
| Decode, concurrency 1 | 30.7 tok/s |
| Decode, concurrency 4 | 87.1 tok/s aggregate (21.8/stream) |
| Decode, concurrency 8 | 167.6 tok/s aggregate (21.0/stream) |

## 4. Quality gap measurement (lm-eval): three-tier vs FP8 baseline

Purpose: step 0 of the recovery proposal -- decide whether GPTQ-calibrated
NVFP4 (step 1) is warranted. gsm8k (limit 250) + 4 MMLU subjects (limit 100
each: elementary_mathematics, college_biology, philosophy,
computer_security), lm-eval local-completions against the served API, both
models with fp8 KV.

RESULTS:

| Task | Three-tier | FP8 baseline | Delta |
|------|-----------|--------------|-------|
| gsm8k strict-match (5-shot, 250) | 0.804 +/- 0.025 | 0.808 +/- 0.025 | -0.4 pts |
| gsm8k flexible-extract | 0.816 | 0.808 | +0.8 pts |
| mmlu college_biology (100) | 0.79 +/- 0.041 | 0.83 +/- 0.038 | -4 pts |
| mmlu computer_security (100) | 0.74 | 0.74 | 0 |
| mmlu elementary_mathematics (100) | 0.55 | 0.56 | -1 pt |
| mmlu philosophy (100) | 0.73 | 0.71 | +2 pts |
| mmlu subset average | 0.7025 | 0.71 | -0.75 pts |

**Verdict: no measurable quality gap.** Every delta is within its standard
error; the averages differ by well under 1 point, and the three-tier WINS two
of six comparisons. Per the recovery proposal's decision rule (gap <= 1 point
=> recovery moot), GPTQ-calibrated re-quantization (step 1) and QAD (step 2)
are NOT warranted. The data-free two-pass three-tier is quality-equivalent to
the near-lossless FP8 baseline at these sample sizes -- the 70B tolerates
NVFP4 MLPs far better than the 8B did, as predicted.

## Card updates to apply

- Recommended serve config becomes fp8 KV: `--kv-cache-dtype fp8_e4m3`.
- New long-context profiles: 131k and 192k at util 0.90; 229k at util 0.95
  (flag: zero headroom).
- performance: tps_prefill 2202, tps_decode 30.7 (conc 1), plus concurrency
  scaling table.
- Operational note: gemma-4-31b-it-nvfp4-sglang-mtp was undeployed for this
  session and intentionally NOT redeployed (user decision).
