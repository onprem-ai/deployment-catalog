# Session: Performance Benchmarks & NGC Container Testing

**Date**: 2026-04-06

## Performance Benchmarks

**Test setup:**
- Container: vLLM v0.17.1 + PR #38135
- Concurrency: 1
- Random input (no cache hits)

**Results:**

| Run | Input Tokens | Output Tokens | Prefill TPS | Decode TPS | TTFT |
|-----|--------------|---------------|-------------|------------|------|
| 1   | 223          | 500           | 1,367       | 196.5      | 0.45s |
| 2   | 1376         | 500           | 5,091       | 176.9      | 1.07s |
| 3   | 222          | 500           | 942         | 196.8      | 0.64s |

**Analysis:**
- **Prefill TPS**: ~5,000 with larger input (1400 tokens), varies significantly with small input
- **Decode TPS**: ~195-197 TPS with small input, ~177 TPS with large input (more KV cache overhead)
- Decode TPS matches documented value (195 TPS)

**Final values for inference.yaml:**
- `tps_prefill: 5000`
- `tps_decode: 195`

---

## NGC Container Testing

**Goal:** Test if NGC vLLM 26.02-py3 works without custom PR #38135

**Test:** Deployed with `nvcr.io/nvidia/vllm:26.02-py3` (vLLM 0.15.1)

**Result:** **Failed** - Engine core initialization failed

```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

**Conclusion:** PR #38135 is still required. NGC containers (tested: 26.02-py3) do not include the necessary patches for GPT OSS Puzzle architecture support.
