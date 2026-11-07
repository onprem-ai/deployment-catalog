# Session: Initial Deployment Planning

**Date:** 2026-02-28
**Hardware:** nvidia-rtx6000bw (96GB VRAM)
**Catalog File:** devstral-2-123b-nvidia-rtx6000bw.yaml

## REMINDER: Use Official APIs Only

- Download via: UI button or POST /api/catalog/{id}/download
- Deploy via: UI button or POST /api/catalog/{id}/deploy
- NO manual downloads, NO manual kubectl apply

## Goal

Deploy Devstral-2-123B-Instruct Q4_K_M on NVIDIA RTX 6000 Blackwell with optimal performance.

## Plan

### Phase 1: Initial Deployment
1. [x] Verify model downloaded (71GB, 2 shards)
2. [x] Verify catalog config exists and is loaded
3. [x] Rebuild backend to pick up catalog config
4. [ ] User triggers Deploy from UI (64k context, q8_0 cache)
5. [ ] Monitor pod startup via `kubectl get pods -n <namespace> -w`
6. [ ] Check pod logs for any errors during model loading

### Phase 2: Verify Basic Functionality
7. [ ] Confirm pod reaches Running/Ready state
8. [ ] Test health endpoint: `curl http://<pod-ip>:8080/health`
9. [ ] Test basic inference with short prompt
10. [ ] Record baseline performance (tok/s, memory usage)

### Phase 3: Optimize if Needed
11. [ ] If OOM or slow: adjust context length or cache type
12. [ ] If crashes: check llama.cpp logs for specific errors
13. [ ] Test with different memory profiles (128k vs 64k)

### Phase 4: Document Results
14. [ ] Record final working config
15. [ ] Update catalog maturity (experimental → stable if working)
16. [ ] Document measured performance metrics

## Initial State

- Downloaded: **YES** (71GB, 2 GGUF shards)
- Model path: `/persistent/models/unsloth--Devstral-2-123B-Instruct-2512-GGUF--Devstral-2-123B-Instruct-2512-Q4_K_M/`
- Catalog config: exists (`devstral-2-123b-nvidia-rtx6000bw.yaml`)
- Maturity: experimental
- Previous deployment attempts: no

## Model Info

- **Source:** unsloth/Devstral-2-123B-Instruct-2512-GGUF
- **Quantization:** Q4_K_M
- **Size:** ~70GB (47GB + 24GB shards)
- **Architecture:** Mistral-based, 123B params
- **Context:** 131072 tokens (native)

## Current Catalog Config

```yaml
# Key settings:
image: ghcr.io/ggml-org/llama.cpp:server-cuda
context: 65536 (conservative start)
cache: q8_0 for K and V
batch: 4096
flash-attn: on
no-mmap: enabled
ngl: 999 (all layers on GPU)
```

## Attempts

### Attempt 1: [PENDING]
**Change:** Initial deployment with current config
**Result:**
**Error:**
**Logs:**
```
```
**Analysis:**

---

## Solution
[To be filled after successful deployment]

## Measured Performance
- Context length:
- Memory usage:
- Tokens/sec:

## Lessons Learned
[To be filled]
