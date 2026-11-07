# Session: Right-Size Resource Requests and Readiness Probe

**Date**: 2026-04-23

## Problem

Pod stuck in Pending state — K8s scheduler rejected it with "Insufficient memory".
The pod's total scheduling footprint was 16 GiB (8+4 GiB container requests + 4 GiB dshm),
but only ~10.5 GiB was available on the single-node cluster.

## Root Cause

Resource requests and readiness probe were set conservatively without measuring actual
RAM usage. The model weights load into VRAM (not RAM), so container RAM usage is just
framework overhead. The 0.9B model is tiny.

## Changes to `inference.yaml`

| Component | Before | After | Rationale |
|-----------|--------|-------|-----------|
| kserve-container request | 8 GiB | 2 GiB | PP-DocLayoutV3 is ~0.2 GiB; 8 GiB was ~40x overprovisioned |
| kserve-container limit | 16 GiB | 8 GiB | Headroom for burst, not steady state |
| vlm-server request | 4 GiB | 2 GiB | Model weights go to VRAM; RAM is just vLLM process overhead |
| vlm-server limit | 12 GiB | 6 GiB | Headroom for burst |
| dshm sizeLimit | 4 GiB | 2 GiB | 0.9B model IPC doesn't need 4 GiB of shared memory |
| readinessProbe initialDelaySeconds | 120 | 20 | Model loads in under 60s with bundled images; 120s was unnecessary wait |
| readinessProbe failureThreshold | 60 | 18 | 60 × 10s = 10 min tolerance was excessive; 18 × 10s = 3 min is plenty |

**Scheduling footprint**: 16 GiB → 6 GiB (container requests + dshm).
**Max startup tolerance**: 720s (12 min) → 200s (3.3 min). Still generous for a 0.9B model.

## Timeline

- **2026-04-23 04:30**: Pod stuck in Pending — node had insufficient memory for scheduling. Reduced overprovisioned resource requests and shared memory.
