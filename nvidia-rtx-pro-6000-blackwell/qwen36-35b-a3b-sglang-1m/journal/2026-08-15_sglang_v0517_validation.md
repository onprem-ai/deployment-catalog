# SGLang v0.5.17 validation

Date: 2026-08-15
Hardware: NVIDIA RTX PRO 6000 Blackwell, 96 GiB VRAM, SM120
Catalog config: `rtx6k--qwen36-35b-a3b-sglang-1m`

## Purpose

Replace the custom SGLang v0.5.11 memory-leak overlay with SGLang v0.5.17. The
newer release includes the upstream Mamba radix-cache LRU reference-cycle fix
and uses the current FlashInfer/CUTLASS kernel stack.

## Image

The image was built from `lmsysorg/sglang:v0.5.17` using:

- `_docker/sglang-blackwell-v0517/Dockerfile`
- `_docker/sglang-blackwell-v0517/build.sh`
- `_docker/sglang-blackwell-v0517/warmup_deep_gemm.sh`

Published image:

```text
onpremai/sglang-blackwell:v0.5.17
sha256:ec5a08568603fb5cdb08cd52ceea205890f5fd44f525f674677077a2c7346810
```

The image was pushed to both the node-local model registry and Docker Hub. The
Docker Hub manifest resolves to the same digest as the node-local registry.

## Native memory-leak fix

The running v0.5.17 container was inspected directly. Its
`LRUList._remove_node` implementation clears both linked-list references after
removal:

```python
setattr(node, self.prv, None)
setattr(node, self.nxt, None)
```

This is the upstream fix for the Mamba radix-cache reference cycle. The custom
v0.5.11 Python overlay is therefore no longer required by these catalog entries.

## Deployment validation

Deployment and inference testing used the application API. Kubernetes commands
were read-only and used only for readiness, logs, and cache inspection.

The local-registry rewrite produced:

```text
127.0.0.1:31998/onprem-ai/sglang-blackwell@sha256:ec5a08568603fb5cdb08cd52ceea205890f5fd44f525f674677077a2c7346810
```

Observed startup milestones:

| Milestone | Elapsed time |
|---|---:|
| Container started | approximately 50 seconds |
| InferenceService Ready | approximately 147 seconds |

The shared readiness helper confirmed the pod was Ready with zero restarts:

```bash
source dev/helpers/wait_for_deployment.sh
wait_for_deployment \
  inference \
  "app=isvc.rtx6k--qwen36-35b-a3b-sglang-1m-predictor" \
  120 \
  --skip-vram-check
```

The app chat-test API returned HTTP 200 and generated the requested final text
`sglang-v0517-ok`.

## Real-workload benchmark

Benchmarks used `POST /api/inference-services/benchmark` with random input. Random prompts avoid radix-cache
hits that would inflate prompt processing measurements.

| Workload | Input tokens | Output tokens | Concurrency | Generation speed | Success rate |
|---|---:|---:|---:|---:|---:|
| Balanced | 482 | 500 | 1 | 329.74 tok/s | 100% |
| Scaling | 201 | 200 | 1 | 397.22 tok/s | 100% |
| Scaling | 201 | 200 | 2 | 402.15 tok/s | 100% |
| Scaling | 201 | 200 | 4 | 815.68 aggregate tok/s | 100% |
| Prefill-focused | 1350 | 100 | 1 | 239.86 tok/s | 100% |
| Long context | 2776 | 100 | 1 | 181.81 tok/s | 100% |
| Decode-focused | approximately 50 words | 500 | 1 | approximately 395 tok/s | request completed |

The benchmark tool emitted invalid negative prompt-throughput values and zero
TTFT values for completed tests. The decode-focused run also failed to serialize
its result because an auxiliary metric was positive infinity. Those fields are
not accepted as performance evidence, so the existing catalog prefill/decode
numbers were not replaced from this run.

## Kernel-cache observation

Before the benchmark suite, the running container had:

```text
cached_ops files: 15
generated files: 95
cache size: 7.7 MiB
```

After balanced, scaling, prefill-focused, long-context, and decode-focused
workloads, the counts were unchanged and no file had a timestamp newer than the
benchmark start. Pod logs contained no JIT, compiler, autotune, DeepGEMM, PTXAS,
or CUDA error messages during the suite.

The build-time warmup is narrow, but these representative workloads did not
trigger additional runtime compilation. A broader model-specific prewarm image
is therefore not justified without a future workload that demonstrates a real
JIT latency problem.

## Memory and health

During the benchmark window, VictoriaMetrics reported the predictor container
working set between 9.402 GiB and 9.454 GiB. The first-to-last change was only
0.051 GiB.

After the suite:

- Pod state: Ready
- Pod restarts: 0
- Host RAM available from `atop`: 43.1 GiB
- GPU memory in use: approximately 69.9 GiB
- Compile or runtime error log lines: none

## Decision

Pin both Qwen3.6 35B-A3B SGLang catalog entries to the v0.5.17 image digest.
Keep the current cache-baked image as-is; do not add a broader warmup matrix
unless future measured workloads show runtime JIT compilation or unacceptable
cold-request latency.
