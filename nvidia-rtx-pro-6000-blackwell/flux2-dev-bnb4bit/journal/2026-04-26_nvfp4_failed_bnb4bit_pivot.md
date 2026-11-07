# 2026-04-26 — NVFP4 Loading Failed + Server Crash → Pivot to bnb-4bit

## NVFP4 Single-File Loading: FAILED

### Attempt 1: `from_single_file` without local config

```python
transformer = Flux2Transformer2DModel.from_single_file(
    "flux2-dev-nvfp4.safetensors",
    quantization_config=NVIDIAModelOptConfig(quant_type="NVFP4", quant_method="modelopt"),
    torch_dtype=torch.bfloat16,
)
```

**Error:** `GatedRepoError` — `from_single_file` tries to fetch `transformer/config.json` from the gated `black-forest-labs/FLUX.2-dev` repo. 401 Unauthorized.

### Attempt 2: `from_single_file` with local config

Downloaded `config.json` from `diffusers/FLUX.2-dev-bnb-4bit` transformer subfolder. Stripped bnb quantization_config. Passed via `config=` parameter.

**Error:** `RuntimeError: chunk expects at least a 1-dimensional tensor` in `convert_flux2_double_stream_blocks`. The checkpoint converter tries to chunk fused QKV weights, but NVFP4 tensors are packed 4-bit with scale factors — they can't be chunked like normal tensors.

### Root Cause

`from_single_file` is not compatible with NVFP4-quantized safetensors. The diffusers checkpoint conversion pipeline assumes standard tensor formats. NVFP4 weights are in a packed format with associated scale tensors that the converter doesn't understand.

No published example exists for loading `FLUX.2-dev-NVFP4` via `from_single_file` in diffusers. The NVFP4 single-file format is intended for TensorRT-LLM visual_gen runtime or TorchAO (both require Blackwell + BF16 base weights to quantize from).

## Server Crash During bnb-4bit Transformer Download

While downloading the bnb-4bit transformer shards (~18 GiB) via `huggingface_hub.hf_hub_download`, the host became completely unresponsive (SSH hung) and only recovered after a forced reboot.

### What the Logs Show

The crash occurred between 04:53 and 05:47 UTC. Evidence from surviving logs:

1. **Last journald entry (boot -1):** `04:53:47` — SSH disconnect
2. **Last rsyslog entry:** `04:55:29` — `systemd-journald[560]: Under memory pressure, flushing caches`
3. **52-minute gap** — no log entries at all until cold boot at `05:47:48`
4. **No OOM killer trace** — no `oom-kill` or kernel panic logged in journald, kern.log, or syslog
5. **Last kernel message in journald:** `04:44:55` — normal CNI traffic, nothing abnormal

### What We Do NOT Know (Inconclusive Root Cause)

- Whether the OOM killer actually fired (no trace in any log)
- Whether it was OOM, a kernel panic, GPU driver crash, or something else entirely
- Which process(es) consumed the RAM (no `oom-kill` log identifying the culprit)
- Why journald stopped writing 8+ minutes before rsyslog's last entry

### Why the Logs Are Incomplete

- rsyslog writes to files on disk — its last entry (`memory pressure, flushing caches`) survived
- journald stores in a memory-mapped journal — under extreme memory pressure, it flushes caches (confirmed by rsyslog) and then can't write further
- The forced reboot prevented any graceful log flush
- No swap means no buffer time between pressure and total unresponsiveness

### Context: No-Swap is Intentional

The host's storage design states:
> **No swap**: AI inference degrades instantly with swap. Better to OOM-kill than hang. Kubernetes schedulers expect swap off.

The system relies on kubelet memory eviction (hard eviction at 2 GiB free) as protection. However, this crash was a **bare-metal Python process outside Kubernetes**, so kubelet eviction could not protect against it.

### Impact

- The host required a forced reboot
- `/tmp/flux-venv/` was lost (wiped on reboot)
- Downloaded transformer files on `/persistent/` survived

## Decision: Pivot to bnb-4bit Approach

### Why NVFP4 is not viable right now

1. `from_single_file` doesn't support NVFP4 tensor format
2. `NVIDIAModelOptConfig` with `from_pretrained` needs diffusers-format directory (not available for NVFP4)
3. Quantizing BF16→NVFP4 at load time requires 64 GiB weights from gated repo + HF auth

### Why bnb-4bit works

1. `diffusers/FLUX.2-dev-bnb-4bit` is non-gated, in diffusers format
2. Native `from_pretrained` support — no checkpoint conversion needed
3. Transformer is ~18 GiB (NF4), similar VRAM footprint to NVFP4 (~20 GiB)
4. Officially documented by Black Forest Labs
5. Transformer shards already downloaded to `/persistent/models/flux2-dev-bnb4bit/transformer/`

### Updated VRAM Estimate (bnb-4bit)

| Component | Size |
|-----------|------|
| NF4 transformer | ~18 GiB |
| Mistral3 text encoder (bnb 4-bit) | ~15 GiB |
| VAE (BF16) | ~0.3 GiB |
| Inference activations | ~5-10 GiB |
| **Total estimate** | **~33-43 GiB** |

## Lessons Learned

### 1. Never Run Heavy AI/ML Processes on the Host — Use Pods

The crash happened because a bare-metal Python process (pip install + huggingface_hub download) ran outside Kubernetes, bypassing all memory protection:

- **Kubelet eviction** only protects against pod memory usage — bare-metal processes are invisible to it
- **Cgroup memory limits** only apply to containers/pods — bare-metal processes have no ceiling
- **No swap** means no buffer — once RAM is exhausted, the system dies instantly

The correct approach: run model testing inside a Kubernetes pod with resource limits. The pod's cgroup enforces a memory ceiling, and if it OOMs, only the pod dies — the node stays healthy.

### 2. The No-Swap Rationale Needs Nuance

The no-swap design states: "AI inference degrades instantly with swap. Better to OOM-kill than hang."

This is true for **CPU-based inference** (measured 8x latency degradation on phi-2). But for **GPU inference**, model weights and KV cache live in VRAM — completely outside the swap path. System RAM is only used for process overhead, k3s, and OS services. A small swap file would:

- Give the OOM killer time to fire and log before the system goes dark
- Not affect GPU inference latency at all (VRAM is not swappable)
- Be consistent with K8s swap support (LimitedSwap GA in K8s 1.34, Aug 2025)
- Guaranteed QoS inference pods get `memory.swap.max = 0` automatically

### 3. Consider `panic_on_oom=1`

With `panic_on_oom=1`, a host-level OOM triggers an immediate kernel panic + reboot instead of the silent death we experienced. This does NOT affect pod OOM kills (those are handled by the cgroup OOM killer, which is a separate mechanism). We would at least get crash logs via kdump/pmesg.

Sources:
- [K8s Swap Memory Management](https://kubernetes.io/docs/concepts/cluster-administration/swap-memory-management/)
- [Tuning Linux Swap for K8s](https://kubernetes.io/blog/2025/08/19/tuning-linux-swap-for-kubernetes-a-deep-dive/)
- [ScaleOps: K8s Swap for GPU Workloads](https://scaleops.com/blog/kubernetes-swap-gpu-workload-sizing/)
- [kernel panic_on_oom docs](https://docs.kernel.org/admin-guide/cgroup-v1/memory.html)

## Next Steps

1. Recreate test venv (lost in reboot)
2. Keep venv minimal — avoid installing nvidia_modelopt and deepspeed (not needed for bnb-4bit)
3. **Test inference inside a pod, not on bare metal**
4. Update server.py and Dockerfile to use bnb-4bit approach
5. Update inference.yaml model_id and source fields
