# 2026-08-15 - Prefill CUDA Graph Startup Failure

**Goal:** Diagnose the first Muse Glimmer NVFP4 plus DFlash deployment and correct the restart loop.

## Failure

The target and FP8-quantized DFlash assistant loaded successfully. SGLang then allocated the target and draft KV pools and completed FlashInfer autotuning. Startup failed while capturing the default breakable prefill CUDA graph:

```text
flashinfer.utils.BackendSupportedError: mm_mxfp8 does not support backend 'cute-dsl' with capability 120
```

The scheduler child emitted `Received sigquit`, but the top-level process exited with code 0. Kubernetes consequently reported each termination as `Completed`, restarted the container, and the original wait helper did not classify the loop as a crash.

## Wait Helper Fix

`dev/helpers/wait_for_deployment.sh` now reads both current and previous container logs after any restart. A restart with fatal log markers fails immediately even when the terminated container exit code is 0. A unit test reproduces the SGLang child-failure behavior and verifies that the helper exits 1 with the previous traceback visible.

## SGLang Research

SGLang's runtime error itself recommends either `tc_piecewise` or disabling prefill CUDA graphs. Upstream defines `tc_piecewise` as a supported prefill backend. Its piecewise implementation splits the forward graph and uses `torch.compile`-managed subgraphs, avoiding the incompatible breakable capture path while preserving prefill graph acceleration.

The catalog launch command now includes:

```text
--cuda-graph-backend-prefill tc_piecewise
```

This is preferred over disabling prefill graph acceleration because the failure is isolated to the default breakable backend and SGLang explicitly recommends `tc_piecewise` first.

## First Redeploy Error

The first source edit placed explanatory comments inside a backslash-continued shell command. In POSIX shell parsing, the continued comment terminated the command, so SGLang launched without the new flag and the next line failed with:

```text
--cuda-graph-backend-prefill: command not found
```

The comments were moved above the command so every continued argument line remains contiguous.

## Second Redeploy: KV Pool Over-allocation

With `tc_piecewise` active, the incompatible backend error was gone. Startup then reached FlashInfer autotuning but OOMed on a 20 MiB allocation. At `min_gi: 82`, the resolved `--mem-fraction-static 0.86` allocated 49.05 GiB for the target KV pool and 22.20 GiB for the DFlash KV pool, leaving only 0.70 GiB before autotuning and effectively no free memory at the failing allocation.

This is not a model-fit failure; it is KV pool over-allocation. The target plus FP8 draft weights use about 22.24 GiB. The profile was reduced explicitly to 75 GiB (`mem-fraction-static` approximately 0.79), returning roughly 7 GiB from KV capacity to startup graphs and autotuning. The full 131K context still has ample capacity because Muse Glimmer uses grouped-query attention and hybrid sliding-window attention.

## Third Redeploy: tc_piecewise Is Also Incompatible

The reduced pool left 20.25 GiB free and eliminated the OOM, proving the memory diagnosis. However, `tc_piecewise` reached its own prefill capture and invoked the same unsupported FlashInfer `mm_mxfp8` CuTeDSL backend on SM120. The flag changed the graph runner, not the incompatible MXFP8 operation selected during capture.

SGLang's error guidance lists disabling prefill CUDA graphs as the second supported workaround. The card now uses:

```text
--cuda-graph-backend-prefill disabled
```

## Fourth Redeploy: DFlash Verify Graph Is Also Incompatible

Disabling prefill graphs allowed startup to continue into DFlash target verification graph capture. The full decode/verify graph then invoked the same unsupported `mm_mxfp8` CuTeDSL backend. Therefore the incompatibility is general to CUDA graph capture of the mixed MXFP8 operations on SM120, not limited to prefill.

A graph-free run proved that ordinary generation also selected CuTeDSL and hung until SGLang's 600-second warmup timeout. Therefore graph capture was only where the unsupported backend failed quickly; disabling graphs did not fix the root cause.

## Root Cause: Incorrect MXFP8 GEMM Backend Selection

FlashInfer documents that SM120/SM121 supports `mm_mxfp8` only through CUTLASS with swizzled scales. CuTeDSL does not support SM12x. SGLang exposes the explicit backend choice and validates `flashinfer_cutlass` for Blackwell devices.

The source now keeps CUDA graphs enabled and explicitly selects the supported backend:

```text
--fp8-gemm-backend flashinfer_cutlass
```

The explicit flag alone did not hold. Inspection of the exact SGLang image source found a small-M optimization in `flashinfer_mxfp8_blockscaled_linear`: when CUTLASS is selected and `M <= 64`, it overrides CUTLASS with CuTeDSL whenever `_is_sm100_supported` is true. In this image that broad predicate also matches SM120, contradicting the adjacent comment that CuTeDSL is SM100-only and causing the crash.

The container startup now applies a narrow, fail-loud source correction before importing SGLang:

```python
if backend == "cutlass" and q_input.shape[0] <= 64 and not _is_sm120_supported:
```

The pinned image predates the later upstream `_is_sm100_supported` guard, so its exact original line is the broader `if backend == "cutlass" and q_input.shape[0] <= 64:` form. The startup assertion and replacement match that pinned source exactly.

The startup patch asserts that the exact original expression occurs once; if the pinned image changes unexpectedly, startup fails rather than applying an ambiguous patch. The command also uses `set -euo pipefail`, preventing a failed assertion from silently continuing into SGLang startup. The explicit `--fp8-gemm-backend flashinfer_cutlass` remains in place.

An earlier shell implementation did not stop after the guard assertion failed because the script lacked `set -e`. That attempt therefore started with unmodified source and reproduced the same CuTeDSL failure. The replacement is now one atomic Python assertion-and-write operation under fail-fast shell execution. The pinned upstream Python file contains Unicode punctuation in comments, so the operation explicitly reads and writes UTF-8; the first ASCII attempt failed loudly before launching SGLang, as intended.

## Outcome

The memory profile is corrected, MXFP8 GEMM is pinned to CUTLASS, and the image's incorrect SM120 small-M override is disabled narrowly. The InferenceService must be redeployed through the application API before capability and benchmark testing can continue.
