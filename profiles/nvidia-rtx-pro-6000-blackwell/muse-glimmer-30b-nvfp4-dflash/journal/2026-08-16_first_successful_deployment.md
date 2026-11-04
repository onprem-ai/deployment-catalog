# 2026-08-16 - First Successful Deployment and Validation

**Goal:** Bring up Muse Glimmer NVFP4 plus DFlash on RTX PRO 6000 Blackwell, verify capabilities through official application APIs, and measure initial performance.

## Working Runtime Correction

The pinned SGLang Muse Glimmer image contains a small-M optimization that overrides an explicitly selected FlashInfer CUTLASS MXFP8 GEMM backend with CuTeDSL. FlashInfer supports MXFP8 on SM120 only through CUTLASS; CuTeDSL rejects compute capability 120.

Container startup now atomically patches that guard to exclude `_is_sm120_supported`, under `set -euo pipefail`, and launches with:

```text
--fp8-gemm-backend flashinfer_cutlass
```

This retained prefill, target verification, and draft verification CUDA graphs. All graph capture completed successfully.

## Successful Startup

The deployment became Ready with zero restarts. Relevant measurements:

- Target mixed NVFP4/MXFP8 weights: 19.21 GiB VRAM.
- FP8-quantized DFlash assistant: 3.03 GiB VRAM.
- Configured memory profile: 75 GiB, resolving to `mem-fraction-static` about 0.78.
- Runtime GPU usage after startup: 79,055 MiB.
- Remaining GPU memory: 18,195 MiB.
- Maximum total token pool: 1,678,100 tokens.
- Context length: 131,072 tokens.
- DFlash block size: 5.
- Service: `rtx6k--muse-glimmer-30b-nvfp4-dflash`.

The 75 GiB value is the first successful profile, not yet a binary-searched minimum.

## Capability API Results

Tests ran through `POST /api/capability-test` against the internal OpenAI-compatible service endpoint.

### Chat

- 6/6 passed.
- System, developer, and ordinary user messages passed in streaming and non-streaming modes.
- The test runner counts a response with reasoning content as successful even when a short output budget prevents a final content message. This behavior is visible in the returned test details and should be considered when interpreting strict instruction-following results.

### Tool Use

- 10/12 passed.
- Single tool selection: passed in streaming and non-streaming modes.
- Multi-tool selection: passed in both modes.
- Tool-result round trip: passed in both modes.
- No-tool restraint: passed in both modes.
- Narration followed by a tool call: passed in both modes.
- Parallel tool calls: failed in both modes; only one call was emitted where the test expected at least two calls in one response.

## Benchmark API Results

Tests ran through `POST /api/inference-services/benchmark` at concurrency 1.

### Decode-focused

- Random input words: 50.
- Prompt tokens: 112.
- Requested/generated output tokens: 500/500.
- Generation speed: 181.18 tokens/s.
- Success rate: 1.0.
- Duration: 2.76 seconds.

### Prefill-focused

- Random input words: 1,500.
- Prompt tokens: 1,407.
- Requested/generated output tokens: 100/100.
- Generation speed: 46.91 tokens/s.
- Success rate: 1.0.
- Duration: 2.13 seconds.

The benchmark API returned negative `prompt_throughput` values for both runs, so no prefill TPS is recorded in the catalog. Negative throughput is invalid measurement output and must not be presented as model performance.

## Outcome

Muse Glimmer now serves successfully with both mixed NVFP4/MXFP8 target weights and DFlash speculative decoding on the RTX PRO 6000. Initial chat and tool behavior is functional, parallel tool calls are not confirmed, and measured single-user decode speed is 181.18 tokens/s. Resource-floor binary search, DFlash acceptance-length collection, valid prefill measurement, and broader reasoning evaluation remain pending.
