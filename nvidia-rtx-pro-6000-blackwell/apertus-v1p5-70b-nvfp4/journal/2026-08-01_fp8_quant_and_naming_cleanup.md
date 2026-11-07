# 2026-08-01 -- FP8 quant of swiss-ai/Apertus-v1.5-70B + naming cleanup

## Summary

Created a full FP8_DYNAMIC quant of the official `swiss-ai/Apertus-v1.5-70B`
release for quality comparison against the NVFP4 three-tier checkpoint.
Renamed the Docker image from `vllm-apertus-infomaniak` to `vllm-apertus-1p5`
(no Infomaniak in the name for the swiss-ai release).

## FP8 quantization

Script: `quantize_apertus_v15_70b_fp8.py` (single-pass FP8_DYNAMY on all
linears except lm_head and embeddings)

- Input: `/persistent/models/swiss-ai/Apertus-v1.5-70B/` (BF16, 135 GiB)
- Output: `/persistent/models/swiss-ai/Apertus-v1.5-70B-FP8/` (71 GiB)
- Duration: 4 min
- Scheme: FP8_DYNAMIC, llm-compressor 0.12.0, single pass
- Ignores: lm_head, model.language_model.embed_tokens,
  re:.*vision_tokenizer.*, re:.*audio_tokenizer.*

### Config patching required

Same as NVFP4: `fix_three_tier_quant_config.py` handles the compressed-tensors
config surgery (target groups, ignore list). Additionally, the HF-format FP8
config uses `text_config` sub-object + `Apertus1p5ForConditionalGeneration`
architecture. Manual changes applied:

1. Flatten `text_config` fields to root level
2. Change `model_type` from `apertus1p5` to `apertus`
3. Change `architectures` from `["Apertus1p5ForConditionalGeneration"]` to
   `["ApertusForConditionalGeneration"]`
4. Add `audio_config` and `vision_config` dicts with `token_offset`
5. Add `post_norm`, `output_vocab_size`, `rope_parameters`, etc. (fields the
   vLLM Apertus model code expects)
6. Symlink `emu35_vison_tokenizer.safetensors` and
   `wavtokenizer_large_unify_600_24k.safetensors` -> the actual shard files
   (required by `apertus_mm.py` model registration)

## Serve test (FP8 via ISVC)

Deployed as `apertus-v15-70b-fp8` using the
`vllm-apertus-1p5` image. Key serving args for the FP8 checkpoint:

```yaml
--limit-mm-per-prompt '{"image": 0, "audio": 0}'  # multimodal disabled for profiling
--max-model-len 4096                                # KV cache fits in remaining VRAM
--kv-cache-dtype fp8_e4m3                           # compress KV cache
--gpu-memory-utilization 0.95
```

Without `--limit-mm-per-prompt`, the vLLM profile run executes dummy vision
encoder forward passes which OOM (71 GiB weights + vision encoder almost fills
the 96 GiB card). For text-only benchmarks, disabling multimodal is safe.

Without `--max-model-len`, vLLM tries to allocate KV cache for
`max_position_embeddings=262144` which requires 80 GiB -- impossible after
loading 71 GiB weights (only ~14 GiB remaining).

### Issue: zombie EngineCore holds VRAM

When lm-eval's vLLM backend crashes, the `VLLM::EngineCore` subprocess stays
alive holding all VRAM. Even `nvidia-smi` on the host shows 0 MiB (because
the process is inside a K8s container cgroup), but from inside the pod it's
77 GiB allocated. New lm-eval attempts fail with "16.84 GiB free" until the
orphan EngineCore is killed:

```bash
kubectl exec -n <namespace> <pod> -- kill -9 <EngineCore-PID>
```

This happened because the previous lm-eval-runner pod was deleted without
killing the EngineCore subprocess first. On pod deletion, the main process
died but the child (EngineCore) got reparented to PID 1 and kept the GPU.

### Image rename: vllm-apertus-infomaniak -> vllm-apertus-1p5

Renamed across all locations:

| File | Change |
|------|--------|
| `build.sh` | `IMAGE="onpremai/vllm-apertus-1p5:\${TAG}"` |
| `inference.yaml` | `image: onpremai/vllm-apertus-1p5:apertus15-79efecfb-p20` |
| Docker tag | `docker tag ... vllm-apertus-1p5:apertus15-79efecfb-p20` |
| Registry push | `docker push <registry>/onprem-ai/vllm-apertus-1p5:...` |

Note: The first registry push from the host did NOT make the image
available to K8s. kubelet pulls via `imagePullSecrets`, which use the
registry's pull credentials. The image was eventually available after
pushing with the pull credentials (docker stored cached creds from a prior
`docker login`).

The image pull secret was not attached to the `default` ServiceAccount in
the namespace. Fixed by:

```bash
kubectl patch sa -n <namespace> default -p '{"imagePullSecrets": [{"name": "<pull-secret>"}]}'
```

## FP8 serve test results

Basic chat completion works (FP8 loads and serves):

```
User: What is 2+2?
Response: 2 + 2 = 4.

User: If a train leaves at 3pm and arrives at 5:30pm, how long is the journey?
Response: The train journey lasts **2.5 hours**.
  Calculation:
  - From 3:00 PM to 5:00 PM is 2 hours,
  - Plus 30 more minutes (until 5:30 PM),
  - Total: **2 hours and 30 minutes** (or 2.5 hours).
```

Tool calling was NOT configured on the FP8 ISVC (no `--tool-call-parser` or
`--enable-auto-tool-choice` args) -- this was intentional since this is a test
deployment for benchmarking, not production.

## MMLU benchmark (100 samples, 5-shot)

Used `lm-eval --model vllm` running inside a pod with the `vllm-apertus-1p5`
image. Both checkpoints tested with identical args:

```
--model_args pretrained=/model,dtype=auto,trust_remote_code=True,
  tensor_parallel_size=1,gpu_memory_utilization=0.95,max_model_len=4096,
  limit_mm_per_prompt='{"image":0,"audio":0}',kv_cache_dtype=fp8_e4m3
--tasks mmlu --num_fewshot 5 --limit 100
```

`--limit-mm-per-prompt {"image":0,"audio":0}` was required to prevent the
vLLM profile_run from executing dummy vision encoder forward passes (which
OOM'd on FP8 with only ~14 GiB remaining after loading 71 GiB weights).
`--max-model-len 4096` was required because the model config declares
`max_position_embeddings=262144`, and vLLM tries to allocate 80 GiB KV cache
for full context length.

### FP8 results (71 GiB checkpoint)

```
| Category        | acc     | stderr   |
|-----------------|---------|----------|
| MMLU overall    | 0.7205 | +/-0.0057 |
| - humanities    | 0.7646 | +/-0.0112 |
| - other         | 0.7269 | +/-0.0119 |
| - social sci.   | 0.8158 | +/-0.0110 |
| - stem          | 0.6258 | +/-0.0107 |
```

Duration: ~1.5 h (6.34 it/s avg). Process became defunct at log write time
but all results were saved correctly. Required manual `kill -9` of orphaned
EngineCore between runs (see zombie issue above).

### NVFP4 results (48 GiB checkpoint)

```
| Category        | acc     | stderr   |
|-----------------|---------|----------|
| MMLU overall    | 0.7126 | +/-0.0057 |
| - humanities    | 0.7615 | +/-0.0114 |
| - other         | 0.7123 | +/-0.0120 |
| - social sci.   | 0.8017 | +/-0.0111 |
| - stem          | 0.6232 | +/-0.0107 |
```

Duration: ~2.5 h (3.31 it/s avg). NVFP4 is slower per-request because
mixed-precision decompression has overhead vs pure FP8 compute.

### Comparison

| Metric        | FP8      | NVFP4    | Delta     | Relative |
|---------------|----------|----------|-----------|----------|
| MMLU overall  | 0.7205   | 0.7126   | -0.0079   | -1.10%   |
| humanities    | 0.7646   | 0.7615   | -0.0031   | -0.41%   |
| other         | 0.7269   | 0.7123   | -0.0146   | -2.01%   |
| social sci.   | 0.8158   | 0.8017   | -0.0141   | -1.73%   |
| stem          | 0.6258   | 0.6232   | -0.0026   | -0.42%   |

Both within 1 standard error of each other (stderr ~0.0057 for the aggregate).
The NVFP4 overall degradation is **~0.8 points** (1.1% relative), which is
excellent for a 2.8x compression ratio (135 GiB BF16 -> 48 GiB NVFP4).

FP8 at 71 GiB is 2.0x compression versus BF16; NVFP4 at 48 GiB is 2.8x.
NVFP4 saves 23 GiB VRAM (32% less than FP8) at the cost of 0.8 MMLU points.

Decode throughput difference was measured separately on the RC
(2026-07-07_two_pass_success_and_benchmarks.md):
- NVFP4: 30.7 TPS (decode) / 2202 TPS (prefill)
- FP8:   20.0 TPS (decode)
NVFP4 is ~50% faster on decode because it is memory-bandwidth bound and
weights are 33% smaller.

### Decision: NVFP4 is the publication candidate

The 1.1% MMLU loss vs FP8 is negligible for most use cases. The 32% VRAM
savings and 50% faster decode make NVFP4 the superior choice for publishing.

## Throughput benchmarks (2026-08-02, llmapibenchmark on vLLM)

Measured using the official `llmapibenchmark` tool (onpremai/llmapibenchmark:latest)
which generates a NEW random prompt per request (avoids KV cache hits). The
benchmark creates a K8s Job that hits the predictor service endpoint.

### Root cause of initial 500 error

The benchmark API calls `_fetch_model_from_endpoint()` which queries
`GET /v1/models` on the provided `base_url`. The API was timing out because
I manually deployed the inference service via `kubectl apply -f` which exposed
port 8080 in the container, but the KServe ClusterIP service maps
port 80 -> targetPort 8080. The benchmark was using port 8080 in the base_url
but should use port 80. Fixed by deleting the manual deployment and redeploying
through the regular catalog deployment.

Additionally, the `llmapibenchmark` Docker image was not in the local registry.
Pulled from Docker Hub (`docker pull onpremai/llmapibenchmark:latest`) and
pushed to the local registry.

### Standard Suite (concurrency=1, random input)

| Test | tokens in | tokens out | gen_tps | prep_tps | TTFT |
|------|-----------|------------|---------|----------|------|
| Decode (50w, 500t) | 116 | 500 | 31.36 | 2570 | 0.1s |
| Prefill (1500w, 100t) | 1493 | 100 | 17.54 | 2335 | 2.5s |
| Balanced (500w, 500t) | 533 | 500 | 30.08 | 2502 | 0.8s |
| Long ctx (3000w, 100t) | 3009 | 100 | 11.61 | 2173 | 5.4s |

Decode TPS ~31 for single user. Prefill TPS ~2500 flat across all workloads.

### Scaling (num_words=200, max_tokens=200)

| concurrency | gen_tps | prep_tps | TTFT |
|-------------|---------|----------|------|
| 1 | 30.26 | 2577 | 0.3s |
| 2 | 57.49 | 2542 | 0.7s |
| 4 | 104.43 | 2539 | 1.3s |
| 8 | 175.76 | 2543 | 2.7s |
| **10** | **202.58** | 2527 | 3.3s |
| **20** | **294.12** | 2506 | 6.8s |
| 30 | 237.78 | 1347 | 18.8s |
| 40 | 272.86 | 1506 | 22.6s |
| 50 | 248.70 | 1247 | 33.9s |
| 60 | 266.95 | 1332 | 38.3s |

**Peak throughput at concurrency=20: 294 t/s decode.**

Prefill TPS stays flat at ~2500 up to concurrency=20, then drops by half
(KV cache contention, model memory bandwidth saturated). TTFT climbs
linearly with concurrency (0.3s at 1 -> 6.8s at 20 -> 38s at 60).

### Catalog update

`inference.yaml` performance section updated:
- Single user: 31.4 tps_decode, 2500 tps_prefill
- concurrency=4: 104 tps_decode
- concurrency=8: 176 tps_decode
- concurrency=20 (peak): 294 tps_decode

## Next steps

- (done) MMLU comparison complete -- NVFP4 wins on VRAM/throughput/weight
- (done) Throughput benchmarks complete -- peak 294 t/s at concurrency=20
- Decide on publication: HF checkpoint + Dockerfile + CI
