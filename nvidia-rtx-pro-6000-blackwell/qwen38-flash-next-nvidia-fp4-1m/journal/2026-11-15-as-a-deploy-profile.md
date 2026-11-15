# 2026-11-15 - As a Deploy Profile: Deploy, Answers and Benchmark Through the Control Plane

**Goal:** Turn the run in `2026-11-15-1m-on-one-gpu-reproduction.md` into this deploy profile,
deploy it through the control plane, check its answers and run the full replica benchmark.

## From the manual run to the profile

`compose.yaml` holds the manual run's command and environment, with these differences, each
for a reason:

| Manual run | Profile | Why |
|---|---|---|
| `--port 8000` | `--port 8080`, `expose: "8080"` | the profile's API port |
| `--served-model-name qwen38-flash-next` | `qwen3.8-flash-next-1m` | the profile's routing name |
| `--gpu-memory-utilization 0.975` | written by the deploy from the engine booking, 93 GiB (0.9729 of the GPU) | a profile states its booking in bytes |
| `--ipc host`, `--ulimit memlock=-1`, `--ulimit stack=...` | not set | a profile may not set host namespaces or ulimits (compose subset); the replica still started and served |
| `-v qfn-cache:/root/.cache` | no cache volume | kernels compile at every start: about 12 minutes from loading to ready |
| `--shm-size 32g` | `shm_size: 32gb` | as written |

Hardware booking 95 GiB per GPU (the GPU reports 95.6 GiB), host memory 48 GiB with a limit of
48G and a reservation of 24G: `VLLM_PLE_CPU_OFFLOAD=1` keeps the PLE n-gram table in host memory.

Weights: `provider: huggingface`, `nvidia/Qwen3.8-Flash-Next-FP4-Local` at
`b9ce18a6887e18c089fdea8509a33e63429a3300`, 31 files (all but `.gitattributes` and
`README.md`), 95.2 GiB. The inference node downloaded them itself in about 45 minutes
(32 MiB/s); the copy from the manual run was not reused.

## Deploy

Development control plane, drafts catalog, inference node `ain-4euk3`, GPU
`GPU-b2a98efd-9e55-6afa-fb6d-3374318d08ca`. The deploy warned that 0.6 GiB of the GPU was
used by other programs while the replica books 95 GiB; it started regardless.

Replica `01c613d1-c954-4159-b88f-adc172bc3055`: weights ready 11:25, ready 11:37 UTC.
Start-up lines: `Model loading took 71.71 GiB`, CUDA graphs 0.39 + 0.26 GiB,
`Available KV cache memory: 16.12 GiB`, `GPU KV cache size: 1,114,309 tokens, Maximum
concurrency for 1,048,576 tokens per request: 1.06x`.

## Answers (through the API gateway)

"17 * 23": 391 with thinking (139 reasoning characters) and without. Two parallel tool calls
(Paris, Tokyo): `finish_reason: tool_calls`, both cities.

## Benchmark

Job `1cff4550-ecf1-4d44-a283-03f89ded266d`, started 11:38:26 UTC, all three workloads,
settings as committed. Result in the next entry.
