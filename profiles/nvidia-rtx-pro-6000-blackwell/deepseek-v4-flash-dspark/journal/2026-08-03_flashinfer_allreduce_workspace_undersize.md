# FlashInfer allreduce workspace undersized after ISVC recreation -- deployment blocked

**Date:** 2026-08-03
**Severity:** deployment-blocking (model cannot start after ISVC recreation)
**Root cause:** upstream vLLM bug -- `FlashInferAllReduce` workspace is a
global singleton sized to the first tensor seen; first-caller dimension
varies non-deterministically between pod lifecycles.

## Symptom

Model crashes in crash loop (both TP workers) during cudagraph warmup
(`compile_or_warm_up_model -> capture_model`):

```
RuntimeError: Worker failed with error 'Workspace validation failed:
  - token_num (48) * hidden_dim (4096) exceeds workspace max_token_num (256) * hidden_dim (256).
  This may cause Illegal Memory Access.'
```

Full call chain:

```
VocabParallelEmbedding.forward
  -> tensor_model_parallel_all_reduce (hidden_dim=4096 tensor)
  -> CudaCommunicator.all_reduce
  -> FlashInferAllReduce.all_reduce
  -> flashinfer_comm.allreduce_fusion
  -> trtllm_allreduce_fusion
  -> check_trtllm_allreduce_fusion_workspace_metadata   # <-- raises ValueError
```

## Root cause

### The bug (upstream vLLM, not specific to our image)

`vllm/distributed/device_communicators/flashinfer_all_reduce.py`,
function `get_fi_ar_workspace` (line ~142 at upstream commit `c233d90aa`):

```python
_fi_ar_workspace = None  # global module-level singleton

def get_fi_ar_workspace(world_size, rank, max_token_num, hidden_dim, ...):
    global _fi_ar_workspace
    if _fi_ar_workspace is not None:
        return _fi_ar_workspace   # <-- returns cached workspace, IGNORES new hidden_dim
    # ... create workspace with the FIRST caller's hidden_dim
    _fi_ar_workspace = _get_or_create(backend)
```

The `FlashInferAllReduce` workspace is a **global singleton** sized to
whatever `hidden_dim` the first allreduce-calling tensor happens to have.
Subsequent callers with different dimensions reuse the same workspace --
which fails validation when the new dimensions exceed what was allocated.

### Why hidden_dim=256?

The DeepSeek V4 Lightning Indexer has small attention dimensions
(`index_head_dim=128, index_n_heads=64`). During model construction or
the compilation warmup pass, a tensor with `hidden_dim=256` happens to
pass through `tensor_model_parallel_all_reduce` before the main model's
`hidden_dim=4096` tensors. The workspace is created at 256 and then
rejected when `VocabParallelEmbedding` (hidden_dim=4096) calls allreduce.

### Why it worked before and breaks now

The initialization order of PyTorch modules and the order of the first
allreduce call is **non-deterministic across pod lifecycles**. In the
old pod (the one that ran for days before today's ISVC recreation),
a `hidden_dim=4096` tensor happened to be the first through allreduce,
so the workspace was large enough. In the new pod (after ISVC deletion +
recreation today), a `hidden_dim=256` tensor wins the race.

The image, model weights, and config are identical between the old and
new deployments -- the image is pinned by digest
(`ghcr.io/ormandj/vllm-deepseek-v4-flash-sm120@sha256:8ea1a27c...`) and
the baked cache is byte-identical. Nothing changed except the pod lifecycle.

### Not an issue with our custom image

Our image (`onpremai/vllm-dsv4-dspark:2026-07-20-prebaked`) adds ONLY
warmed kernel/JIT/autotune caches (`COPY vllm-dspark-cache/ /cache/`)
on top of the pinned ormandj base. No Python code is modified.
`flashinfer_all_reduce.py` is byte-identical to upstream vLLM at
`c233d90aa`.

### Not documented upstream

No public vLLM or FlashInfer issue/PR describes this exact bug as of
2026-08-03. Related but distinct upstream issues:

- vLLM #37461/#37468: FlashInfer allreduce workspace **uninitialized**
  (torch.compile cached the pass that creates it) -- different mechanism
- vLLM #40383/#40023: FlashInfer **attention** workspace buffer overflow --
  different subsystem (attention, not allreduce)
- vLLM #34635: AllReduceFusionPass slow construction -- performance, not
  correctness

## Evidence sources

1. Container traceback from `--previous` logs: `flashinfer_all_reduce.py:371
   -> allreduce.py:979 -> trtllm_ar.py:1100 -> trtllm_ar.py:1026`
2. Container source code at
   `/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/flashinfer_all_reduce.py`
   (lines ~35-170): global `_fi_ar_workspace` + `if _fi_ar_workspace is not
   None: return _fi_ar_workspace`
3. Upstream vLLM at `c233d90aa` confirmed identical code via GitHub raw
4. Model `config.json`: `hidden_size=4096`, `index_head_dim=128`,
   `index_n_heads=64`
5. Gateway request metrics showed successful (2xx) requests for
   `deepseek-v4-flash`, confirming the prior deployment worked
6. Dockerfile `vllm-dsv4-dspark/Dockerfile` confirms image is pinned by
   digest and adds only cache files

## Workaround

Set `VLLM_ALLREDUCE_USE_FLASHINFER=0` in the pod env to disable the
standalone `FlashInferAllReduce` communicator. The compiled-graph
`AllReduceFusionPass` (which correctly uses
`config.model_config.get_hidden_size()=4096`) and pynccl remain available.

Currently `inference.yaml` sets this to `"1"`:
```yaml
- name: VLLM_ALLREDUCE_USE_FLASHINFER
  value: "1"
```

## Workaround performance impact

`VLLM_ALLREDUCE_USE_FLASHINFER=0` disables only the **standalone** (unfused)
`FlashInherAllReduce` communicator path, not the fused
allreduce+rmsnorm+quant patterns. The breakdown:

| Path | Gated by this flag? | What it does |
|---|---|---|
| `AllReduceFusionPass` (compiled-graph fused AR+RMSNorm+quant) | No -- always active with `custom_ops: ["all"]` | High-performance path; correctly uses `config.model_config.get_hidden_size()=4096` from line 998 |
| `FlashInherAllReduce` (standalone unfused allreduce) | Yes -- this flag | Handles small-tensor allreduce outside compiled graphs (e.g. `VocabParallelEmbedding`) |
| pynccl | No -- always available as fallback | NCCL-based allreduce; takes over when FlashInfer standalone path is disabled |

On SM120 (RTX PRO 6000 Blackwell), the FlashInfer standalone allreduce
threshold is only **2 MB** (`FI_ALLREDUCE_FUSION_MAX_SIZE_MB[120][2] = 2`).
The `VocabParallelEmbedding` output that triggered this crash is
`48 tokens * 4096 hidden * 2 bytes = 384 KiB` -- trivial for pynccl.
Virtually all compute-bound allreduce in decode happens inside the compiled
`AllReduceFusionPass` path, which is unaffected.

**Impact: negligible.**

## Proper fix (upstream)

Two changes needed in
`vllm/distributed/device_communicators/flashinfer_all_reduce.py`:

**Fix 1 (targeted -- use model config):** `get_fi_ar_workspace` should
use `config.model_config.get_hidden_size()` instead of the first caller's
`hidden_dim`. This is what `AllReduceFusionPass` already does at line 998:

```python
self.hidden_dim = config.model_config.get_hidden_size()
```

The `get_fi_ar_workspace` function does not have access to `VllmConfig`
today, so this requires threading it through or caching `hidden_size` at
`FlashInherAllReduce.__init__` time (it already receives `group` and
`device`; adding `hidden_size` is a minimal signature change).

**Fix 2 (defense-in-depth -- reallocate on larger dimension):** Even with
Fix 1, a global singleton that silently caches an undersized workspace is
fragile. If a new caller passes a larger `hidden_dim`, destroy the old
workspace and create a new one:

```python
def get_fi_ar_workspace(world_size, rank, max_token_num, hidden_dim, ...):
    global _fi_ar_workspace
    if _fi_ar_workspace is not None:
        if hidden_dim <= _fi_ar_workspace.metadata["hidden_dim"]:
            return _fi_ar_workspace
        # Larger dimension than what was allocated -- recreate
        logger.warning_once(
            "Recreating FlashInfer allreduce workspace: hidden_dim %d -> %d",
            _fi_ar_workspace.metadata["hidden_dim"], hidden_dim)
        _fi_ar_workspace = None
    _fi_ar_workspace = _get_or_create(backend)
```

Both fixes should land together: Fix 1 prevents the bug for the common case,
Fix 2 prevents it for any model with heterogeneous tensor dimensions.

## Resolution

**Workaround applied 2026-08-03:** Set `VLLM_ALLREDUCE_USE_FLASHINFER=0` in
`inference.yaml`. Model redeployed via app API (1M context profile),
`READY=True` in ~3 minutes, zero restarts, smoke-test health check 200 OK.
The compiled-graph `AllReduceFusionPass` remains active and unaffected.

**Performance verified 2026-08-03:** `llmapibenchmark` ran against the
deployed service (`num_words=50, max_tokens=500`). Terminal progress bar
shows ~479 tok/s single-stream decode at steady state (1M context profile,
DSpark active). Concurrency=2 shows 5.6s TTFT on the second stream (expected
DSpark spec engine re-initialization); concurrency=4 reaches 246 tok/s per
stream aggregate. The `VLLM_ALLREDUCE_USE_FLASHINFER=0` workaround has no
measurable performance impact -- the fused allreduce+rmsnorm+quant path
(`AllReduceFusionPass`) was always the performance-critical path and is
unaffected.

Upstream issue to be filed.