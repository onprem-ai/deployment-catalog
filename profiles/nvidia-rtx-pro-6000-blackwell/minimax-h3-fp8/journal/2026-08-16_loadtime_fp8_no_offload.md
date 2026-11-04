# 2026-08-16 -- MiniMax H3: Official diffusers load-time FP8 quant (single card, NO offload)

**Goal:** Serve official MiniMax H3 via the diffusers Modular pipeline, quantized to FP8 AT
LOAD, with ALL components resident on a single RTX PRO 6000 Blackwell (95.6 GiB usable) and
**NO offloading** (hard project requirement).

## Why THIS card (pivot decision 2026-08-16)

This card replaces the original plan to serve via the unsloth pre-quantized .pt ONLY for the
"single-card diffusers" niche. Reason for the pivot (research, not guessing):

- The unsloth `MiniMax-H3-FP8.pt` loader is NOT documented in any citable source: the unsloth
  README says "a loader picks between them by filename" but never shows the code, there is no
  example script or docs page in the repo, and the model repo ships no loader (only `.pt` files,
  `text_encoders/`, README). Per the "NEVER GUESS PARAMETERS" rule we will not reverse-engineer/
  guess the load path.
- The OFFICIAL diffusers path is fully documented and citable: `ModularPipeline` +
  `MiniMaxH3Transformer3DModel.from_pretrained(..., quantization_config=TorchAoConfig(...))`
  (diffusers MiniMax-H3 docs, "Memory" section; FLUX.1-dev-torchao-fp8 example).
- CORRECTION after component-level accounting: the unsloth ~36.97 GB render peak belongs to its
  pruned denoiser + quantized encoder artifact and MUST NOT be transferred to official load-time
  FP8. Official source arithmetic gives roughly 31 GiB transformer + 33 GiB text encoder +
  9.7 GiB video VAE + 0.6 GiB audio VAE before activations. The official path may fit 95.6 GiB,
  but is tight and must reserve the full card until measured.
- unsloth `.pt` remains a future option (separate card `minimax-h3-fp8-unsloth`, still
  experimental, kept for later). This card is the RELIABLE first path.

## NO OFFLOAD (hard invariant; startup design corrected after hardware check)

Actual node resources checked 2026-08-16:
- system RAM: 60 GiB total, 38 GiB available, NO swap
- each RTX PRO 6000: 97,887 MiB total (~95.6 GiB)

Therefore the docs' `low_cpu_mem_usage=False` example and any claimed 100 GiB pod memory limit are
impossible on this node. The server instead loads each sharded component with
`low_cpu_mem_usage=True` and explicit `device_map="cuda:0"`, quantizing each eligible Linear
weight at its CUDA target while loading. This avoids retaining the complete bf16 model in host RAM.

Runtime invariants:
- no `ComponentsManager`
- no `enable_model_cpu_offload`, `enable_sequential_cpu_offload`, or group offload
- no `device_map="auto"` (it could silently place weights on CPU)
- startup fails if any transformer, text encoder, video VAE, or audio VAE parameter/buffer is not
  on CUDA

This is supported by diffusers' sharded `low_cpu_mem_usage=True` loader and TorchAO quantizer:
`device_map="cuda"` is documented, and each eligible Linear weight is moved to its target device
then quantized. It still requires hardware validation with one GPU free.

## Version awareness (citable, decided 2026-08-16)

- **diffusers version:** MiniMax-H3 PR #14355 merged into diffusers main on 2026-08-05. The old
  `minimax-h3` branch referenced by MiniMax's requirements no longer exists. The serving image pins
  exact diffusers commit `d5baa4fb548294f47dbca49890abd4b291204c60` (2026-08-15), which contains
  the merged H3 integration; it never installs a floating branch.
  - Ref: https://github.com/huggingface/diffusers/pull/14355
  - Ref: https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3
- **official model revision:** `42ed227ee7df40d41602854ae760620d6eb651fe`, pinned in the card.
- **quantization dtype choice = FP8** (`Float8WeightOnlyConfig`), consistent with the earlier
  FP8-vs-INT8 decision (see minimax-h3-fp8-unsloth journal): FP8 needs no `torch.compile`,
  negligible quality delta, simpler/more robust than INT8. FP8 at-load is supported by
  `TorchAoConfig` (diffusers/FLUX.1-dev-torchao-fp8 example is citable).
  - Ref: https://huggingface.co/diffusers/FLUX.1-dev-torchao-fp8
- **transformers:** Qwen3-VL conditioner needs `Qwen3VLForConditionalGeneration`, so
  transformers>=4.45 (MiniMax requirements).

## Load path (target, from diffusers docs -- to be implemented in server.py)

```python
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Float8WeightOnlyConfig

pipe = ModularPipeline.from_pretrained(MODEL_DIR)   # local dir, no fetch
pipe.update_components(
    transformer=MiniMaxH3Transformer3DModel.from_pretrained(
        MODEL_DIR, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(Float8WeightOnlyConfig(),
            modules_to_not_convert=[
                "proj_in","audio_proj_in","context_embedder","time_embedder","time_proj",
                "token_refiner","norm_out","proj_out","audio_proj_out"]),
        device_map="cuda:0", low_cpu_mem_usage=True, local_files_only=True),
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(Float8WeightOnlyConfig(version=2)),
        device_map="cuda:0", low_cpu_mem_usage=True, local_files_only=True),
)
pipe.load_components(
    names=["tokenizer", "processor", "scheduler", "audio_scheduler"],
    pretrained_model_name_or_path=MODEL_DIR, local_files_only=True)
pipe.load_components(
    names=["vae", "audio_vae"], pretrained_model_name_or_path=MODEL_DIR,
    dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True,
    local_files_only=True)
pipe.transformer.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
# NO group offload / NO ComponentsManager CPU offload -- all model tensors must be on CUDA.
```

NOTE: the exact `modules_to_not_convert` list comes from the diffusers docs' INT8 example; the FP8
variant may allow a simpler config. Validate the exact args against the docs/installed diffusers
before finalising server.py (open item).

## Download (official API, started 2026-08-16)

Triggered via `POST /api/model-catalog/download?config_id=rtx6k--minimax-h3-fp8`
(before adding any inference_service). Download id `hf:MiniMaxAI--MiniMax-H3--model_index`,
dir `/persistent/models/MiniMaxAI--MiniMax-H3--model_index`. 61 files, exactly
144,016,413,562 bytes / 134.13 GiB total (transformer + text encoder + VAEs + configs). Disk free
at start: ~257 GB. The FL2VA/transformer shards already downloaded for the
sglang card are NOT reusable (different shard naming -- FL2VA/transformer uses model-00001..13,
the diffusers root transformer uses diffusion_pytorch_model-00001..14).

## Download completion and integrity verification (2026-08-16)

The official API download completed: all 61 declared files under
`/persistent/models/MiniMaxAI--MiniMax-H3--model_index`, 135 GiB by `du`. App verification
examined all 61 files with zero failures; no `.aria2` control files remain. Live catalog status:
`downloaded=true`, `downloading=false`.

Integrity details:
- every large LFS transformer/text-encoder/VAE shard matches the resolver's declared byte size
- regular Git-managed JSON/tokenizer/index files differ from the resolver's stored `size` because
  it persisted Git pointer/API metadata lengths rather than served content lengths
- representative pinned-revision verification used Git object hashes: local `model_index.json`
  hash `454be19164878173a962c39c81633e4377880fca` and local scheduler config hash
  `1d44f95b955a648ad4fab455e4fe11053888c30b` exactly match Hugging Face ETags at model commit
  `42ed227ee7df40d41602854ae760620d6eb651fe`

The download started minutes before the source revision was added to the card, so its persisted
state says `revision=None` and URLs contain `resolve/main`. Hugging Face `main` remained at the
exact pinned commit throughout the download (checked before, during, and after), and the ETag/blob
verification above confirms pinned content. The card itself now pins the exact revision for all
future downloads.

## Serving image implementation (2026-08-16)

Created `catalog/rtx6k/_docker/minimax-h3-fp8-server/{Dockerfile,server.py}` following the
repo's FastAPI media-server pattern. Stack pins:
- PyTorch 2.11.0 + CUDA 12.8
- torchvision 0.26.0 (official PyTorch 2.11 pair)
- torchao 0.17.0 (compatibility matrix targets torch 2.11)
- transformers 5.12.0
- diffusers exact commit `d5baa4fb548294f47dbca49890abd4b291204c60`

The server has explicit request parameters, validates H3 canvas/frame constraints, serializes
requests with one lock, muxes jointly generated video+audio to MP4, and fails startup if model
parameters/buffers remain on CPU. Unit tests assert direct CUDA + low-host-memory loader arguments,
absence of every offload mechanism, media muxing, dependency pins, and request validation.

Local image build **completed successfully** under
`onpremai/minimax-h3-fp8-server:torch211-cu128` using an isolated anonymous Docker config for
public base-image pulls. Local image ID after the final cached rebuild (including inference mode):
`sha256:a596d463f83987ab8e667d4bb2749ccd3eadf563b6eade7b6d639342423ade7a` (~11.48 GB).

Smoke validation inside the GPU-enabled image:
- imports: torch 2.11.0+cu128, torchao 0.17.0, transformers 5.12.0, diffusers 0.40.0.dev0
- MiniMaxH3Transformer3DModel, ModularPipeline, Qwen3VL and TorchAoConfig imports passed
- an actual CUDA Linear was quantized to TorchAO `Float8Tensor` v2 and completed a BF16-input
  matmul on `cuda:0`

`TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1` skips optional prebuilt MXFP8/SM90 extensions that have a
Python ABI/architecture mismatch. `Float8WeightOnlyConfig` v2 uses the Python tensor-subclass path,
confirmed by the CUDA smoke test. diffusers still logs that serialized pre-quantized checkpoint
safe globals are unavailable; this card quantizes official bf16 weights at load and does not load
serialized TorchAO checkpoints.

Docker Hub answered `401 Unauthorized` on push, so PUSHING to `onprem-ai` and obtaining a
registry digest remain blocked until the Docker Hub credentials are refreshed.

Hardware validation also requires one GPU to be freed through the app lifecycle: currently
another deployed model occupies both RTX PRO 6000 cards (~96 GiB each).

## References (fetched 2026-08-16)

- diffusers MiniMax-H3 pipeline docs (checkpoint layout, two schedulers, Memory/no-offload) --
  https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3
- MiniMax H3 requirements.txt (diffusers>=0.32.2, install `minimax-h3` branch; transformers>=4.45)
  -- https://github.com/MiniMax-AI/MiniMax-H3/blob/main/requirements.txt
- diffusers ModularPipeline docs (from_pretrained + load_components + quantization_config) --
  https://huggingface.co/docs/diffusers/main/en/modular_diffusers/modular_pipeline
- diffusers torchao quant guide (TorchAoConfig / AOBaseConfig, e.g. Float8WeightOnlyConfig) --
  https://huggingface.co/docs/diffusers/main/en/quantization/torchao
- diffusers/FLUX.1-dev-torchao-fp8 (FP8 at-load example) -- https://huggingface.co/diffusers/FLUX.1-dev-torchao-fp8
- unsloth MiniMax-H3-FP8 (FP8 render-peak ~37 GB measurement; no citable loader)
  -- https://huggingface.co/unsloth/MiniMax-H3-FP8

## Notebook / Open Items
- [x] Validate TorchAO v2 FP8 config, selective exclusions, sharded low-CPU-memory loader,
      explicit CUDA device map, and no-offload invariants in unit tests.
- [x] Build and smoke-test local `onpremai/minimax-h3-fp8-server:torch211-cu128` image.
- [ ] Refresh Docker Hub credentials, push image, capture @sha256 digest, add inference_service.
- [ ] Free one GPU through the official app deployment lifecycle, then measure startup host RAM,
      resident/peak VRAM, and generation wall-clock. Reject the path if all model tensors plus
      activations do not fit without offload.
- [ ] License territorial check for every node's physical location before deploy.
