# 2026-08-16 -- First deployment and T2VA workflow correction

## First successful startup

The image was pushed to a local registry and
deployed through `POST /api/model-catalog/deploy` on GPU 0.

Initial image digest:
`sha256:93aa4a8691490ddc7e8763355db391e05eb0339f73c57d15ae34cf2106de7fb5`.

Startup succeeded with NO offload:
- startup time: ~33 seconds after container start
- allocated/peak VRAM reported by torch: 75.05 GiB
- nvidia-smi after load: ~76.0 GiB used, ~19 GiB free
- pod RAM current/peak after load: ~23.55 / 23.69 GiB
- GPU 1 remained unused
- readiness and `/health` succeeded; `offload_enabled=false`

## First generation failure (error path visible)

Request: text only, 960x544, 124 frames, 8 sigma points, seed 42.

The API returned HTTP 500 and the full traceback was visible in pod logs:

```text
ValueError: torch.cat(): expected a non-empty list of Tensors
  diffusers/modular_pipelines/minimax_h3/before_denoise.py:955
  block_state.condition_rows = torch.cat(packed)
```

Root cause: the server instantiated the explicit `fl2va` workflow even though its request model
only accepts text. Pinned diffusers source documents:
- `t2va` requires `prompt`
- `fl2va` requires `prompt` plus `image` or `last_image`
- text-only requests skip keyframe blocks and use `t2va`

References:
- `docs/source/en/api/pipelines/minimax_h3.md` at diffusers commit
  `d5baa4fb548294f47dbca49890abd4b291204c60`
- `src/diffusers/modular_pipelines/minimax_h3/modular_blocks_minimax_h3.py` at the same commit

## TDD fix

A unit assertion was added first requiring:

```python
ModularPipeline.from_pretrained(model_dir, workflow="t2va", local_files_only=True)
```

It failed against the deployed `workflow="fl2va"` implementation, then passed after the one-line
fix. All five focused server tests pass.

Corrected image:
- local image ID: `sha256:53ac46626d203b86322093039f09af08454e958c852db71a9c3939ae9b2d6bc4`
- model-registry digest:
  `sha256:3945deb439a297e38265056480f7abd31f000d484c7fd5292b53bfc4cdfb607e`
- tag: `torch211-cu128-t2va`

The card now pins the corrected registry digest. First/last-frame request fields remain deferred;
the public server API honestly supports T2VA only for this first deployable version.
