<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/onprem_logo_text_white.svg">
    <img alt="onprem.ai" src="docs/onprem_logo_text_black.svg" width="320">
  </picture>
</p>

<p align="center">
  <b>Deployment catalog</b>
</p>

<p align="center">
  Deploy AI with confidence: hardware-specific, declarative AI deployment recipes, optimized
  for production performance and reliability.
</p>

<p align="center">
  Reproducible in every step: every image, weight revision and engine setting pinned,
  deployed independently or in 1 click on the <a href="https://onprem.ai">onprem.ai</a> platform.
</p>

## Layout

Each top-level directory holds the deploy profiles for one GPU, for example
`nvidia-rtx-pro-6000-blackwell/`. Each directory with an `ai-deploy.yaml` is one deploy
profile: the model, the engine and the GPUs it is qualified for, with its `compose.yaml`
and the files its containers read. `journal/` holds the dated notes of its bring-up.

## Usage

The profiles are read by the onprem.ai control plane. Add this repository under
**Configuration → Catalogs** with a branch, tag or commit, and optionally a subdirectory
such as `nvidia-rtx-pro-6000-blackwell`.
