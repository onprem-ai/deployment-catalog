# deployment-catalog

Declarative, reproducible AI model deployments: every image, weight revision and engine
setting pinned, deployed by the [onprem.ai](https://onprem.ai) platform.

| GPU | Deploy profiles | Directory |
|---|---|---|
| NVIDIA RTX PRO 6000 Blackwell | 62 | [`profiles/nvidia-rtx-pro-6000-blackwell/`](profiles/nvidia-rtx-pro-6000-blackwell/) |

Each directory with an `ai-deploy.yaml` is one deploy profile: the model, the engine and
the GPUs it is qualified for, with its `compose.yaml` and the files its containers read.
`journal/` holds the dated notes of its bring-up.

The profiles are read by the onprem.ai control plane. Add this repository under
**Configuration → Catalogs** with a branch, tag or commit, and optionally a subdirectory
such as `profiles/nvidia-rtx-pro-6000-blackwell`.
