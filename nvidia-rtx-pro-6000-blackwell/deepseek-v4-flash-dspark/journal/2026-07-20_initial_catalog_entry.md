# Initial catalog entry -- from 3-day manual bring-up

**Date:** 2026-07-20
**Basis:** manual deployment validated 2026-07-19/20 on this node (tmux +
docker), promoted to catalog entry. The bring-up covered:

- IOMMU NCCL hang, host RAM exhaustion, FlashInfer autotuner bug,
  empty-prefill bug
- thermal validation at 500 W, fan2go GPU-temp fan control
- catalog entry decisions incl. image build-vs-mirror
- fan2go + power-limit provisioning

## Measured numbers (2026-07-19/20, this node)

- 219 tok/s single-stream decode (831 tok in 3.8 s), DSpark acceptance 51.7%
- Weights 79.3 GiB/GPU (TP=2), KV cache 8.65 GiB at util 0.944
- 256k ctx / 8 seqs validated end-to-end; 1M ctx boots (KV 1,072,998 tokens)
  but not e2e validated
- Thermal: 2x500 W sustained -> GPU plateau 77-78C closed chassis with
  fan2go curve; stock 600 W saturates the fans (keep the 500 W cap)

## First-of-kind aspects

- First multi-GPU (TP=2) entry: nvidia.com/gpu: "2", hostIPC already a
  catalog convention
- First entry with a mandatory pre-seeded kernel-cache hostPath volume:
  cold compile needs 30-60 min and more RAM than the 32G host has. Cache
  must be synced from shared storage before deploy
- Third-party image pinned by digest (ghcr.io/ormandj/...); own build
  planned

## Known constraints / to verify at rc testing

- Model dir must exist at the conventional host path with all 48 shards
  (present on this node; shared storage has the master copy)
- Whether the backend recognizes the hand-rsynced model as "downloaded"
  (a _download.json from the original download is present) or requires a
  re-registration through the download flow
- Readiness probe allows ~7.5 min; warm-cache startup measured ~5 min
