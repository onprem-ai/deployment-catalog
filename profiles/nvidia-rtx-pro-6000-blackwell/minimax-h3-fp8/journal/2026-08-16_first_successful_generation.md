# 2026-08-16 -- First successful T2VA generation (NO offload)

## Deployment

Corrected T2VA image deployed from the model registry:

`127.0.0.1:31998/onprem-ai/minimax-h3-fp8-server@sha256:3945deb439a297e38265056480f7abd31f000d484c7fd5292b53bfc4cdfb607e`

The app deployment API rewrote the card's canonical image ref to the model-registry pull address.
GPU 0 was selected explicitly; GPU 1 remained free.

## Startup measurement

- startup: ~31-33 seconds
- torch allocated VRAM after load: 75.05 GiB
- pod RAM current/peak after load: ~23.55 / 23.69 GiB
- all model parameter/buffer residency assertions passed
- `/health`: `offload_enabled=false`
- InferenceService Ready=True

## First successful generation

Request:

```json
{
  "prompt": "A red fox trotting through a snowy pine forest, snow crunching underfoot, cinematic natural light, synchronized forest ambience",
  "seed": 42,
  "height": 544,
  "width": 960,
  "num_frames": 124,
  "num_inference_steps": 8
}
```

Result:
- HTTP 200
- wall-clock: 39.3 seconds
- 7 denoiser evaluations (8 sigma grid points includes terminal zero)
- output: 794,485-byte MP4
- duration: 5.175 seconds
- video: H.264, 960x544, 24 FPS
- audio: AAC, 32 kHz, stereo

Measured resource peaks:
- torch VRAM peak: 78.31 GiB
- nvidia-smi total observed: 81,895 MiB (~80.0 GiB)
- GPU free at peak: 15,358 MiB
- pod RAM peak: 34,692,263,936 bytes (~32.3 GiB)
- pod memory limit: 48 GiB
- NO CPU/disk/model/group/sequential offload

The deployed service remained healthy and Ready after generation.

## Maximum-duration validation

An attempted 362-frame request exposed a server-validation bug: although 362 satisfies the
`17*n+5` VAE lattice, it exceeds H3's 360-frame / 15-second ceiling. Upstream rejected it before
inference. A failing unit test was added, then `MAXIMUM_NUM_FRAMES` was corrected to 345, the
largest valid lattice value (14.375 seconds).

The 345-frame request then succeeded with the same prompt/canvas/8 sigma points:
- wall-clock: 157.7 seconds
- output: 4,517,091-byte MP4
- duration: exactly 14.375 seconds
- video: H.264, 960x544, 24 FPS
- audio: AAC, 32 kHz, stereo
- torch VRAM peak: 84.26 GiB
- nvidia-smi total observed: ~87.0 GiB (8,140 MiB free)
- pod RAM peak: 39,630,954,496 bytes (~36.9 GiB)
- pod remained healthy and Ready
- NO offload

The measured peak supports keeping `min_gi: 94` as operational headroom. Final validation-fix
image:
- local image ID: `sha256:44454b7e5430874290bd5f63397879cac775bee05fea61712e61cb6f55b1f297`
- model-registry digest:
  `sha256:5579de3127b0e3279eba7a52c1f5e5d1b07515f46691ccbdd726b538e4498442`
- tag: `torch211-cu128-t2va-max345`
