# 2026-08-16 - First Deployment and API Validation

**Goal:** Prove the PP-OCRv6 card works end to end through catalog download, catalog deployment, AI Gateway routing, and the authenticated playground API.

## Download and deployment

Both pinned Hugging Face sources downloaded through `POST /api/model-catalog/download` and passed downloader verification:

- Detection: 4/4 files, 62.30 MB payload
- Recognition: 4/4 files, 76.86 MB payload
- Runtime image copied into the node model registry through the same application download action

An existing model deployment on the GPU was removed through the application inference-service API to release the GPU. PP-OCRv6 was then deployed through `POST /api/model-catalog/deploy` with `gpu_indices=0`.

The deployment became Ready with zero restarts. Startup logs explicitly showed both local models:

```text
Creating model: ('PP-OCRv6_medium_det', '/models/detection', None)
Creating model: ('PP-OCRv6_medium_rec', '/models/recognition', None)
Uvicorn running on http://0.0.0.0:8080
```

## Authenticated application-path validation

Requests were sent to the API gateway's passthrough path `POST /pp-ocrv6/ocr` with an API gateway key; the API gateway routes it by the profile's `path_prefix` (`/{routing_model_id}`).

### Base64 image

Used the official `general_ocr_002.png` fixture as base64. The request returned HTTP 200 in 0.544 seconds with:

- `errorCode: 0`
- One OCR page
- 33 detected polygons
- 33 recognized strings and confidence scores
- Correct Chinese and English boarding-pass text, including `登机牌`, `BOARDING PASS`, flight `MU 2379`, names, and ticket information

### Image URL and word boxes

Used the official `general_ocr_001.png` URL with `returnWordBox: true`. The request returned HTTP 200 in 7.890 seconds with one page, four recognized text regions, and word/region box output.

## Resource observations

On the RTX PRO 6000 Blackwell:

- Idle GPU memory after startup: 779 MiB
- GPU memory after validation requests: 1,109 MiB
- Idle pod RAM: 1,015 MiB
- Restarts: 0

The Paddle runtime manages its GPU allocation independently of the catalog memory value, and deployment succeeded with the catalog scheduler floor set to zero. The card therefore records `min_gi: 0`; the measured operational footprint remains documented here.

## Outcome

PP-OCRv6 medium is working end to end through the official application path. Detection, recognition, confidence scores, polygons, base64 input, URL input, and returned boxes were validated. The card remains experimental pending broader multilingual and industrial fixtures plus repeated throughput/latency measurements.
