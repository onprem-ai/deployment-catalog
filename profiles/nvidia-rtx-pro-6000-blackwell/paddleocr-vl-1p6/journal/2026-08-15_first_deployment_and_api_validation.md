# 2026-08-15 - First Deployment and API Validation

**Goal:** Deploy PaddleOCR-VL 1.6 and validate the documented
`POST /layout-parsing` contract through the application's playground proxy.

## Deployment

The model weights and both pinned offline images were downloaded through the
model-catalog API. The running Qwen3.8 1M service was then removed through the
inference-service API, and PaddleOCR-VL 1.6 was deployed on GPU 0 through the
model-catalog API.

The initial catalog directory used a period (`paddleocr-vl-1.6`), which the
deployment API correctly rejected because it generated an invalid Kubernetes
DNS label. The directory was renamed to `paddleocr-vl-1p6`; the public model ID
and gateway route remain `paddleocr-vl-1.6`.

Both containers became Ready without restarts. Runtime logs confirmed that the
pinned offline images contain the intended assets:

```text
pipeline_name: PaddleOCR-VL-1.6
model_name: PaddleOCR-VL-1.6-0.9B
vLLM served model: PaddleOCR-VL-1.6-0.9B
model path: /home/paddleocr/.paddlex/official_models/PaddleOCR-VL-1.6
context length: 16384
model load: 1.8232 GiB
GPU KV cache: 318,688 tokens
```

## API Contract Test

Requests were sent to the API gateway's passthrough path
`POST /paddleocr-vl-1.6/...` with an API gateway key, which exercises the
authenticated path through the API gateway rather than bypassing it with a
direct request to the container.

### Catalog Health-check Example

The card's embedded base64 1x1 image request returned HTTP 200 with:

```text
errorCode: 0
errorMsg: Success
result.layoutParsingResults: one page
result.dataInfo: 1 x 1 image
```

An empty block list is expected for a one-pixel test image. This confirms the
1.5-compatible request fields (`file`, `fileType`) and response envelope
(`result.layoutParsingResults`) remain valid in 1.6.

### Official PaddleOCR Demo Image

The official documentation image URL was then submitted with the same request
shape. The service returned HTTP 200, `errorCode: 0`, one parsed page, and a
non-empty structured result for a 1524x1368 Chinese newspaper page.

Observed output included:

- Markdown beginning with `#` and the recognized article title.
- 31 layout blocks.
- `doc_title`, `paragraph_title`, `text`, `image`, and `vision_footnote` labels.
- Bounding boxes, polygon points, block IDs, groups, and reading order.
- Embedded output images and the full reconstructed article text.

The recognized title was the Chinese text meaning "Support bilateral
exchanges, build bridges of friendship," matching the source document. This
exercises layout detection, VLM recognition, reading-order reconstruction, and
Markdown/JSON post-processing across all three pipeline stages.

## Outcome

PaddleOCR-VL 1.6 is deployed, Ready, and left running. The documented 1.5-style
`/layout-parsing` API is compatible in the tested request and response paths,
and the official sample produces substantive structured OCR output. The card
remains experimental pending broader document fixtures, resource-floor
measurement, restart-coupling validation, and performance benchmarking.
