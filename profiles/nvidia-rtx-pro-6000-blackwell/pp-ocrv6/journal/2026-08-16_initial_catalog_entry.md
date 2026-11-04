# 2026-08-16 - Initial PP-OCRv6 Catalog Entry

**Goal:** Add and validate the PP-OCRv6 medium text detection and recognition pipeline on NVIDIA RTX PRO 6000 Blackwell.

## Research

PP-OCRv6 shipped with PaddleOCR 3.7.0 on 2026-06-11. The medium server tier uses two mandatory models:

- `PaddlePaddle/PP-OCRv6_medium_det` at revision `8e0f56fb2ef86b461d99cfc7ac5c137738985f61`
- `PaddlePaddle/PP-OCRv6_medium_rec` at revision `e5a92bcbc5cc1b494628e458d267778f0704fd7c`

The published medium-tier metrics are 86.2 detection Hmean and 83.2 recognition average accuracy. The recognition model supports 50 languages. This is traditional OCR for text polygons and recognized strings, not a replacement for PaddleOCR-VL structured document parsing.

PaddleX 3.7 basic serving exposes the general OCR pipeline at `POST /ocr`. The request accepts a base64 file or URL in `file`, with `fileType: 1` for images.

## Runtime

Built `onpremai/pp-ocrv6-server:3.7.0` from the official PaddlePaddle 3.3.1 CUDA 12.9 image and installed pinned PaddleOCR/PaddleX 3.7 packages. Published image digest:

```text
sha256:272a3b71d2843e031cb60730c912d2a9910dbde30ebde0a2913bb4e8d8c887fe
```

The card mounts both catalog-downloaded model repositories and writes an explicit pipeline config. Optional document orientation, unwarping, and text-line orientation modules are disabled so the card downloads and runs exactly the two PP-OCRv6 medium models declared in `sources`.

## Multi-source paths

The catalog renderer previously exposed only the first source through `MODEL_HOST_PATH`, forcing multi-source cards to hardcode the downloader's local directory for the second source. Indexed `SOURCE_N_HOST_PATH` variables were added and existing MTP/DFlash cards were migrated. A catalog-wide test now rejects hardcoded `/persistent/models/` host paths.

## Outcome

The initial card, pinned source revisions, pinned runtime image, API check, and journal entry are present. Deployment, API validation, resource-floor measurement, and performance results will be recorded in a follow-up entry.
