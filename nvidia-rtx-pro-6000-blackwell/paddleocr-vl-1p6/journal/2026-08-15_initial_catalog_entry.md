# 2026-08-15 - Initial PaddleOCR-VL 1.6 Catalog Entry

**Goal:** Add a separate experimental PaddleOCR-VL 1.6 full-pipeline card while
preserving the stable 1.5 deployment for rollback and comparison.

## Upstream Findings

PaddleOCR-VL 1.6 was released on 2026-05-28 with PaddleOCR 3.6.0. Paddle states
that 1.6 uses exactly the same model architecture as 1.5 and supports zero-cost
plug-and-play migration. The documented full service retains the same
three-stage design and `/layout-parsing` API:

1. PP-DocLayoutV3 layout detection.
2. PaddleOCR-VL-1.6-0.9B region recognition through a vLLM sidecar.
3. Structured Markdown/JSON aggregation and formatting.

The official 1.6 pipeline config changes the versioned names to:

```text
pipeline_name: PaddleOCR-VL-1.6
model_name: PaddleOCR-VL-1.6-0.9B
```

The endpoint contract remains `POST /layout-parsing`. This card preserves the
1.5 request and response health check so exact compatibility is tested through
the application path at deployment time.

## Model and Image Pins

The Hugging Face source is pinned to inspected commit:

```text
PaddlePaddle/PaddleOCR-VL-1.6
c5630abae1d940eafe0697512a0325494b02ab42
```

The selected model payload is approximately 1.80 GiB. The two current official
SM120 offline image tags were resolved through the Baidu registry API and
pinned by digest on 2026-08-15:

```text
paddleocr-vl:
  sha256:0971c409d1cab2b12aa17b76855e36ac8eb9fb1adc97dbeea15e9b09432a4a3b
paddleocr-genai-vllm-server:
  sha256:bffd525308facf5dba2f8eca44ab476704a0ae3bfdcba25f77655973e4c0a7ca
```

The offline images are required because air-gapped operation cannot fetch
models at pod startup.

## Capabilities and Accuracy

PaddleOCR-VL 1.6 retains the compact 0.9B architecture, 16K context, 109
languages, and multimodal document parsing. Paddle reports 96.33% on
OmniDocBench v1.6, with improvements in tables, formulas, text, charts, seals,
ancient documents, and rare Chinese characters.

No estimated performance values were copied from the 1.5 card. Throughput and
resource floors must be measured on the target RTX PRO 6000 deployment.

## Decisions

- Added a new card instead of mutating stable 1.5, preserving rollback.
- Used `model_id: paddleocr-vl-1.6` so both versions can coexist without route
  or model identity collisions.
- Kept maturity experimental pending download, deployment, API compatibility,
  OCR quality, memory, and restart-coupling tests.
- Retained the established two-container liveness coupling and disabled vLLM
  multimodal processor caching because those fixes address production failure
  modes independent of the model version.

## Sources

- https://www.paddleocr.ai/latest/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html
- https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html
- https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/pipeline_config_vllm.yaml
- https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6
- https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.6.0

## Outcome

Created the experimental 1.6 catalog card and mandatory journal. It is not yet
deployed or validated; the existing 1M Qwen deployment was left unchanged.
