# 2026-08-15 - Kubernetes-safe Catalog Slug

**Goal:** Correct the catalog directory slug after the first API deployment
attempt exposed the generated service-name constraint.

## Failure

The initial directory `paddleocr-vl-1.6` generated an InferenceService name
containing a period. The deployment API correctly rejected it because the
result was not a Kubernetes DNS label.

## Fix

Renamed only the catalog directory to `paddleocr-vl-1p6`. The public model ID
remains `paddleocr-vl-1.6`, preserving the version users see and the gateway
route. No model identity was inferred or changed.

## Outcome

The config ID now generates a Kubernetes-safe service name:
`rtx6k--paddleocr-vl-1p6`.
