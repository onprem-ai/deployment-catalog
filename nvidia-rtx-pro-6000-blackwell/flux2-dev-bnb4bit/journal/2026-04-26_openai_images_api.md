# 2026-04-26 — Add OpenAI-Compatible Image Generation API

**Goal:** Extend flux2-server with the OpenAI `/v1/images/generations` standard endpoint, and add a backend proxy handler to route requests through the platform API.

## Why OpenAI `/v1/images/generations`

This is the de facto standard for text-to-image APIs. Implemented by:
- vLLM-Omni (the leading open-source diffusion inference server)
- OpenVINO Model Server
- DeepInfra and all major cloud providers

All major clients speak it natively: OpenAI Python SDK, LiteLLM, OpenWebUI.

Alternatives considered:
- **A1111 `/sdapi/v1/txt2img`**: Legacy, A1111-specific. Richer fields but no ecosystem adoption outside A1111.
- **ComfyUI `/prompt`**: Graph-based, not a simple text-in/image-out API. Async polling model.

## What Was Implemented

### flux2-server (`server.py`)

1. **`POST /v1/images/generations`** — OpenAI-compatible endpoint
   - Request: `prompt`, `model` (ignored), `n`, `size`, `response_format`, plus FLUX-specific extensions (`seed`, `num_inference_steps`, `guidance_scale`)
   - Response: `{ "created": <timestamp>, "data": [{ "b64_json": "<base64 PNG>" }] }`
   - Supported sizes: 256x256, 512x512, 768x768, 1024x1024, 1024x1536, 1536x1024
   - Default inference steps auto-selected by resolution (20 for ≤512, 24 for 768, 28 for ≥1024)
   - Supports `n` > 1 with per-image seed incrementing

2. **Refactored**: Extracted `_generate_image()` helper used by both `/v1/images/generations` and legacy `/generate`

3. **Legacy `/generate`** endpoint preserved (returns raw PNG binary)

### Backend proxy

Added `POST /api/inference-services/images/generations?name=...&namespace=...`
- Same pattern as existing `chat/completions` and `embeddings` handlers
- Discovers KServe pod via label selector, proxies to `http://{pod_ip}:{port}/v1/images/generations`
- 600s timeout (image generation can take 30s+ per image)
- Standard error handling (404/502/503/504/500)

## TODO

- **`response_format: "url"` support**: Currently returns 400. Needs temp file storage + a GET endpoint to serve generated images by ID. Consider using a TTL-based cache directory with periodic cleanup.

## Outcome

OpenAI-compatible image generation API implemented on both the container server and backend proxy layers. Legacy endpoint preserved for backward compatibility.
