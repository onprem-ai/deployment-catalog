# Session: Full Pipeline Exploration, v1.5 Migration, and KServe Sidecar Deployment

**Date**: 2026-04-14

## Full Pipeline: Official PaddleOCR Docker Deployment

### Architecture — 3-Stage Pipeline

PaddleOCR-VL is designed as a 3-stage pipeline. Our initial KServe deployment only runs
Stage 2 (VLM via vLLM). The official Docker setup runs ALL 3 stages:

```
Client → paddleocr-vl-api (:8091/layout-parsing)     ← Stage 1 + 3
              ↓
         paddleocr-vlm-server (:8090/v1)              ← Stage 2 (vLLM)
```

| Stage | Component | What It Does |
|-------|-----------|-------------|
| 1 | **PP-DocLayoutV3** (layout detection) | Localizes semantic regions (text, tables, formulas, charts), predicts reading order. RT-DETR-based detection + pointer network for ordering |
| 2 | **PaddleOCR-VL-0.9B** (VLM recognition) | Per-region recognition with task-specific prompts (OCR, Table, Formula, Chart). Served via vLLM OpenAI API |
| 3 | **Post-processing** (aggregation) | Aggregates all region outputs in reading order, produces structured Markdown + JSON with bounding boxes |

### Docker Images (Blackwell SM120)

Two containers are required:

| Container | Image | Size | Role |
|-----------|-------|------|------|
| `paddleocr-vlm-server` | `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120-offline` | 14.4 GB | Stage 2: VLM via vLLM |
| `paddleocr-vl-api` | `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline` | 12.1 GB | Stage 1 + 3: Layout + post-processing |

The `-offline` suffix means models are bundled inside (no download on startup).
The `-sm120` suffix is for Blackwell GPUs (RTX 5090, RTX 6000 BW, etc.).

**Image tags for other hardware:**
- Standard NVIDIA: `latest-nvidia-gpu-offline`
- Blackwell SM120: `latest-nvidia-gpu-sm120-offline`
- AMD: see `accelerators/amd-gpu/` in PaddleOCR repo

### Bundled Model Version

The SM120-offline images bundle **PaddleOCR-VL-1.5** (not v1.0):
- Located at `/home/paddleocr/.paddlex/official_models/PaddleOCR-VL-1.5/`
- Model name: `PaddleOCR-VL-1.5-0.9B`
- v1.5 upgrades over v1.0: seal recognition, text spotting, PP-DocLayoutV3, cross-page table merging

**Important**: If you pass `--model_name PaddleOCR-VL-0.9B` (v1.0), the container will
try to **download** v1.0 from the internet even though v1.5 is bundled. Always use
`--model_name PaddleOCR-VL-1.5-0.9B` with these images.

### How to Run

**Step 1: Start VLM server (Stage 2)**

```bash
# Create vLLM config to limit GPU memory (default 0.5 is too high if other models are loaded)
cat > /tmp/vllm_config.yaml << 'EOF'
gpu-memory-utilization: 0.15
max-num-seqs: 32
EOF

docker run -d --name paddleocr-vlm-server \
  --gpus all \
  --network host \
  --shm-size 64g \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  -v /tmp/vllm_config.yaml:/opt/vllm_config.yaml:ro \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120-offline \
  paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B \
    --host 0.0.0.0 --port 8090 --backend vllm \
    --backend_config /opt/vllm_config.yaml
```

**Step 2: Create pipeline config**

```yaml
# /tmp/pipeline_config_vllm.yaml
pipeline_name: PaddleOCR-VL-1.5
batch_size: 64
use_queues: True
use_doc_preprocessor: False
use_layout_detection: True
use_chart_recognition: False
use_seal_recognition: False
format_block_content: False
merge_layout_blocks: True

SubModules:
  LayoutDetection:
    module_name: layout_detection
    model_name: PP-DocLayoutV3
    model_dir: null
    batch_size: 8
    threshold: 0.3
    layout_nms: True
    layout_unclip_ratio: [1.0, 1.0]
  VLRecognition:
    module_name: vl_recognition
    model_name: PaddleOCR-VL-1.5-0.9B
    model_dir: null
    batch_size: 4096
    genai_config:
      backend: vllm-server
      server_url: http://127.0.0.1:8090/v1

Serving:
  extra:
    max_num_input_imgs: null
```

**Step 3: Start pipeline API (Stage 1 + 3)**

```bash
docker run -d --name paddleocr-vl-api \
  --gpus all \
  --network host \
  --shm-size 64g \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  -v /tmp/pipeline_config_vllm.yaml:/home/paddleocr/pipeline_config_vllm.yaml:ro \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline \
  /bin/bash -c "paddlex --serve --pipeline /home/paddleocr/pipeline_config_vllm.yaml --port 8091"
```

### API Endpoints

The pipeline API (`paddleocr-vl-api`) exposes:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/layout-parsing` | **Full 3-stage pipeline** — layout detection + VLM recognition + markdown aggregation |
| `POST` | `/restructure-pages` | Page restructuring (not yet tested) |

The VLM server (`paddleocr-vlm-server`) exposes standard vLLM/OpenAI endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List models |
| `POST` | `/v1/chat/completions` | **Stage 2 only** — direct VLM inference |

### `/layout-parsing` Request Format

```bash
curl -X POST http://localhost:8091/layout-parsing \
  -H "Content-Type: application/json" \
  -d '{
    "file": "<base64-encoded image>",
    "fileType": 1
  }'
```

**Parameters:**
- `file`: Base64-encoded image (PNG, JPG) or PDF
- `fileType`: `1` = base64 image data
- `useImgOrientationCls`: `true` to auto-rotate (requires `use_doc_preprocessor: True` in config)
- `useImgUnwrapping`: `true` to unwarp curved documents
- `useSealTextDet`: `true` to detect seal text

### `/layout-parsing` Response Format

```json
{
  "logId": "uuid",
  "result": {
    "layoutParsingResults": [{
      "prunedResult": {
        "width": 800,
        "height": 600,
        "parsing_res_list": [
          {
            "block_label": "text",       // header, text, table, formula, chart, image, footer, etc.
            "block_content": "...",       // Recognized text / HTML table / LaTeX formula
            "block_bbox": [x1, y1, x2, y2],
            "block_id": 0,
            "block_order": 1,            // Reading order (null for ignored labels)
            "block_polygon_points": [[x,y], ...]  // Polygon coordinates
          }
        ]
      },
      "markdown": {
        "text": "...",                   // Full document as Markdown (text + HTML tables + LaTeX)
        "images": {}
      },
      "outputImages": {
        "layout_det_res": "base64..."    // Layout detection visualization
      }
    }]
  }
}
```

### Tested Results

**Test: Document with title, paragraphs, and table**

Input: 800x600 image with "Annual Report 2026" header, two text paragraphs, and a
quarterly revenue table.

Layout detection results:

| Block | Label | Content (truncated) |
|-------|-------|---------------------|
| 0 | `header` | Annual Report 2026 |
| 1 | `text` | The company achieved record revenue of $42.5 million... |
| 2 | `table` | `<table><tr><td>Quarter</td><td>Revenue</td>...` (full HTML) |
| 3 | `text` | Strategic investments in AI infrastructure drove... |

Markdown output:
```
The company achieved record revenue of $42.5 million in the fiscal year
2025-2026, representing a 23% increase over the previous year.

<table border=1>...</table>

Strategic investments in AI infrastructure drove significant efficiency
gains across all divisions.
```

**Observations:**
- PP-DocLayoutV3 correctly identified header, text blocks, and table
- Reading order preserved (block_order: 1, 2)
- Table recognized as full HTML with all cells correct
- Headers excluded from markdown (configurable via `markdown_ignore_labels`)
- VLM v1.5 produces clean output with no garbage (no `temperature=0` needed at pipeline level)

### VLM v1.5 vs v1.0 Comparison

| Feature | v1.0 (current KServe deployment) | v1.5 (official Docker) |
|---------|----------------------------------|------------------------|
| Model name | `PaddleOCR-VL` | `PaddleOCR-VL-1.5` |
| Parameters | 0.9B | 0.9B (same) |
| max_model_len | 8192 | **16384** (2x larger) |
| Layout model | PP-DocLayoutV2 | **PP-DocLayoutV3** (polygon support) |
| Task prompts | 4 (OCR, Table, Formula, Chart) | **6** (+Seal, +Spotting) |
| Cross-page tables | No | **Yes** (auto-merge) |
| vLLM version | v0.18.2rc1 | v0.10.2 (older but stable) |
| Garbage output | Requires `temperature=0` | No garbage observed |

### Configuration Tips

**`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`**: Skip connectivity check to Baidu CDN
on startup. Essential for air-gapped environments or slow connections to China.

**`gpu-memory-utilization` in backend config**: Default is `0.5` (47.5 GiB). If other
models are loaded on the same GPU, reduce this. The 0.9B model only needs ~2 GiB for
weights + a few GiB for KV cache. We used `0.15` successfully alongside other models.

**`shm_size: 64g`**: Required by PaddlePaddle for shared memory IPC between processes.

**Pipeline config options:**
- `use_layout_detection: True/False` — enable/disable Stage 1 (layout detection)
- `use_chart_recognition: True/False` — enable chart-to-table conversion
- `use_seal_recognition: True/False` — enable seal text detection (v1.5)
- `use_doc_preprocessor: True/False` — enable orientation correction + unwarping
- `merge_layout_blocks: True` — merge adjacent blocks of same type
- `markdown_ignore_labels: [...]` — exclude labels from markdown output

### Known Issues with Official Docker Setup

| Issue | Details | Workaround |
|-------|---------|------------|
| `gpu-memory-utilization` default too high | Default 0.5 requires ~47.5 GiB free VRAM | Pass `--backend_config` with lower value |
| Model download on wrong `--model_name` | Using `PaddleOCR-VL-0.9B` triggers v1.0 download even though v1.5 is bundled | Use `PaddleOCR-VL-1.5-0.9B` |
| Connectivity check blocks startup | Checks Baidu CDN connectivity, slow outside China | Set `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` |
| FlashInfer not available | Warning about fallback to PyTorch sampling | Performance impact minimal for 0.9B model |
| Older vLLM (v0.10.2) in container | Official image uses vLLM v0.10.2, not latest | Works fine for this model; newer vLLM has garbage output issues anyway |

### Comparison: vLLM-only vs Full Pipeline

| Aspect | vLLM-only (current KServe) | Full Pipeline (official Docker) |
|--------|---------------------------|--------------------------------|
| Stages | Stage 2 only | All 3 stages |
| Layout detection | None | PP-DocLayoutV3 (RT-DETR + pointer network) |
| Reading order | Not handled | Predicted by layout model |
| Input | Single element (image + task prompt) | Full page / multi-page document |
| Output | Raw text from VLM | Structured Markdown + JSON with bboxes |
| API | OpenAI `/v1/chat/completions` | PaddleX `/layout-parsing` |
| Containers | 1 (vLLM) | 2 (VLM server + pipeline API) |
| Use case | Element-level recognition, custom pipelines | **Complete document parsing** |

### Mounting Pre-Downloaded Model

Instead of using the bundled model inside the container, you can mount our
HuggingFace-downloaded model files. This avoids downloading and lets us use
our existing model management infrastructure.

**Our model path**: `/persistent/models/PaddlePaddle--PaddleOCR-VL--model`

**How to mount into VLM server:**

```bash
docker run -d --name paddleocr-vlm-server \
  --gpus all --network host --shm-size 64g \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  -v /tmp/vllm_config.yaml:/opt/vllm_config.yaml:ro \
  -v /persistent/models/PaddlePaddle--PaddleOCR-VL--model:/model:ro \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120-offline \
  paddleocr genai_server --model_name PaddleOCR-VL-0.9B \
    --model_dir /model \
    --host 0.0.0.0 --port 8090 --backend vllm \
    --backend_config /opt/vllm_config.yaml
```

Key flags:
- `--model_dir /model` — tells PaddleOCR to use the mounted directory instead of downloading
- `-v .../PaddlePaddle--PaddleOCR-VL--model:/model:ro` — mounts our pre-downloaded files read-only
- `--model_name PaddleOCR-VL-0.9B` — must match the model version (v1.0)

**Pipeline config must also use the correct model name:**

```yaml
VLRecognition:
  model_name: PaddleOCR-VL-0.9B       # Not PaddleOCR-VL-1.5-0.9B
  genai_config:
    server_url: http://127.0.0.1:8090/v1
```

**Tested**: Full 3-stage pipeline (layout detection + VLM + markdown output) works
identically with mounted model vs bundled model. Output is the same.

#### v1.0 vs v1.5 File Differences

Our download is v1.0 (`PaddlePaddle/PaddleOCR-VL`). The container bundles v1.5.

| File | v1.0 (ours) | v1.5 (bundled) | Impact |
|------|-------------|----------------|--------|
| `config.json` | Identical | Identical | None |
| `model.safetensors` | 1,917,255,968 bytes | Same | **Same weights** |
| `modeling_paddleocr_vl.py` | 2487 lines | 2651 lines | v1.5 adds seal + spotting tasks |
| `chat_template.jinja` | `"Assistant: "` | `"Assistant:\n"` | Minor formatting |
| `tokenizer.json` | 11,187,679 bytes | 11,189,060 bytes | Minor differences |
| `tokenizer_config.json` | 185,587 bytes | 186,947 bytes | Minor differences |

**Conclusion**: The model weights are identical. The differences are in custom code
(`modeling_paddleocr_vl.py`) and tokenizer configs. v1.0 works fine for 4 core tasks
(OCR, Table, Formula, Chart). For seal recognition and text spotting, upgrade to v1.5
by downloading from `PaddlePaddle/PaddleOCR-VL-1.5`.

---

## Decision: Switch to PaddleOCR-VL-1.5 Exclusively

**Decision**: Remove v1.0 from catalog config and use v1.5 (`PaddlePaddle/PaddleOCR-VL-1.5`) as the only supported version. There is no reason to keep v1.0.

**Rationale**:
- Model weights are identical (same `model.safetensors`, 1,917,255,968 bytes)
- v1.5 has strictly more features: seal recognition, text spotting, PP-DocLayoutV3, cross-page table merging
- v1.5 has 2x context length (16384 vs 8192)
- v1.5 produces cleaner output — no garbage token issues observed in testing
- v1.0 custom code (`modeling_paddleocr_vl.py`) is a strict subset of v1.5
- Official Docker images already bundle v1.5 exclusively

**Changes applied to `inference.yaml`**:
- `source.repo`: `PaddlePaddle/PaddleOCR-VL` → `PaddlePaddle/PaddleOCR-VL-1.5`
- Removed PP-DocLayoutV2 subdirectory files from `source.files` (v1.5 doesn't ship them)
- `capabilities.context_length`: 8192 → 16384
- Memory profiles updated: added 16k context option (8 GiB), adjusted 8k and 4k
- `display.name`: `PaddleOCR-VL` → `PaddleOCR-VL 1.5`
- `display.release_date`: `2025-11-01` → `2026-01-29`
- `--served-model-name`: `paddleocr-vl` → `paddleocr-vl-1.5`
- `ai_gateway.model_name`: `paddleocr-vl` → `paddleocr-vl-1.5`
- Header comments updated to reflect v1.5 pipeline (PP-DocLayoutV3, 16K context)

**Next**: Reload catalog, download v1.5 model files, redeploy KServe InferenceService.

---

## Decision: Two-Container KServe Sidecar Deployment

**Decision**: Replace single vLLM container (Stage 2 only) with two-container KServe
pod using official Baidu Docker images for the full 3-stage pipeline.

**Architecture**:
```
KServe Pod (paddleocr-vl-1-5)
├── kserve-container (pipeline API) — port 8080 (KServe-facing)
│   Image: paddleocr-vl:latest-nvidia-gpu-sm120-offline (12.1 GB)
│   Stages 1 + 3: PP-DocLayoutV3 layout detection + post-processing
│   Serves: /layout-parsing, /health
│
└── vlm-server (sidecar) — port 8090 (pod-internal)
    Image: paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120-offline (14.4 GB)
    Stage 2: PaddleOCR-VL-1.5-0.9B via vLLM
    Bundled model at /home/paddleocr/.paddlex/official_models/PaddleOCR-VL-1.5/
```

**Key implementation details**:
- Pipeline config written inline via heredoc (VLM URL: `http://127.0.0.1:8090/v1`)
- Pipeline API waits for VLM sidecar health before starting (`curl` loop)
- vLLM backend config written inline with `gpu-memory-utilization: ${GPU_MEMORY_UTILIZATION}`
- Shared memory via emptyDir (medium: Memory, 2Gi) + hostIPC: true
- **Critical**: Both containers need `NVIDIA_VISIBLE_DEVICES=all` and
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility` env vars — official PaddleOCR images
  don't set these, and the NVIDIA Container Runtime won't inject GPU devices without them
- AI Gateway integration omitted — `/layout-parsing` is not OpenAI-compatible
- Source section kept (satisfies schema gate) but containers use bundled models
- Images must be pre-imported into containerd (`docker save | ctr -n k8s.io images import`)

**Tested**: Full pipeline produces correct Markdown with layout blocks and reading order.

---

## Next Steps

1. **Evaluate v1.5 upgrade for KServe deployment** — v1.5 has 2x context, no garbage issues, and more tasks
2. **Consider full pipeline integration** — either as a separate KServe service or sidecar container
3. **Test with real documents** — PDFs, scanned pages, complex layouts with mixed content
4. **Benchmark throughput** — compare single-element vLLM vs full pipeline for batch processing
5. **Download v1.5 model** — get `PaddlePaddle/PaddleOCR-VL-1.5` via our catalog downloader for the extra tasks

## Timeline

- **2026-04-14 07:07**: Pulled official Docker images (SM120 Blackwell variants, ~26.5 GB total)
- **2026-04-14 07:19**: Started VLM server with bundled v1.5 model on port 8090
- **2026-04-14 07:20**: VLM v1.5 confirmed working — clean OCR output, no garbage
- **2026-04-14 07:22**: Started pipeline API on port 8091
- **2026-04-14 07:23**: Full 3-stage pipeline tested — layout detection + table recognition + markdown output working
- **2026-04-14 08:02**: Mounted pre-downloaded v1.0 model into VLM server — works with `--model_dir`
- **2026-04-14 08:03**: Full pipeline tested with mounted model — identical results to bundled
- **2026-04-14 08:30**: Decision: ditch v1.0, switch exclusively to v1.5 (see above)
- **2026-04-14 09:10**: Deployed two-container KServe sidecar (pipeline API + VLM server)
- **2026-04-14 09:11**: VLM sidecar failed — "Failed to infer device type". Missing `NVIDIA_VISIBLE_DEVICES` env var
- **2026-04-14 09:12**: Fixed: added `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility` to both containers
- **2026-04-14 09:13**: Both containers running (2/2). Full 3-stage pipeline verified via `/layout-parsing`
