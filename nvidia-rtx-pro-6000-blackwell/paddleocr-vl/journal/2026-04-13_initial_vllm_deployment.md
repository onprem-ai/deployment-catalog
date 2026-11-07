# Session: Initial vLLM Deployment and Garbage Output Fix

**Date**: 2026-04-13
**Status**: WORKING (with required client-side params)
**Catalog ID**: `rtx6000bw--paddleocr-vl`
**Hardware**: nvidia-rtx6000bw (96GB VRAM)
**Inference Engine**: vLLM v0.18.2rc1 (`vllm/vllm-openai:gemma4-cu130`)

## Summary

PaddleOCR-VL is Baidu's 0.9B parameter OCR-focused vision-language model. Deployed on
RTX 6000 Blackwell using vLLM with custom remote code. Works correctly for OCR tasks
(text extraction, table recognition) **but requires specific client-side parameters**
to avoid garbage output.

## Performance

| Metric | Value |
|--------|-------|
| VRAM Usage | ~6 GiB (tiny model on 96GB GPU) |
| Context Length | 8192 tokens |
| GPU Memory Utilization | 0.06 (6%) |
| Status | Working with caveats |

## Final Working Configuration

### Server-side (vLLM args)

```
--model /model
--served-model-name paddleocr-vl
--trust-remote-code              # Required: model ships custom Python code
--gpu-memory-utilization 0.06    # Tiny model, minimal VRAM
--max-model-len 8192
--tensor-parallel-size 1
--chat-template /model/chat_template.jinja
--no-enable-prefix-caching       # Prevents stale KV cache reuse with images
--max-num-batched-tokens 16384   # Limits batch size for stability
--mm-processor-cache-gb 0        # Disables multimodal processor cache (critical)
```

### Client-side (MUST be set by caller)

```json
{
  "temperature": 0,
  "repetition_penalty": 1.1,
  "max_tokens": 4096
}
```

**`temperature: 0` is the single most impactful parameter.** Without it, the model
produces garbage/repetitive tokens regardless of server-side settings.

## Deployment Steps

1. Model downloaded via catalog UI (19 files from `PaddlePaddle/PaddleOCR-VL`, ~2 GiB)
2. Deployed via `/api/catalog/deploy?config_id=rtx6000bw--paddleocr-vl`
3. Accessible externally via AI Gateway

## Issues & Solutions

### Issue 1: Garbage/Repetitive Token Output

**Symptom**: Model returns valid OCR text mixed with repetitive garbage tokens
(e.g., repeated special characters, duplicated fragments, nonsensical sequences).

**Root Cause**: Known vLLM issue with PaddleOCR-VL. The interaction between vLLM's
multimodal processor caching, prefix caching, and PaddleOCR-VL's custom code causes
the model to enter repetitive generation loops when using non-greedy sampling.

**Investigation**:
- Web search confirmed this is a known issue in vLLM GitHub issues
- PaddleOCR official docs recommend `temperature=0` for all OCR tasks
- vLLM multimodal processor cache can serve stale processed images

**Fix (applied in order of impact)**:

| Fix | Where | Impact |
|-----|-------|--------|
| `temperature: 0` | Client request | **Critical** — switches to greedy decoding, eliminates random sampling garbage |
| `repetition_penalty: 1.1` | Client request | Moderate — penalizes repeated tokens |
| `--mm-processor-cache-gb 0` | vLLM server args | Moderate — prevents multimodal cache from serving stale data |
| `--no-enable-prefix-caching` | vLLM server args | Minor — prevents KV cache reuse across different images |
| `--max-num-batched-tokens 16384` | vLLM server args | Minor — limits batch size for stability |

### Issue 2: Download SHA Verification Failure

**Symptom**: `generation_config.json` and `processor_config.json` stuck at "downloading"
status — files downloaded correctly but hash verification failed.

**Root Cause**: HuggingFace tree API returns SHA1 (40 chars) for some small files instead
of SHA256 (64 chars). The downloader expected SHA256 for all files, causing the
comparison to always fail for these two files.

**Fix**: Manually verified file content against HuggingFace web UI, then updated
`_download.json` state to mark files as completed.

**Note**: This is a systemic issue in the HuggingFace downloader — may affect other
models with small config files. See also: `e1d12b6` commit for subdirectory structure fix.

### Issue 3: max_tokens Exceeds max_model_len

**Symptom**: API error `max_tokens=32768 cannot be greater than max_model_len=8192`

**Root Cause**: Frontend client was sending `max_tokens=32768` (default for larger models).
PaddleOCR-VL has a small 8192 context window.

**Fix**: Client-side — set `max_tokens` to 4096 or lower for this model.

## Approaches NOT Taken

| Approach | Why Not |
|----------|---------|
| Higher `gpu-memory-utilization` | Model is tiny (0.9B), no benefit from allocating more VRAM |
| `--enable-prefix-caching` | Causes stale KV cache issues with multimodal inputs |

## OCR Usage Patterns

### Task Prompts (Stage 2 / vLLM direct)

The VLM responds to task-specific prompts:

| Task | Prompt | Output Format |
|------|--------|---------------|
| Full text OCR | `OCR:` | Plain text |
| Table recognition | `Table Recognition:` | HTML `<table>` markup |
| Formula extraction | `Formula Recognition:` | LaTeX (`$$ ... $$`) |
| Chart analysis | `Chart Recognition:` | Structured text / HTML table |
| Seal recognition | `Seal Recognition:` | Plain text (v1.5 only) |
| Text spotting | `Spotting:` | Localization + recognition (v1.5 only) |

**Note**: The VLM is a parsing model, NOT a Q&A model. Use exact prompt strings above.

## Timeline

- **2026-04-13 21:04**: Initial deployment via catalog API (v1.0, vLLM-only)
- **2026-04-13 21:15**: Download stuck — fixed `_download.json` state manually
- **2026-04-13 21:30**: Basic OCR working, but garbage output observed
- **2026-04-13 22:00**: Researched vLLM + PaddleOCR-VL known issues
- **2026-04-13 22:30**: Added `--mm-processor-cache-gb 0` to catalog config
- **2026-04-13 23:18**: Redeployed with updated config via PUT `/api/inference-services/yaml`
- **2026-04-13 23:19**: Confirmed working with `temperature=0` on client side
