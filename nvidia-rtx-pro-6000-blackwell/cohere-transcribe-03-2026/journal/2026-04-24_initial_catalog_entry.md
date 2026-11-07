> **Time-Sensitive Information**
>
> This document contains investigation notes for rapidly evolving ML frameworks.
> Information 2+ weeks old may be outdated. Before relying on findings:
> 1. Check the session date
> 2. Search for updated information
> 3. Verify against current framework releases

# Session: Initial Catalog Entry

**Date**: 2026-04-24
**Status**: SUCCESS

**Goal**: Create catalog entry for first transcription model in the catalog.

## Model Info

- **Model**: CohereLabs/cohere-transcribe-03-2026
- **Architecture**: Fast-Conformer encoder + X-attention Transformer decoder (2B params)
- **Source**: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
- **Hardware**: NVIDIA RTX 6000 Blackwell (96 GB VRAM)

## Architecture Details

Conformer-based encoder-decoder with asymmetric design:
- **Encoder**: Fast-Conformer (>90% of 2B params) — convolutional + self-attention layers
- **Decoder**: Lightweight X-attention Transformer — minimized for fast autoregressive decoding
- **Tokenizer**: 16k multilingual BPE with byte fallback
- **Training**: From scratch on 500K hours of curated audio-transcript pairs (no Whisper distillation)
- **Custom code**: `CohereAsrForConditionalGeneration` — requires `--trust-remote-code`

## Hardware Requirements

- **GPU VRAM**: ~5 GiB (trivial for RTX 6000 Blackwell)
- **System RAM**: 4 GiB minimum (8 GiB recommended for audio processing overhead)
- **Storage**: ~4.13 GiB (single `model.safetensors`)

## Supported Languages

14 languages (must specify per request — no auto-detection):
AR, DE, EL, EN, ES, FR, IT, JA, KO, NL, PL, PT, VI, ZH

## Final Working Configuration

vLLM ≥0.19.0 with encoder-decoder optimizations (PR #38120, contributed by Cohere):

```bash
pip install librosa soundfile --quiet
python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name cohere-transcribe-03-2026 \
  --host 0.0.0.0 --port 8080 \
  --trust-remote-code \
  --gpu-memory-utilization 0.07
```

**Key settings**:
- `--trust-remote-code`: Required for custom `CohereAsrForConditionalGeneration` architecture
- `librosa` + `soundfile`: Audio processing dependencies not bundled in base vLLM Docker image
- `gpu-memory-utilization`: Very low (~6 GiB / 96 GiB) — model is tiny for this GPU

**API endpoint**: `POST /v1/audio/transcriptions` (OpenAI-compatible)

```bash
curl -X POST http://endpoint/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=cohere-transcribe-03-2026" \
  -F "language=en"
```

## Research Findings

- vLLM ≥0.19.0 has native support via PR #38120 (encoder-decoder optimizations)
- 2x throughput improvement over naive implementation via:
  - Fine-grained concurrent execution of encoder requests with variable sequence lengths
  - Convolutional encoder executed on minimally padded batches
  - Packed representation conversion for FlashAttention-based decoder
- Model is gated (auto-approval) — HuggingFace token required for download
- Audio deps (`librosa`, `soundfile`) may not be in base vLLM Docker image — install at startup
- Benchmarks: #1 on HF Open ASR Leaderboard (5.42% avg WER), 524x real-time speed

## Routing Decision

`envoy` (path-based) because `/v1/audio/transcriptions` is not chat/completions.

## Deployment Results

- Downloaded via catalog API (4 files needed manual retry due to failed HEAD requests)
- Deployed via `POST /api/catalog/deploy` — pod ready in ~5 minutes (image pull + pip install + model load)
- Transcription tested successfully via ClusterIP and envoy gateway
- Test audio: VoxPopuli EN demo → "If not, there will be a big crisis between you and the European Parliament."

**Measured VRAM**: 5888 MiB (~5.75 GiB) — `min_gi: 6` in catalog is accurate.

## Notes

- `vllm/vllm-openai:v0.19.0` required `pip install librosa soundfile` at startup (~20s added)
- `context_length` must be omitted (not set to 0) — Pydantic `ge=1` constraint rejects 0
- `HF_API_KEY` must be set in backend env for gated model downloads
- Pod was in ContainerCreating for ~3.5 min (image pull), then Running for ~1.5 min (pip install + model load + warmup)

## Known Limitations

- **No automatic language detection** — language must be specified in each request
- **No timestamps** — no word-level or segment-level timestamps in output
- **No speaker diarization** — single-speaker output only
- **Eager transcriber** — transcribes silence/background noise; recommend VAD preprocessing
- **No code-switching** — inconsistent on mixed-language audio
- **35-second chunking** — native chunking for long-form audio, but may cause boundary artifacts

## TODO

- [x] Verify `vllm/vllm-openai:v0.19.0` image includes audio deps — confirmed: needs `pip install librosa soundfile` at startup
- [x] Measure actual VRAM usage — 5888 MiB, `min_gi: 6` is correct
- [ ] Test with various audio formats (MP3, FLAC, OGG)
- [ ] Test long-form audio (>35 seconds) chunking behavior
- [ ] Benchmark RTFx (real-time factor) on RTX 6000 Blackwell
- [ ] Test multi-language transcription
- [ ] Consider building custom Docker image with audio deps baked in if startup install is too slow
