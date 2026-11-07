#!/bin/bash
# Test LTX-2.3 inference using a Python venv with ltx-pipelines.
# Run this AFTER downloads complete and the model directory is organized.
#
# Usage: bash test_inference.sh

set -euo pipefail

MODEL_DIR="/persistent/models/ltx-2.3"
VENV_DIR="/tmp/ltx-venv"
OUTPUT="/tmp/ltx-test-output.mp4"

echo "=== Checking model files ==="
for f in \
    "$MODEL_DIR/ltx-2.3-22b-distilled-1.1.safetensors" \
    "$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    "$MODEL_DIR/gemma-3-12b-it-qat-q4_0-unquantized/config.json"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        exit 1
    fi
done

echo ""
echo "=== Setting up Python venv ==="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    "$VENV_DIR/bin/pip" install ltx-pipelines
fi

echo ""
echo "=== Running inference test ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$VENV_DIR/bin/python3" -c "
import torch
import time

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GiB')

from ltx_pipelines.distilled import DistilledPipeline
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.utils.media_io import encode_video

print()
print('Loading pipeline with FP8 quantization...')
start = time.monotonic()

pipeline = DistilledPipeline(
    distilled_checkpoint_path='$MODEL_DIR/ltx-2.3-22b-distilled-1.1.safetensors',
    spatial_upsampler_path='$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors',
    gemma_root='$MODEL_DIR/gemma-3-12b-it-qat-q4_0-unquantized',
    loras=[],
    quantization=QuantizationPolicy.fp8_cast(),
)

load_time = time.monotonic() - start
print(f'Pipeline loaded in {load_time:.1f}s')
print(f'VRAM allocated: {torch.cuda.memory_allocated() / (1024**3):.1f} GiB')
print()

print('Generating test video (25 frames, 512x768)...')
start = time.monotonic()

video_iter, audio = pipeline(
    prompt='A golden retriever puppy playing in autumn leaves in a sunny park',
    seed=42,
    height=512,
    width=768,
    num_frames=25,
    frame_rate=25.0,
    images=[],
)

encode_video(
    video=video_iter,
    fps=25,
    audio=audio,
    output_path='$OUTPUT',
    video_chunks_number=1,
)

gen_time = time.monotonic() - start
print(f'Video generated in {gen_time:.1f}s')
print(f'Output: $OUTPUT')
print(f'VRAM peak: {torch.cuda.max_memory_allocated() / (1024**3):.1f} GiB')
print()
print('SUCCESS')
"

echo ""
echo "=== Output file ==="
ls -lh "$OUTPUT"
