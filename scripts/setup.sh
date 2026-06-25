#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

mkdir -p "$MODELS_DIR"

# mel_spectrogram.onnx and beat_this_small.onnx from beat-this-rs
BEAT_THIS_TAG="0.3.0"
BEAT_THIS_BASE="https://raw.githubusercontent.com/danigb/beat-this-rs/refs/tags/v${BEAT_THIS_TAG}/models"

for f in mel_spectrogram.onnx beat_this_small.onnx; do
    if [ ! -f "$MODELS_DIR/$f" ]; then
        echo "Downloading $f..."
        curl -fL "$BEAT_THIS_BASE/$f" -o "$MODELS_DIR/$f"
    else
        echo "$f already exists, skipping"
    fi
done

echo ""
echo "=== htdemucs_6s ONNX models ==="
echo "The htdemucs_6s ONNX models are too large to distribute via git."
echo "To use stem separation, export the model from Python:"
echo "  1. Install demucs: pip install demucs"
echo "  2. Export to ONNX: python scripts/export_demucs_onnx.py"
echo "  Or download pre-exported models from the project's Releases page."
echo ""
echo "Place the following files in $MODELS_DIR :"
echo "  - htdemucs_6s.onnx  (fp32, ~270 MB, best quality)"
echo "  - htdemucs_6s_faster_(fp16weights).onnx  (fp16, ~136 MB, faster)"
echo ""
echo "Setup complete."
