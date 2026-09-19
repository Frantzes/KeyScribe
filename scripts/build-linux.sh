#!/usr/bin/env bash

# shellcheck shell=bash

set -euo pipefail

TARGET=""
OUTPUT_ROOT="build/linux"
SKIP_CARGO_BUILD=0

usage() {
    cat <<'EOF'
Usage: scripts/build-linux.sh [options]

Options:
  --target <triple>     Rust target triple (default: host Linux target)
  --output-root <path>  Output folder root (default: build/linux)
  --skip-cargo-build    Skip cargo build step
  -h, --help            Show this help
EOF
}

detect_host_target() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    if [[ "$os" != "Linux" ]]; then
        echo "This script must run on Linux." >&2
        exit 1
    fi

    case "$arch" in
        x86_64)
            echo "x86_64-unknown-linux-gnu"
            ;;
        aarch64|arm64)
            echo "aarch64-unknown-linux-gnu"
            ;;
        *)
            echo "Unsupported Linux architecture: $arch" >&2
            exit 1
            ;;
    esac
}

target_to_arch_label() {
    case "$1" in
        x86_64-unknown-linux-gnu)
            echo "x64"
            ;;
        aarch64-unknown-linux-gnu)
            echo "arm64"
            ;;
        *)
            echo "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-'
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            [[ $# -ge 2 ]] || { echo "Missing value for --target" >&2; exit 1; }
            TARGET="$2"
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || { echo "Missing value for --output-root" >&2; exit 1; }
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --skip-cargo-build)
            SKIP_CARGO_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$TARGET" ]]; then
    TARGET="$(detect_host_target)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ "$SKIP_CARGO_BUILD" -eq 0 ]]; then
    echo "Building release binary for $TARGET..."
    cargo build --release --target "$TARGET"
fi

BUNDLE_BINARY_NAME="keyscribe"
# Honor CARGO_TARGET_DIR when set (e.g. container builds that keep the
# host's target/ untouched); fall back to the workspace target/ dir.
CARGO_TARGET_BASE="${CARGO_TARGET_DIR:-$REPO_ROOT/target}"
BINARY_CANDIDATES=(
    "$CARGO_TARGET_BASE/$TARGET/release/$BUNDLE_BINARY_NAME"
    "$REPO_ROOT/target/$TARGET/release/$BUNDLE_BINARY_NAME"
    "$CARGO_TARGET_BASE/release/$BUNDLE_BINARY_NAME"
    "$REPO_ROOT/target/release/$BUNDLE_BINARY_NAME"
)

BINARY_PATH=""
for candidate in "${BINARY_CANDIDATES[@]}"; do
    if [[ -f "$candidate" ]]; then
        BINARY_PATH="$candidate"
        break
    fi
done

if [[ -z "$BINARY_PATH" ]]; then
    echo "Could not find $BUNDLE_BINARY_NAME in target/$TARGET/release or target/release" >&2
    exit 1
fi

# --- Core ONNX models (small, required for transcription) ---
# htdemucs_6s.onnx is NOT bundled: it is downloaded on first local Demucs use
# (default separation is MVSep cloud). The GPU pack (CUDA/cuDNN/ORT CUDA
# provider) is not bundled either - GPU acceleration on Linux is available
# via the Flatpak build or a system CUDA/cuDNN install. See src/assets.rs.
MODEL_SOURCE_DIR="$REPO_ROOT/models"
mkdir -p "$MODEL_SOURCE_DIR"

ASSET_BASE="https://github.com/Frantzes/KeyScribe/releases/download/assets-v1"
REQUIRED_MODELS=("beat_this_small.onnx" "mel_spectrogram.onnx" "basic-pitch.onnx")

for MODEL_NAME in "${REQUIRED_MODELS[@]}"; do
    if [ ! -f "$MODEL_SOURCE_DIR/$MODEL_NAME" ]; then
        echo "Downloading $MODEL_NAME from GitHub Releases..."
        if command -v curl >/dev/null 2>&1; then
            curl -fL "$ASSET_BASE/$MODEL_NAME" -o "$MODEL_SOURCE_DIR/$MODEL_NAME"
        elif command -v wget >/dev/null 2>&1; then
            wget -qO "$MODEL_SOURCE_DIR/$MODEL_NAME" "$ASSET_BASE/$MODEL_NAME"
        else
            echo "Error: curl or wget required to download models." >&2
            exit 1
        fi
    fi
done

BUNDLED_MODELS=(
    "basic-pitch.onnx"
    "beat_this_small.onnx"
    "mel_spectrogram.onnx"
    "melody_quantizer.onnx"
    "melody_quantizer.onnx.data"
    "melody_quantizer_v2_seq.onnx"
)
MODEL_FILES=()
for MODEL_NAME in "${BUNDLED_MODELS[@]}"; do
    if [[ -f "$MODEL_SOURCE_DIR/$MODEL_NAME" ]]; then
        MODEL_FILES+=("$MODEL_SOURCE_DIR/$MODEL_NAME")
    fi
done
if [[ ${#MODEL_FILES[@]} -eq 0 ]]; then
    echo "Missing model files in models/" >&2
    exit 1
fi

ARCH_LABEL="$(target_to_arch_label "$TARGET")"
BUNDLE_NAME="keyscribe-linux-$ARCH_LABEL"
BUNDLE_DIR="$REPO_ROOT/$OUTPUT_ROOT/$BUNDLE_NAME"
MODELS_DIR="$BUNDLE_DIR/models"

rm -rf "$BUNDLE_DIR"
mkdir -p "$MODELS_DIR"

cp "$BINARY_PATH" "$BUNDLE_DIR/$BUNDLE_BINARY_NAME"
chmod +x "$BUNDLE_DIR/$BUNDLE_BINARY_NAME"
for MODEL_PATH in "${MODEL_FILES[@]}"; do
    cp "$MODEL_PATH" "$MODELS_DIR/$(basename "$MODEL_PATH")"
done

# --- FFmpeg Bundling ---
FFMPEG_VENDOR_DIR="$REPO_ROOT/vendor/ffmpeg"
FFMPEG_BIN_PATH="$FFMPEG_VENDOR_DIR/ffmpeg"
# Newer BtbN tarballs place the binary at bin/ffmpeg instead of the root.
FFMPEG_BIN_ALT="$FFMPEG_VENDOR_DIR/bin/ffmpeg"

if [[ ! -f "$FFMPEG_BIN_PATH" ]]; then
    echo "FFmpeg not found in vendor/ffmpeg. Downloading static build..."
    mkdir -p "$FFMPEG_VENDOR_DIR"

    FFMPEG_TAR="$FFMPEG_VENDOR_DIR/ffmpeg.tar.xz"
    # Using the BtbN GPL static build for Linux x64
    FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"

    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$FFMPEG_TAR" "$FFMPEG_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$FFMPEG_TAR" "$FFMPEG_URL"
    else
        echo "Error: curl or wget is required to download FFmpeg." >&2
        exit 1
    fi

    echo "Extracting FFmpeg..."
    # Extract just the ffmpeg binary from the tarball
    # The tarball has a top-level directory like ffmpeg-7.1-amd64-static/
    tar -xJf "$FFMPEG_TAR" -C "$FFMPEG_VENDOR_DIR" --strip-components=1 --wildcards "*/ffmpeg"
    # Newer BtbN builds nest it one level deeper (bin/ffmpeg); hoist it up.
    if [[ ! -f "$FFMPEG_BIN_PATH" && -f "$FFMPEG_BIN_ALT" ]]; then
        mv "$FFMPEG_BIN_ALT" "$FFMPEG_BIN_PATH"
        rmdir "$FFMPEG_VENDOR_DIR/bin" 2>/dev/null || true
    fi

    rm -f "$FFMPEG_TAR"
elif [[ -f "$FFMPEG_BIN_ALT" && ! -f "$FFMPEG_BIN_PATH" ]]; then
    # Already downloaded by an older run of this script (nested bin/ layout).
    mv "$FFMPEG_BIN_ALT" "$FFMPEG_BIN_PATH"
    rmdir "$FFMPEG_VENDOR_DIR/bin" 2>/dev/null || true
fi

if [[ -f "$FFMPEG_BIN_PATH" ]]; then
    cp "$FFMPEG_BIN_PATH" "$BUNDLE_DIR/ffmpeg"
    chmod +x "$BUNDLE_DIR/ffmpeg"
    echo "Included ffmpeg from: $FFMPEG_BIN_PATH"
else
    echo "Warning: Failed to prepare ffmpeg. Video features may not work." >&2
fi

# --- ONNX Runtime core (pinned Microsoft PyPI wheel) ---
# Provides libonnxruntime.so (CPU inference + host for the CUDA provider).
# GPU acceleration on Linux comes from the Flatpak build or a system
# CUDA 12 + cuDNN 9 install; the portable zip is CPU-only.
ORT_VENDOR_DIR="$REPO_ROOT/vendor/ort-core"
ORT_PINNED_WHEEL="onnxruntime_gpu-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
ORT_WHEEL_URL="https://files.pythonhosted.org/packages/d0/2c/5b3fd4748cf7ed291eae541a37e426efc20ea04cb6e6a05768304ab0aa41/$ORT_PINNED_WHEEL"
ORT_WHEEL_SHA="eb0e38f0c1ef3b76ae0081c8e51eed20dd8925aa916f0fc6f9b8b17d05610e99"

if ! ls "$ORT_VENDOR_DIR"/libonnxruntime.so* >/dev/null 2>&1; then
    echo "ONNX Runtime core not found in vendor/ort-core. Downloading pinned 1.24.4 wheel..."
    mkdir -p "$ORT_VENDOR_DIR"
    ORT_WHEEL_PATH="$ORT_VENDOR_DIR/$ORT_PINNED_WHEEL"
    if [[ ! -f "$ORT_WHEEL_PATH" ]]; then
        if command -v curl >/dev/null 2>&1; then
            curl -fL -o "$ORT_WHEEL_PATH" "$ORT_WHEEL_URL"
        elif command -v wget >/dev/null 2>&1; then
            wget -qO "$ORT_WHEEL_PATH" "$ORT_WHEEL_URL"
        else
            echo "Error: curl or wget required to download ONNX Runtime." >&2
            exit 1
        fi
    fi
    if ! echo "$ORT_WHEEL_SHA  $ORT_WHEEL_PATH" | sha256sum -c - >/dev/null 2>&1; then
        rm -f "$ORT_WHEEL_PATH"
        echo "Error: ONNX Runtime wheel checksum mismatch; download discarded." >&2
        exit 1
    fi

    EXTRACT_DIR="$ORT_VENDOR_DIR/extract"
    rm -rf "$EXTRACT_DIR"
    mkdir -p "$EXTRACT_DIR"
    python3 -m zipfile -e "$ORT_WHEEL_PATH" "$EXTRACT_DIR"
    CAPI_DIR="$EXTRACT_DIR/onnxruntime/capi"
    if [[ -d "$CAPI_DIR" ]]; then
        for src in "$CAPI_DIR"/libonnxruntime.so*; do
            if [[ -f "$src" || -L "$src" ]]; then
                cp -P "$src" "$ORT_VENDOR_DIR/"
            fi
        done
        echo "Cached ONNX Runtime core in vendor/ort-core"
    else
        echo "Error: onnxruntime/capi missing from wheel." >&2
        exit 1
    fi
    rm -rf "$EXTRACT_DIR"
    rm -f "$ORT_WHEEL_PATH"
else
    echo "ONNX Runtime core already cached in vendor/ort-core"
fi

# The app loads the exact name `libonnxruntime.so` via ort::init_from.
ORT_VERSIONED=$(ls "$ORT_VENDOR_DIR"/libonnxruntime.so.* 2>/dev/null | grep -v '\.so$' | head -1)
if [[ -n "$ORT_VERSIONED" && ! -e "$ORT_VENDOR_DIR/libonnxruntime.so" ]]; then
    ln -sf "$(basename "$ORT_VERSIONED")" "$ORT_VENDOR_DIR/libonnxruntime.so"
    echo "Created libonnxruntime.so symlink -> $(basename "$ORT_VERSIONED")"
fi

for src in "$ORT_VENDOR_DIR"/libonnxruntime.so*; do
    if [[ -f "$src" || -L "$src" ]]; then
        cp -P "$src" "$BUNDLE_DIR/"
    fi
done

cat > "$BUNDLE_DIR/README-portable.txt" <<'EOF'
KeyScribe portable Linux bundle

Contents:
- keyscribe
- ffmpeg
- models/*.onnx
- libonnxruntime.so (ONNX Runtime core)

Downloaded automatically on first use (no action needed):
- Demucs htdemucs_6s model - only when you run stem separation with a local
  Demucs model (the default MVSep separation runs in the cloud)

GPU acceleration is not bundled in the portable zip: use the Flatpak build
(includes CUDA/cuDNN) or install CUDA 12 + cuDNN 9 system-wide.

All AI inference (note detection, stem separation, beat tracking) runs
in-process via ONNX Runtime - no Python or external runtime required.

Run ./keyscribe from this folder so the relative model, ffmpeg, and ONNX
Runtime paths work.
EOF

mkdir -p "$REPO_ROOT/$OUTPUT_ROOT"
ARCHIVE_PATH="$REPO_ROOT/$OUTPUT_ROOT/$BUNDLE_NAME.zip"
rm -f "$ARCHIVE_PATH"

cd "$BUNDLE_DIR" && zip -r "$ARCHIVE_PATH" .

echo "Portable bundle directory: $BUNDLE_DIR"
echo "Portable bundle archive:   $ARCHIVE_PATH"
