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
BINARY_CANDIDATES=(
    "$REPO_ROOT/target/$TARGET/release/$BUNDLE_BINARY_NAME"
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

MODEL_SOURCE_DIR="$REPO_ROOT/models"
mapfile -t MODEL_FILES < <(find "$MODEL_SOURCE_DIR" -maxdepth 1 -type f -name '*.onnx' | sort)
if [ ${#MODEL_FILES[@]} -eq 0 ]; then
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
    
    rm -f "$FFMPEG_TAR"
fi

if [[ -f "$FFMPEG_BIN_PATH" ]]; then
    cp "$FFMPEG_BIN_PATH" "$BUNDLE_DIR/ffmpeg"
    chmod +x "$BUNDLE_DIR/ffmpeg"
    echo "Included ffmpeg from: $FFMPEG_BIN_PATH"
else
    echo "Warning: Failed to prepare ffmpeg. Video features may not work." >&2
fi

# --- CUDA / cuDNN / ONNX Runtime GPU Bundling ---
# The CUDA execution provider (libonnxruntime_providers_cuda.so) is loaded
# by ort at runtime. It needs CUDA 12 runtime .so files and cuDNN 9 .so
# files on the library search path. We bundle them next to the executable
# and preload them at startup (see src/demucs.rs preload_cuda_dylibs).

CUDA_SO_NAMES=(
    "libcudart.so.12"
    "libcublas.so.12"
    "libcublasLt.so.12"
    "libcufft.so.11"
    "libcurand.so.10"
    "libnvrtc.so.12"
)
CUDNN_SO_NAMES=(
    "libcudnn.so.9"
    "libcudnn_graph.so.9"
    "libcudnn_ops.so.9"
    "libcudnn_heuristic.so.9"
    "libcudnn_adv.so.9"
    "libcudnn_cnn.so.9"
    "libcudnn_engines_precompiled.so.9"
    "libcudnn_engines_runtime_compiled.so.9"
)
ORT_SO_PATTERNS=(
    "libonnxruntime.so*"
    "libonnxruntime_providers_cuda.so*"
    "libonnxruntime_providers_shared.so*"
)

# 1) CUDA runtime .so files: copy from local CUDA toolkit install if present.
CUDA_TOOLKIT_DIRS=(
    "/usr/local/cuda-12.6/lib64"
    "/usr/local/cuda-12.5/lib64"
    "/usr/local/cuda-12.4/lib64"
    "/usr/local/cuda-12.3/lib64"
    "/usr/local/cuda-12.2/lib64"
    "/usr/local/cuda-12.1/lib64"
    "/usr/local/cuda/lib64"
)
CUDA_TOOLKIT_LIB=""
for dir in "${CUDA_TOOLKIT_DIRS[@]}"; do
    if [[ -f "$dir/libcudart.so.12" ]]; then
        CUDA_TOOLKIT_LIB="$dir"
        break
    fi
done
if [[ -n "$CUDA_TOOLKIT_LIB" ]]; then
    echo "CUDA runtime found in: $CUDA_TOOLKIT_LIB"
    for so in "${CUDA_SO_NAMES[@]}"; do
        src="$CUDA_TOOLKIT_LIB/$so"
        if [[ -f "$src" ]]; then
            cp -P "$src" "$BUNDLE_DIR/$so"
        fi
    done
    echo "Bundled CUDA runtime .so files"
else
    echo "Warning: CUDA 12 toolkit not found in default paths. GPU acceleration unavailable." >&2
fi

# 2) cuDNN 9 .so files: download from NVIDIA if not already cached in vendor/cudnn.
CUDNN_VENDOR_DIR="$REPO_ROOT/vendor/cudnn"
CUDNN_READY=false
for so in "${CUDNN_SO_NAMES[@]}"; do
    if [[ -f "$CUDNN_VENDOR_DIR/$so" ]]; then
        CUDNN_READY=true
        break
    fi
done
if ! $CUDNN_READY; then
    echo "cuDNN 9 .so files not found in vendor/cudnn. Downloading..."
    mkdir -p "$CUDNN_VENDOR_DIR"
    CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz"
    CUDNN_TAR="$CUDNN_VENDOR_DIR/cudnn.tar.xz"
    if command -v curl >/dev/null 2>&1; then
        curl -L -o "$CUDNN_TAR" "$CUDNN_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$CUDNN_TAR" "$CUDNN_URL"
    else
        echo "Error: curl or wget required to download cuDNN." >&2
    fi
    if [[ -f "$CUDNN_TAR" ]]; then
        echo "Extracting cuDNN..."
        EXTRACT_TMP="$CUDNN_VENDOR_DIR/extract"
        mkdir -p "$EXTRACT_TMP"
        tar -xJf "$CUDNN_TAR" -C "$EXTRACT_TMP"
        CUDNN_LIB_DIR=$(find "$EXTRACT_TMP" -type d -name "lib" | head -1)
        if [[ -n "$CUDNN_LIB_DIR" ]]; then
            for so in "${CUDNN_SO_NAMES[@]}"; do
                src="$CUDNN_LIB_DIR/$so"
                if [[ -f "$src" ]]; then
                    cp -P "$src" "$CUDNN_VENDOR_DIR/$so"
                fi
            done
            echo "Cached cuDNN 9 .so files in vendor/cudnn"
        else
            echo "Warning: Could not locate cuDNN lib directory in archive." >&2
        fi
        rm -rf "$EXTRACT_TMP"
        rm -f "$CUDNN_TAR"
    fi
else
    echo "cuDNN 9 .so files already cached in vendor/cudnn"
fi

# Copy cuDNN .so files into the bundle.
for so in "${CUDNN_SO_NAMES[@]}"; do
    src="$CUDNN_VENDOR_DIR/$so"
    if [[ -f "$src" ]]; then
        cp -P "$src" "$BUNDLE_DIR/$so"
    fi
done

# 3) Download ONNX Runtime GPU build (libonnxruntime.so etc.) from the pip wheel.
ORT_VENDOR_DIR="$REPO_ROOT/vendor/ort-gpu"
if ! ls "$ORT_VENDOR_DIR"/libonnxruntime.so* >/dev/null 2>&1; then
    echo "ONNX Runtime GPU .so files not found in vendor/ort-gpu. Downloading..."
    mkdir -p "$ORT_VENDOR_DIR"
    DOWNLOAD_SUCCESS=false

    # Method 1: pip download from system Python.
    PYTHON_CMD=""
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_CMD="python"
    fi
    if [[ -n "$PYTHON_CMD" ]]; then
        echo "  Trying pip download via $PYTHON_CMD..."
        if $PYTHON_CMD -m pip download onnxruntime-gpu==1.24.2 --no-deps -d "$ORT_VENDOR_DIR" 2>/dev/null; then
            WHL_FILE=$(find "$ORT_VENDOR_DIR" -maxdepth 1 -name "*.whl" | head -1)
            if [[ -n "$WHL_FILE" ]]; then
                DOWNLOAD_SUCCESS=true
            fi
        fi
    fi

    # Method 2: Direct download from PyPI JSON API.
    if ! $DOWNLOAD_SUCCESS; then
        echo "  Trying direct download from PyPI (via JSON API)..."
        PYPI_JSON=""
        if command -v curl >/dev/null 2>&1; then
            PYPI_JSON=$(curl -sL "https://pypi.org/pypi/onnxruntime-gpu/1.24.2/json")
        elif command -v wget >/dev/null 2>&1; then
            PYPI_JSON=$(wget -qO- "https://pypi.org/pypi/onnxruntime-gpu/1.24.2/json")
        fi
        if [[ -n "$PYPI_JSON" ]] && [[ -n "$PYTHON_CMD" ]]; then
            DL_URL=$(echo "$PYPI_JSON" | $PYTHON_CMD -c "
import sys, json
data = json.load(sys.stdin)
for u in data.get('urls', []):
    url = u['url']
    if ('linux_x86_64' in url or 'manylinux' in url) and 'x86_64' in url and u['packagetype'] == 'bdist_wheel':
        print(url)
        break
" 2>/dev/null)
            if [[ -n "$DL_URL" ]]; then
                WHL_FILE="$ORT_VENDOR_DIR/onnxruntime_gpu.linux.whl"
                if command -v curl >/dev/null 2>&1; then
                    curl -sL -o "$WHL_FILE" "$DL_URL"
                elif command -v wget >/dev/null 2>&1; then
                    wget -qO "$WHL_FILE" "$DL_URL"
                fi
                if [[ -f "$WHL_FILE" ]]; then
                    DOWNLOAD_SUCCESS=true
                fi
            fi
        fi
    fi

    if $DOWNLOAD_SUCCESS; then
        echo "  Extracting .so files from wheel..."
        ZIP_FILE="$WHL_FILE.zip"
        mv "$WHL_FILE" "$ZIP_FILE"
        EXTRACT_DIR="$ORT_VENDOR_DIR/extract"
        mkdir -p "$EXTRACT_DIR"
        unzip -q -o "$ZIP_FILE" -d "$EXTRACT_DIR" 2>/dev/null || \
            $PYTHON_CMD -m zipfile -e "$ZIP_FILE" "$EXTRACT_DIR" 2>/dev/null
        CAPI_DIR="$EXTRACT_DIR/onnxruntime/capi"
        if [[ -d "$CAPI_DIR" ]]; then
            for pattern in "${ORT_SO_PATTERNS[@]}"; do
                for src in "$CAPI_DIR"/$pattern; do
                    if [[ -f "$src" ]] || [[ -L "$src" ]]; then
                        cp -P "$src" "$ORT_VENDOR_DIR/"
                    fi
                done
            done
            echo "  Cached ORT 1.24.2 GPU .so files in vendor/ort-gpu"
        else
            echo "  Warning: Could not find onnxruntime/capi in wheel." >&2
        fi
        rm -rf "$EXTRACT_DIR"
        rm -f "$ZIP_FILE"
    else
        echo "  Warning: All download methods failed. GPU acceleration unavailable." >&2
    fi
else
    echo "ORT GPU .so files already cached in vendor/ort-gpu"
fi

# Copy ORT .so files into the bundle (including versioned symlinks).
for pattern in "${ORT_SO_PATTERNS[@]}"; do
    for src in "$ORT_VENDOR_DIR"/$pattern; do
        if [[ -f "$src" ]] || [[ -L "$src" ]]; then
            cp -P "$src" "$BUNDLE_DIR/"
        fi
    done
done

cat > "$BUNDLE_DIR/README-portable.txt" <<'EOF'
KeyScribe portable Linux bundle

Contents:
- keyscribe
- ffmpeg
- models/*.onnx
- CUDA 12 runtime + cuDNN 9 .so files + libonnxruntime_providers_cuda.so (GPU accel)

All AI inference (note detection, stem separation, beat tracking) runs
in-process via ONNX Runtime — no Python or external runtime required.
GPU acceleration requires an NVIDIA GPU with CUDA-capable drivers.

Run ./keyscribe from this folder so the relative model, ffmpeg, and
CUDA/cuDNN .so paths work.
EOF

mkdir -p "$REPO_ROOT/$OUTPUT_ROOT"
ARCHIVE_PATH="$REPO_ROOT/$OUTPUT_ROOT/$BUNDLE_NAME.zip"
rm -f "$ARCHIVE_PATH"

cd "$BUNDLE_DIR" && zip -r "$ARCHIVE_PATH" .

echo "Portable bundle directory: $BUNDLE_DIR"
echo "Portable bundle archive:   $ARCHIVE_PATH"
