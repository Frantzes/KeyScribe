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

cat > "$BUNDLE_DIR/README-portable.txt" <<'EOF'
KeyScribe portable Linux bundle

Contents:
- keyscribe
- models/*.onnx

Run ./keyscribe from this folder so relative model path works.
EOF

mkdir -p "$REPO_ROOT/$OUTPUT_ROOT"
ARCHIVE_PATH="$REPO_ROOT/$OUTPUT_ROOT/$BUNDLE_NAME.zip"
rm -f "$ARCHIVE_PATH"

python3 - "$BUNDLE_DIR" "$ARCHIVE_PATH" <<'PY'
import pathlib
import sys
import zipfile

bundle_dir = pathlib.Path(sys.argv[1])
archive_path = pathlib.Path(sys.argv[2])

with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
    for item in bundle_dir.rglob("*"):
        if item.is_file():
            archive.write(item, item.relative_to(bundle_dir))
PY

echo "Portable bundle directory: $BUNDLE_DIR"
echo "Portable bundle archive:   $ARCHIVE_PATH"
