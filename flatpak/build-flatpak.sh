#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────
# KeyScribe Flatpak build script
#
# Prerequisites:
#   flatpak, flatpak-builder
#   org.freedesktop.Platform//24.08
#   org.freedesktop.Sdk//24.08
#   org.freedesktop.Sdk.Extension.rust-stable//24.08
#
# Usage:
#   ./build-flatpak.sh [--install] [--gpu]
#
#   --install   Install the built Flatpak for the local user
#   --gpu       Bundle CUDA runtime from host (/usr/local/cuda-12.x/)
#               Requires CUDA 12 toolkit installed on build host
#   --help      Show this help
#
# The ONNX Runtime GPU .so is always bundled and works for CPU too.
# The app auto-falls back to CPU if the CUDA runtime is not found.
#
# NOTE: The onnxruntime-gpu wheel URL + sha256 are pinned directly in
# com.frantzes.keyscribe.yml, so flatpak-builder can be run directly on the
# manifest without this script. To bump the ONNX Runtime version, update the
# URL and sha256 in the manifest (get them from
# https://pypi.org/pypi/onnxruntime-gpu/<version>/json).
# ──────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$SCRIPT_DIR/com.frantzes.keyscribe.yml"
BUILD_DIR="$REPO_ROOT/build/flatpak"
GPU_MODE=0
INSTALL=0

usage() {
    cat <<'EOF'
Usage: build-flatpak.sh [--gpu] [--install]

  --install   Install the built Flatpak (flatpak-builder --user --install)
  --gpu       Bundle CUDA 12 runtime from build host (/usr/local/cuda-12.x/)
              for NVIDIA GPU acceleration
  --help      Show this help

Notes:
  - ONNX Runtime GPU .so is always bundled; CPU fallback works automatically
  - Without --gpu: CPU inference only (no NVIDIA CUDA needed)
  - With --gpu: requires CUDA 12 toolkit on the build host
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_MODE=1; shift ;;
        --install) INSTALL=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ── Prerequisites ─────────────────────────────────
echo "Checking prerequisites..."

for cmd in flatpak flatpak-builder; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: '$cmd' not found. Install it:" >&2
        echo "  sudo apt install flatpak flatpak-builder" >&2
        exit 1
    fi
done

FLATPAK_RUNTIMES=(
    "org.freedesktop.Platform//24.08"
    "org.freedesktop.Sdk//24.08"
    "org.freedesktop.Sdk.Extension.rust-stable//24.08"
)

for rt in "${FLATPAK_RUNTIMES[@]}"; do
    if ! flatpak info "$rt" &>/dev/null; then
        echo "Installing Flatpak runtime: $rt"
        flatpak install --user --noninteractive flathub "$rt"
    fi
done

# ── GPU mode: check CUDA toolkit ──────────────────
if [[ "$GPU_MODE" -eq 1 ]]; then
    echo "GPU mode enabled. Checking for CUDA 12 toolkit..."

    CUDA_TOOLKIT_DIRS=(
        "/usr/local/cuda-12.6/lib64"
        "/usr/local/cuda-12.5/lib64"
        "/usr/local/cuda-12.4/lib64"
        "/usr/local/cuda/lib64"
    )
    CUDA_FOUND=""
    for dir in "${CUDA_TOOLKIT_DIRS[@]}"; do
        if [[ -f "$dir/libcudart.so.12" ]]; then
            CUDA_FOUND="$dir"
            break
        fi
    done

    if [[ -z "$CUDA_FOUND" ]]; then
        echo "Warning: CUDA 12 toolkit not found on build host." >&2
        echo "  GPU acceleration will not be available. Install CUDA 12" >&2
        echo "  at one of: ${CUDA_TOOLKIT_DIRS[*]}" >&2
    else
        echo "  CUDA toolkit found: $CUDA_FOUND"
    fi
fi

# ── Flatpak build ─────────────────────────────────
echo ""
echo "Building Flatpak..."
echo "  Manifest: $MANIFEST"
echo "  Build dir: $BUILD_DIR"

INSTALL_ARGS=""
if [[ "$INSTALL" -eq 1 ]]; then
    INSTALL_ARGS="--user --install"
fi

# Use --disable-rofiles-fuse when running inside WSL or other
# environments where FUSE mounts are unavailable.
DISABLE_ROFILES=""
if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    DISABLE_ROFILES="--disable-rofiles-fuse"
fi

flatpak-builder \
    --force-clean \
    --ccache \
    --keep-build-dirs \
    $DISABLE_ROFILES \
    $INSTALL_ARGS \
    "$BUILD_DIR" \
    "$MANIFEST"

echo ""
echo "Build complete!"
echo "  Flatpak build dir: $BUILD_DIR"

if [[ "$INSTALL" -eq 1 ]]; then
    echo "  Installed as: com.frantzes.keyscribe"
    echo "  Run: flatpak run com.frantzes.keyscribe"
else
    echo "  Install: flatpak-builder --user --install $BUILD_DIR $MANIFEST"
    echo "  Bundle:  flatpak build-bundle $BUILD_DIR keyscribe.flatpak com.frantzes.keyscribe"
fi
