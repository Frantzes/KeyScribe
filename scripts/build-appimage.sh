#!/usr/bin/env bash
#
# KeyScribe AppImage build script (x86_64 Linux)
#
# Assembles an AppDir from the portable bundle produced by
# scripts/build-linux.sh and packs it with appimagetool.
#
# Usage:
#   ./scripts/build-appimage.sh [--skip-bundle] [--output <file>]
#
#   --skip-bundle   Reuse the existing build/linux/keyscribe-linux-x64 dir
#                   instead of re-running scripts/build-linux.sh
#   --output <file> Output AppImage path
#                   (default: build/linux/KeyScribe-x86_64.AppImage)
#
# Prerequisites: curl or wget (to fetch appimagetool).
# The AppImage contains the keyscribe binary, bundled ffmpeg, ONNX models,
# ONNX Runtime + CUDA 12 / cuDNN 9 libs (GPU accel where available, CPU
# fallback otherwise). No installation needed — just run the .AppImage.
#
# NOTE on distro compatibility: the bundled binary is built on the host
# distro, so its glibc floor is the host's glibc (check with ldd --version).
# For maximum compatibility across older distros, run this script inside an
# older container (e.g. Ubuntu 22.04) instead.

set -euo pipefail

SKIP_BUNDLE=0
OUTPUT=""

usage() {
    cat <<'EOF'
Usage: scripts/build-appimage.sh [options]

Options:
  --skip-bundle   Reuse existing build/linux/keyscribe-linux-x64
  --output <file> Output AppImage path
                  (default: build/linux/KeyScribe-x86_64.AppImage)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-bundle)
            SKIP_BUNDLE=1
            shift
            ;;
        --output)
            [[ $# -ge 2 ]] || { echo "Missing value for --output" >&2; exit 1; }
            OUTPUT="$2"
            shift 2
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUNDLE_DIR="$REPO_ROOT/build/linux/keyscribe-linux-x64"

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$REPO_ROOT/build/linux/KeyScribe-x86_64.AppImage"
fi

if [[ "$SKIP_BUNDLE" -eq 0 ]]; then
    echo "Building portable bundle..."
    bash "$SCRIPT_DIR/build-linux.sh"
fi

if [[ ! -x "$BUNDLE_DIR/keyscribe" ]]; then
    echo "Error: $BUNDLE_DIR/keyscribe not found. Run scripts/build-linux.sh first." >&2
    exit 1
fi

# ── appimagetool ──────────────────────────────────
TOOLS_DIR="$REPO_ROOT/build/tools"
mkdir -p "$TOOLS_DIR"
APPIMAGETOOL="$TOOLS_DIR/appimagetool-x86_64.AppImage"
if [[ ! -x "$APPIMAGETOOL" ]]; then
    echo "Downloading appimagetool..."
    APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
    if command -v curl >/dev/null 2>&1; then
        curl -fL -o "$APPIMAGETOOL" "$APPIMAGETOOL_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$APPIMAGETOOL" "$APPIMAGETOOL_URL"
    else
        echo "Error: curl or wget required to download appimagetool." >&2
        exit 1
    fi
    chmod +x "$APPIMAGETOOL"
fi

# ── AppDir ────────────────────────────────────────
APPDIR="$REPO_ROOT/build/linux/KeyScribe.AppDir"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/256x256/apps"

echo "Assembling AppDir..."
cp -a "$BUNDLE_DIR/keyscribe" "$APPDIR/usr/bin/"
cp -a "$BUNDLE_DIR/ffmpeg" "$APPDIR/usr/bin/" 2>/dev/null || true
cp -a "$BUNDLE_DIR"/lib*.so* "$APPDIR/usr/bin/"
cp -a "$BUNDLE_DIR/models" "$APPDIR/usr/bin/"

# Bundle libasound (ALSA) from the build environment so the AppImage plays
# audio on minimal systems. X11/Wayland/GL/OpenSSL intentionally stay on the
# host (all desktop distros ship them; bundling them breaks drivers).
if command -v ldconfig >/dev/null 2>&1; then
    ASOUND_REAL="$(ldconfig -p 2>/dev/null | grep -m1 'libasound\.so\.2 ' | awk '{print $NF}')"
    if [[ -n "${ASOUND_REAL:-}" && -f "$ASOUND_REAL" ]]; then
        ASOUND_DIR="$(dirname "$ASOUND_REAL")"
        cp -aP "$ASOUND_DIR"/libasound.so.2* "$APPDIR/usr/bin/" 2>/dev/null || true
        echo "Bundled ALSA lib from: $ASOUND_REAL"
    fi
fi

cp -a "$REPO_ROOT/flatpak/com.frantzes.keyscribe.desktop" \
      "$APPDIR/usr/share/applications/"
cp -a "$REPO_ROOT/icon.png" \
      "$APPDIR/usr/share/icons/hicolor/256x256/apps/com.frantzes.keyscribe.png"
cp -a "$REPO_ROOT/icon.png" "$APPDIR/.DirIcon"
ln -sf "usr/share/applications/com.frantzes.keyscribe.desktop" \
       "$APPDIR/com.frantzes.keyscribe.desktop"
ln -sf "usr/share/icons/hicolor/256x256/apps/com.frantzes.keyscribe.png" \
       "$APPDIR/com.frantzes.keyscribe.png"

# AppRun: everything the app needs sits in usr/bin (binary, .so files,
# ffmpeg, models/). Point the loader there and launch.
cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "${0}")")"
export LD_LIBRARY_PATH="$HERE/usr/bin:${LD_LIBRARY_PATH:-}"
# Prefer bundled ffmpeg (same dir as the binary).
export PATH="$HERE/usr/bin:${PATH}"
exec "$HERE/usr/bin/keyscribe" "$@"
EOF
chmod +x "$APPDIR/AppRun"

# ── Pack ──────────────────────────────────────────
echo "Packing AppImage..."
ARCH=x86_64 "$APPIMAGETOOL" "$APPDIR" "$OUTPUT"

echo ""
echo "AppImage: $OUTPUT"
ls -lh "$OUTPUT"
