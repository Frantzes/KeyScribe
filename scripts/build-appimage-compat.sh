#!/usr/bin/env bash
#
# KeyScribe max-compatibility AppImage build (Ubuntu 22.04 container).
#
# The native AppImage (scripts/build-appimage.sh) inherits the host's glibc
# (e.g. Arch glibc 2.44) and only runs on distros that new. This script
# builds the whole bundle inside an Ubuntu 22.04 container (glibc 2.35) so
# the resulting AppImage runs on Ubuntu 22.04+, Debian 12+, Fedora 38+,
# Arch, and other modern x86_64 distros.
#
# Usage:
#   ./scripts/build-appimage-compat.sh [--output <file>]
#
#   --output <file> Output AppImage path
#                   (default: build/linux/KeyScribe-x86_64-compat.AppImage)
#
# Prerequisites: docker (user must be able to run containers).
#
# What it does:
#   1. Starts an ubuntu:22.04 container with the repo mounted at /work.
#   2. Installs build deps (compilers, X11/Wayland/ALSA headers, Python...).
#   3. Installs Rust via rustup (fresh toolchain for glibc 2.35).
#   4. Runs scripts/build-linux.sh + scripts/build-appimage.sh inside.
#   5. Leaves the AppImage at build/linux/KeyScribe-x86_64-compat.AppImage.
#
# The host's downloaded assets (models/, vendor/) are reused from the mount
# so nothing downloads twice. The container uses its own CARGO_TARGET_DIR
# (/tmp/ktarget) so the host's target/ dir is untouched.

set -euo pipefail

OUTPUT="build/linux/KeyScribe-x86_64-compat.AppImage"

usage() {
    cat <<'EOF'
Usage: scripts/build-appimage-compat.sh [options]

Options:
  --output <file> Output AppImage path
                  (default: build/linux/KeyScribe-x86_64-compat.AppImage)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker not found." >&2
    exit 1
fi

# Resolve output to an in-repo path the container can write via the mount.
case "$OUTPUT" in
    /*) OUTPUT_IN_REPO_CHECK="$OUTPUT" ;;
    *) OUTPUT_IN_REPO_CHECK="$REPO_ROOT/$OUTPUT" ;;
esac
if [[ "$OUTPUT_IN_REPO_CHECK" != "$REPO_ROOT"* ]]; then
    echo "Error: --output must be inside the repo (container writes via the /work mount)." >&2
    exit 1
fi
OUTPUT_CONTAINER="/work/${OUTPUT_IN_REPO_CHECK#"$REPO_ROOT"/}"

echo "Building max-compat AppImage in Ubuntu 22.04 container..."
echo "  Output: $OUTPUT_IN_REPO_CHECK"

docker run --rm \
    --device /dev/fuse --cap-add SYS_ADMIN \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -v "$REPO_ROOT:/work" \
    -v "$HOME/.cargo/registry:/tmp/cargo-registry" \
    -e CARGO_HOME=/tmp/cargo-home \
    -e RUSTUP_HOME=/tmp/rustup \
    -e CARGO_TARGET_DIR=/tmp/ktarget \
    -e CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse \
    ubuntu:22.04 bash -c '
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates pkg-config git file \
    clang libclang-dev \
    libasound2-dev libssl-dev \
    libxcb1-dev libx11-dev libxkbcommon-dev libxkbcommon-x11-dev \
    libwayland-dev libegl1-mesa-dev libgl1-mesa-dev \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    python3 unzip zip xz-utils binutils fuse libfuse2
mkdir -p /tmp/cargo-home
if [ -d /tmp/cargo-registry ]; then
    mkdir -p /tmp/cargo-home
    cp -a /tmp/cargo-registry /tmp/cargo-home/registry 2>/dev/null || true
    # verovioxide-sys stashes its compiled C++ (built against the host
    # glibc) at registry/src/target/verovio-cache. Drop it so the container
    # compiles Verovio from source against its own (older) glibc instead of
    # linking host objects (which fail with e.g. undefined __isoc23_*).
    rm -rf /tmp/cargo-home/registry/src/target
fi
curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="/tmp/cargo-home/bin:$PATH"
rustc --version
cd /work
bash scripts/build-linux.sh
bash scripts/build-appimage.sh --skip-bundle --output "$0"
# The container runs as root, which would leave root-owned files on the
# host mount. Hand them back to the invoking user.
chown -R "${HOST_UID:-0}:${HOST_GID:-0}" /work/build
' "$OUTPUT_CONTAINER"

echo ""
echo "Compat AppImage: $OUTPUT_IN_REPO_CHECK"
ls -lh "$OUTPUT_IN_REPO_CHECK"
