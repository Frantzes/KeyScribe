#!/usr/bin/env bash

set -euo pipefail

TARGET=""
OUTPUT_ROOT="build/macos"
SKIP_CARGO_BUILD=0
PORTABLE_ONLY=0
BUNDLE_ID="${BUNDLE_ID:-com.frantzes.keyscribe}"
APP_VERSION_INPUT="${APP_VERSION:-}"
APP_BUILD_INPUT="${APP_BUILD:-}"
DMG_ONLY="${DMG_ONLY:-0}"

usage() {
    cat <<'EOF'
Usage: scripts/build-macos.sh [options]

Options:
  --target <triple>     Rust target triple (default: host macOS target)
  --output-root <path>  Output folder root (default: build/macos)
  --skip-cargo-build    Skip cargo build step
    --portable-only       Build only portable zip (skip .app/.pkg/.dmg artifacts)
  -h, --help            Show this help

Environment overrides:
    APP_VERSION           App version (default: tag name or Cargo.toml version)
    APP_BUILD             CFBundleVersion build string (default: APP_VERSION)
    BUNDLE_ID             macOS bundle/package identifier
    DMG_ONLY              Set to 1 to skip portable zip and only build dmg app installer
EOF
}

detect_host_target() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    if [[ "$os" != "Darwin" ]]; then
        echo "This script must run on macOS (Darwin)." >&2
        exit 1
    fi

    case "$arch" in
        arm64|aarch64)
            echo "aarch64-apple-darwin"
            ;;
        x86_64)
            echo "x86_64-apple-darwin"
            ;;
        *)
            echo "Unsupported macOS architecture: $arch" >&2
            exit 1
            ;;
    esac
}

target_to_arch_label() {
    case "$1" in
        aarch64-apple-darwin)
            echo "arm64"
            ;;
        x86_64-apple-darwin)
            echo "x64"
            ;;
        *)
            echo "$1" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9._-' '-'
            ;;
    esac
}

normalize_version_string() {
    local raw="${1:-}"
    raw="${raw#refs/tags/}"
    raw="${raw#v}"
    raw="$(echo "$raw" | sed -E 's/[^0-9.]+/./g; s/[.]+/./g; s/^\.//; s/\.$//')"

    if [[ -z "$raw" ]]; then
        raw="0.1.0"
    fi

    echo "$raw"
}

read_cargo_version() {
    awk -F'"' '/^version[[:space:]]*=/{print $2; exit}' "$REPO_ROOT/Cargo.toml" 2>/dev/null || true
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
        --portable-only)
            PORTABLE_ONLY=1
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

if [[ -z "$APP_VERSION_INPUT" ]]; then
    APP_VERSION_INPUT="${GITHUB_REF_NAME:-}"
fi
if [[ -z "$APP_VERSION_INPUT" ]]; then
    APP_VERSION_INPUT="$(read_cargo_version)"
fi

APP_VERSION="$(normalize_version_string "$APP_VERSION_INPUT")"

if [[ -z "$APP_BUILD_INPUT" ]]; then
    APP_BUILD_INPUT="$APP_VERSION"
fi

APP_BUILD="$(normalize_version_string "$APP_BUILD_INPUT")"

if [[ "$SKIP_CARGO_BUILD" -eq 0 ]]; then
    echo "Building release binary for $TARGET..."
    cargo build --release --target "$TARGET"
fi

echo "Packaging app metadata: version=$APP_VERSION build=$APP_BUILD bundle_id=$BUNDLE_ID"

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

MODEL_PATH="$REPO_ROOT/models/basic-pitch.onnx"
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Missing model file: models/basic-pitch.onnx" >&2
    exit 1
fi

ARCH_LABEL="$(target_to_arch_label "$TARGET")"
BUNDLE_NAME="keyscribe-macos-$ARCH_LABEL"
if [[ "$DMG_ONLY" -eq 0 ]]; then
    BUNDLE_DIR="$REPO_ROOT/$OUTPUT_ROOT/$BUNDLE_NAME"
    MODELS_DIR="$BUNDLE_DIR/models"

    rm -rf "$BUNDLE_DIR"
    mkdir -p "$MODELS_DIR"

    cp "$BINARY_PATH" "$BUNDLE_DIR/$BUNDLE_BINARY_NAME"
    chmod +x "$BUNDLE_DIR/$BUNDLE_BINARY_NAME"
    cp "$MODEL_PATH" "$MODELS_DIR/basic-pitch.onnx"

    cat > "$BUNDLE_DIR/README-portable.txt" <<'EOF'
KeyScribe portable macOS bundle

Contents:
- keyscribe
- models/basic-pitch.onnx

Run ./keyscribe from this folder so relative model path works.
If Gatekeeper blocks launch, run:
xattr -dr com.apple.quarantine ./keyscribe
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
fi

if [[ "$PORTABLE_ONLY" -eq 0 ]]; then
    APP_STAGE_DIR="$REPO_ROOT/$OUTPUT_ROOT/$BUNDLE_NAME-app"
    APP_BUNDLE_DIR="$APP_STAGE_DIR/KeyScribe.app"
    APP_CONTENTS_DIR="$APP_BUNDLE_DIR/Contents"
    APP_MACOS_DIR="$APP_CONTENTS_DIR/MacOS"
    APP_RESOURCES_DIR="$APP_CONTENTS_DIR/Resources"
    APP_MODELS_DIR="$APP_RESOURCES_DIR/models"

    rm -rf "$APP_STAGE_DIR"
    mkdir -p "$APP_MACOS_DIR" "$APP_MODELS_DIR"

    cp "$BINARY_PATH" "$APP_MACOS_DIR/keyscribe-bin"
    chmod +x "$APP_MACOS_DIR/keyscribe-bin"
    cp "$MODEL_PATH" "$APP_MODELS_DIR/basic-pitch.onnx"

    if [[ -f "$REPO_ROOT/icon.png" ]]; then
        cp "$REPO_ROOT/icon.png" "$APP_RESOURCES_DIR/AppIcon.png"
    fi

    cat > "$APP_MACOS_DIR/KeyScribe" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="$SCRIPT_DIR/../Resources"

cd "$RESOURCES_DIR"
exec "$SCRIPT_DIR/keyscribe-bin" "$@"
EOF
    chmod +x "$APP_MACOS_DIR/KeyScribe"

    cat > "$APP_CONTENTS_DIR/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>KeyScribe</string>
    <key>CFBundleExecutable</key>
    <string>KeyScribe</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>KeyScribe</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$APP_VERSION</string>
    <key>CFBundleVersion</key>
    <string>$APP_BUILD</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
</dict>
</plist>
EOF

    # DMG-only mode: keep legacy app-zip/package install paths commented for quick restore.
    # APP_ARCHIVE_PATH="$REPO_ROOT/$OUTPUT_ROOT/$BUNDLE_NAME-app.zip"
    # rm -f "$APP_ARCHIVE_PATH"
    # ditto -c -k --sequesterRsrc --keepParent "$APP_BUNDLE_DIR" "$APP_ARCHIVE_PATH"

    # PKG_STAGE_DIR="$APP_STAGE_DIR/pkg-root"
    # rm -rf "$PKG_STAGE_DIR"
    # mkdir -p "$PKG_STAGE_DIR"
    # cp -R "$APP_BUNDLE_DIR" "$PKG_STAGE_DIR/KeyScribe.app"

    # PKG_PATH="$REPO_ROOT/$OUTPUT_ROOT/$BUNDLE_NAME.pkg"
    # rm -f "$PKG_PATH"
    # pkgbuild \
    #     --root "$PKG_STAGE_DIR" \
    #     --identifier "$BUNDLE_ID" \
    #     --version "$APP_VERSION" \
    #     --install-location "/Applications" \
    #     "$PKG_PATH"

    DMG_STAGE_DIR="$APP_STAGE_DIR/dmg-root"
    rm -rf "$DMG_STAGE_DIR"
    mkdir -p "$DMG_STAGE_DIR"
    cp -R "$APP_BUNDLE_DIR" "$DMG_STAGE_DIR/KeyScribe.app"
    ln -sfn /Applications "$DMG_STAGE_DIR/Applications"

    DMG_PATH="$REPO_ROOT/$OUTPUT_ROOT/$BUNDLE_NAME.dmg"
    rm -f "$DMG_PATH"
    hdiutil create \
        -volname "KeyScribe" \
        -srcfolder "$DMG_STAGE_DIR" \
        -ov \
        -format UDZO \
        "$DMG_PATH"

    echo "App bundle directory:      $APP_BUNDLE_DIR"
    # echo "App bundle archive:        $APP_ARCHIVE_PATH"
    # echo "Installer package:         $PKG_PATH"
    echo "Drag-and-drop disk image:  $DMG_PATH"
fi

