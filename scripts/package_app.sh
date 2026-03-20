#!/usr/bin/env bash
# package_app.sh — Build a fully self-contained VaniLipi.app for macOS Apple Silicon.
#
# Uses the existing pyenv Python at ~/.pyenv/versions/3.11.9 — no internet required.
# Bundles: Python runtime + stdlib, Homebrew dylibs, venv deps, ML models, frontend.
#
# Prerequisites (build machine only):
#   • macOS 13+ on Apple Silicon
#   • ~/.pyenv/versions/3.11.9  (arm64 Python 3.11)
#   • ./venv/                   (project virtualenv with all deps installed)
#   • models/asr/whisper-large-v3/       (local ASR model weights)
#   • models/translation/indictrans2-1b/ (local translation model weights)
#   • frontend/dist/             (built React app — auto-built if missing)
#
# Output:
#   dist/VaniLipi.app              — standalone app (test in-place)
#   dist/VaniLipi-<version>-arm64.dmg — distributable disk image
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="1.0"
APP_NAME="VaniLipi"
DIST="$REPO/dist"

# Build in /tmp to avoid macOS auto-quarantining the .app while we write into it
BUILD_DIR="/tmp/vanilipi_build_$$"
APP_DIR="$BUILD_DIR/${APP_NAME}.app"
CONTENTS="$APP_DIR/Contents"
RESOURCES="$CONTENTS/Resources"
MACOS_BIN="$CONTENTS/MacOS"

FINAL_APP_DIR="$DIST/${APP_NAME}.app"

PYBASE="$HOME/.pyenv/versions/3.11.9"
VENV="$REPO/venv"

# Dylibs to bundle: libpython itself + Homebrew deps
HOMEBREW_DYLIBS=(
    "$PYBASE/lib/libpython3.11.dylib"
    "/opt/homebrew/opt/gettext/lib/libintl.8.dylib"
    "/opt/homebrew/opt/libomp/lib/libomp.dylib"
    "/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib"
    "/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib"
    "/opt/homebrew/opt/readline/lib/libreadline.8.dylib"
    "/opt/homebrew/opt/tcl-tk/lib/libtcl8.6.dylib"
    "/opt/homebrew/opt/tcl-tk/lib/libtk8.6.dylib"
    "/opt/homebrew/opt/xz/lib/liblzma.5.dylib"
)

# ── Pre-flight checks ─────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════╗"
echo "║       VaniLipi Standalone Packager           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: This script must run on Apple Silicon (arm64). Got: $(uname -m)"
    exit 1
fi

if [[ ! -x "$PYBASE/bin/python3.11" ]]; then
    echo "ERROR: Python not found at $PYBASE/bin/python3.11"
    echo "Install with: pyenv install 3.11.9"
    exit 1
fi

if [[ ! -d "$VENV/lib/python3.11/site-packages" ]]; then
    echo "ERROR: venv not found at $VENV"
    echo "Run: python3.11 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

for model_path in "models/asr/whisper-large-v3" "models/translation/indictrans2-1b"; do
    if [[ ! -d "$REPO/$model_path" ]]; then
        echo "ERROR: Model directory missing: $REPO/$model_path"
        exit 1
    fi
done

if [[ ! -f "$REPO/frontend/dist/index.html" ]]; then
    echo "Building frontend..."
    (cd "$REPO/frontend" && npm run build)
fi

echo "✓ Pre-flight checks passed"
echo ""

# ── Clean slate ───────────────────────────────────────────────────────────────

mkdir -p "$DIST"
rm -rf "$BUILD_DIR" "$FINAL_APP_DIR"
mkdir -p "$BUILD_DIR" "$MACOS_BIN" "$RESOURCES"

# ── Bundle Python runtime ─────────────────────────────────────────────────────
# Copy stdlib (without site-packages — we'll add venv deps separately).
# Skip test/ and __pycache__ to save ~200MB.

echo "=== Bundling Python runtime ==="
BUNDLE_PY="$RESOURCES/python"
mkdir -p "$BUNDLE_PY/bin" "$BUNDLE_PY/lib"

# Python binary
cp "$PYBASE/bin/python3.11" "$BUNDLE_PY/bin/python3"
chmod +x "$BUNDLE_PY/bin/python3"

# Stdlib — tar with exclusions (avoids extended-attr issues on protected pyenv test files)
mkdir -p "$BUNDLE_PY/lib/python3.11"
(cd "$PYBASE/lib" && \
 tar -c \
     --exclude='python3.11/site-packages' \
     --exclude='python3.11/test' \
     --exclude='*/__pycache__' \
     --exclude='*.pyc' \
     python3.11 \
 | tar -x -C "$BUNDLE_PY/lib" --no-same-owner 2>/dev/null || true)
# Strip extended attributes (quarantine flags etc.) and ensure write access
xattr -cr "$BUNDLE_PY" 2>/dev/null || true
chmod -R u+rwX "$BUNDLE_PY"

# lib-dynload (.so extension modules) was already included in the rsync above.

echo "✓ Python runtime copied ($(du -sh "$BUNDLE_PY" | cut -f1))"
echo ""

# ── Bundle venv site-packages ─────────────────────────────────────────────────

echo "=== Bundling Python dependencies (venv site-packages) ==="
SITE="$BUNDLE_PY/lib/python3.11/site-packages"
mkdir -p "$SITE"
(cd "$VENV/lib/python3.11" && \
 tar -c \
     --exclude='*/__pycache__' \
     --exclude='*.pyc' \
     site-packages \
 | tar -x -C "$BUNDLE_PY/lib/python3.11" --no-same-owner 2>/dev/null || true)
echo "✓ Dependencies bundled ($(du -sh "$SITE" | cut -f1))"
echo ""

# ── Bundle Homebrew dylibs ────────────────────────────────────────────────────

echo "=== Bundling Homebrew dynamic libraries ==="
LIBS_DIR="$RESOURCES/libs"
mkdir -p "$LIBS_DIR"

for dylib in "${HOMEBREW_DYLIBS[@]}"; do
    if [[ -f "$dylib" ]]; then
        cp "$dylib" "$LIBS_DIR/"
        echo "  ✓ $(basename "$dylib")"
    else
        echo "  WARNING: $dylib not found — skipping"
    fi
done
echo ""

# ── Rewrite dylib paths in Python binary and extension .so files ──────────────
# Replace absolute Homebrew paths with @loader_path relative refs so the app
# works on any machine regardless of whether Homebrew is installed.

echo "=== Rewriting dynamic library paths ==="

# All binaries that may reference Homebrew paths
BINARIES=()
BINARIES+=("$BUNDLE_PY/bin/python3")

# Python extension modules (.so files in lib-dynload and site-packages)
while IFS= read -r -d '' so; do
    BINARIES+=("$so")
done < <(find "$BUNDLE_PY/lib" "$SITE" -name "*.so" -print0 2>/dev/null)

# Also fix the dylibs themselves (they may link to each other)
while IFS= read -r -d '' dylib; do
    BINARIES+=("$dylib")
done < <(find "$LIBS_DIR" -name "*.dylib" -print0 2>/dev/null)

REWRITE_COUNT=0
for bin in "${BINARIES[@]}"; do
    changed=false
    for dylib in "${HOMEBREW_DYLIBS[@]}"; do
        dylib_name="$(basename "$dylib")"
        # Check if this binary references the absolute Homebrew path
        if otool -L "$bin" 2>/dev/null | grep -q "$dylib"; then
            install_name_tool -change "$dylib" "@loader_path/../libs/$dylib_name" "$bin" 2>/dev/null || true
            changed=true
        fi
    done
    if $changed; then
        REWRITE_COUNT=$((REWRITE_COUNT + 1))
    fi
done

echo "✓ Rewrote paths in $REWRITE_COUNT binaries"
echo ""

# ── Bundle ffmpeg/ffprobe ────────────────────────────────────────────────────
# Required for audio format conversion and duration detection.

echo "=== Bundling ffmpeg ==="
FFMPEG_BIN="$RESOURCES/bin"
mkdir -p "$FFMPEG_BIN"

for tool in ffmpeg ffprobe; do
    TOOL_PATH="$(which "$tool" 2>/dev/null || echo "/opt/homebrew/bin/$tool")"
    if [[ -x "$TOOL_PATH" ]]; then
        cp "$TOOL_PATH" "$FFMPEG_BIN/"
        echo "  ✓ $tool"
    else
        echo "  WARNING: $tool not found — audio conversion may fail"
    fi
done
echo ""

# ── App source ────────────────────────────────────────────────────────────────

echo "=== Copying app source ==="
APP_SRC="$RESOURCES/app"
mkdir -p "$APP_SRC"
cp -R "$REPO/backend"          "$APP_SRC/backend"
mkdir -p "$APP_SRC/frontend"
cp -R "$REPO/frontend/dist"    "$APP_SRC/frontend/dist"
echo "✓ App source copied"
echo ""

# ── ML Models ─────────────────────────────────────────────────────────────────

echo "=== Copying ML models (this may take a few minutes) ==="
cp -R "$REPO/models" "$RESOURCES/models"
echo "✓ Models copied"
echo "    ASR:         $(du -sh "$RESOURCES/models/asr" | cut -f1)"
echo "    Translation: $(du -sh "$RESOURCES/models/translation" | cut -f1)"
echo ""

# ── Launcher script ───────────────────────────────────────────────────────────

cat > "$MACOS_BIN/$APP_NAME" << 'LAUNCHER'
#!/usr/bin/env bash
# VaniLipi launcher — runs inside the macOS app bundle.
# Opens a native window via pywebview (WKWebView). No browser required.
set -euo pipefail

BUNDLE="$(cd "$(dirname "$0")/.." && pwd)"
RESOURCES="$BUNDLE/Resources"
PYTHON="$RESOURCES/python/bin/python3"
APP_DIR="$RESOURCES/app"
LIBS_DIR="$RESOURCES/libs"

# Use bundled Python's own stdlib, not any system Python
export PYTHONHOME="$RESOURCES/python"
export PYTHONPATH="$APP_DIR"

# Bundled models (overrides repo-relative default in config.py)
export REPO_MODELS_DIR="$RESOURCES/models"

# Make bundled dylibs findable at runtime
export DYLD_LIBRARY_PATH="$LIBS_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export DYLD_FALLBACK_LIBRARY_PATH="$LIBS_DIR"

# Inhibit OCSP checking issues on machines without internet
export REQUESTS_CA_BUNDLE=""

# Add bundled ffmpeg to PATH
export PATH="$RESOURCES/bin:$PATH"

# Launch native window (pywebview starts server + opens WKWebView)
cd "$APP_DIR"
exec "$PYTHON" -m backend.native 2>>/tmp/vanilipi.log
LAUNCHER

chmod +x "$MACOS_BIN/$APP_NAME"
echo "✓ Launcher script written"

# ── App icon ──────────────────────────────────────────────────────────────────
ICNS_SRC="$REPO/frontend/public/VaniLipi.icns"
if [[ -f "$ICNS_SRC" ]]; then
    cp "$ICNS_SRC" "$RESOURCES/VaniLipi.icns"
    echo "✓ App icon added"
fi

# ── Info.plist ────────────────────────────────────────────────────────────────

cat > "$CONTENTS/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>             <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>      <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>       <string>com.vanilipi.app</string>
    <key>CFBundleVersion</key>          <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key> <string>${VERSION}</string>
    <key>CFBundleExecutable</key>       <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>      <string>APPL</string>
    <key>LSMinimumSystemVersion</key>   <string>13.0</string>
    <key>NSHighResolutionCapable</key>  <true/>
    <key>CFBundleIconFile</key>       <string>VaniLipi</string>
    <key>NSMicrophoneUsageDescription</key>
        <string>VaniLipi needs microphone access for live recording.</string>
</dict>
</plist>
PLIST

echo "✓ Info.plist written"

# ── Move app to final location ────────────────────────────────────────────────

echo ""
echo "=== Finalizing app bundle ==="
xattr -cr "$APP_DIR" 2>/dev/null || true
mv "$APP_DIR" "$FINAL_APP_DIR"
rm -rf "$BUILD_DIR"
APP_DIR="$FINAL_APP_DIR"
echo "✓ App moved to $APP_DIR"

# ── Remove quarantine + ad-hoc sign ──────────────────────────────────────────

echo ""
echo "=== Code signing (ad-hoc) ==="
codesign --deep --force --sign - "$APP_DIR" 2>/dev/null \
    && echo "✓ Ad-hoc signed" \
    || echo "WARNING: codesign failed (app may still work — right-click → Open to bypass Gatekeeper)"

# ── Build DMG ────────────────────────────────────────────────────────────────

echo ""
echo "=== Building DMG ==="
DMG_NAME="${APP_NAME}-${VERSION}-arm64.dmg"
DMG_PATH="$DIST/$DMG_NAME"
STAGE="/tmp/vanilipi_dmg_stage_$$"

rm -f "$DMG_PATH"
mkdir -p "$STAGE"
ditto "$APP_DIR" "$STAGE/${APP_NAME}.app"
ln -s /Applications "$STAGE/Applications"

hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$STAGE" \
    -ov \
    -format ULFO \
    "$DMG_PATH"

rm -rf "$STAGE"

# ── Summary ───────────────────────────────────────────────────────────────────

APP_SIZE=$(du -sh "$APP_DIR" | cut -f1)
DMG_SIZE=$(du -sh "$DMG_PATH" | cut -f1)

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  Build complete!                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  App:  $APP_DIR"
echo "        Size: $APP_SIZE"
echo ""
echo "  DMG:  $DMG_PATH"
echo "        Size: $DMG_SIZE"
echo ""
echo "─── To install ────────────────────────────────────────────────"
echo "  1. Mount the DMG (double-click)"
echo "  2. Drag VaniLipi → Applications"
echo "  3. Right-click → Open (first launch, to bypass Gatekeeper)"
echo ""
echo "─── To share ──────────────────────────────────────────────────"
echo "  AirDrop, USB drive, or cloud storage (file is ~${DMG_SIZE})"
echo ""
