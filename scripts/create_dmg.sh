#!/usr/bin/env bash
# Create a distributable VaniLipi.dmg for sharing on Apple Silicon Macs.
#
# What the DMG contains:
#   VaniLipi.app       — double-click to launch (requires prior install)
#   Install.command    — double-click first to set up Python deps
#   README.txt         — quick start instructions
#
# Usage:  bash scripts/create_dmg.sh
# Output: dist/VaniLipi-1.0.dmg
#
# Requirements: macOS (uses hdiutil, which is built-in)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="1.0"
DMG_NAME="VaniLipi-${VERSION}"
BUILD_DIR="$REPO_DIR/dist/dmg_staging"
OUTPUT_DMG="$REPO_DIR/dist/${DMG_NAME}.dmg"
APP_NAME="VaniLipi"

echo "=== Building VaniLipi.app ==="
bash "$REPO_DIR/scripts/create_app.sh" 2>/dev/null || true

echo ""
echo "=== Creating DMG staging area ==="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Copy .app from Desktop (created by create_app.sh)
DESKTOP_APP="$HOME/Desktop/${APP_NAME}.app"
if [[ ! -d "$DESKTOP_APP" ]]; then
  echo "ERROR: $DESKTOP_APP not found. Run scripts/create_app.sh first."
  exit 1
fi
cp -R "$DESKTOP_APP" "$BUILD_DIR/${APP_NAME}.app"

# Create Install.command (Terminal script that installs dependencies)
cat > "$BUILD_DIR/Install.command" << 'INSTALL'
#!/usr/bin/env bash
# VaniLipi first-time setup
# Double-click this file to install Python dependencies.
set -e
cd "$(dirname "$0")"

# Find the repo — it must be in the same location as this DMG was extracted from,
# or the user must have cloned the repo themselves.
echo "╔══════════════════════════════════════╗"
echo "║     VaniLipi — First-Time Setup      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check for the repo at common locations
REPO=""
for dir in \
  "$HOME/VaniLipi" \
  "$HOME/Desktop/VaniLipi" \
  "$HOME/Documents/VaniLipi" \
  "$HOME/Projects/VaniLipi"; do
  if [[ -f "$dir/requirements.txt" ]]; then
    REPO="$dir"
    break
  fi
done

if [[ -z "$REPO" ]]; then
  echo "ERROR: Could not find VaniLipi repo."
  echo "Please clone the repo and place it in one of these locations:"
  echo "  ~/VaniLipi, ~/Desktop/VaniLipi, ~/Documents/VaniLipi"
  echo ""
  echo "Then double-click Install.command again."
  exit 1
fi

echo "Found VaniLipi at: $REPO"
echo ""
bash "$REPO/scripts/install.sh"
INSTALL
chmod +x "$BUILD_DIR/Install.command"

# Create README.txt
cat > "$BUILD_DIR/README.txt" << 'README'
VaniLipi — Indian Language Transcription
==========================================

FIRST TIME SETUP (one time only):
  1. Clone or download the VaniLipi repo to your Mac
  2. Double-click "Install.command" in this DMG
     (it will install Python dependencies)
  3. Double-click VaniLipi.app

LAUNCHING (after setup):
  Double-click VaniLipi.app

REQUIREMENTS:
  • Apple Silicon Mac (M1/M2/M3/M4)
  • macOS 12 or later
  • ~8GB free disk space (for ML models)
  • Python 3.11+ (brew install python@3.11)

TROUBLESHOOTING:
  • "App can't be opened": System Settings → Privacy & Security → Open Anyway
  • Port 7860 in use: edit scripts/launch.sh, change the port number
  • Model download fails: check your internet connection, then try again

MORE INFO:
  See README.md in the VaniLipi repository.

LICENSE: MIT
README

echo "=== Creating DMG ==="
mkdir -p "$REPO_DIR/dist"

# Use hdiutil to create a compressed DMG from the staging folder
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$BUILD_DIR" \
  -ov \
  -format UDZO \
  "$OUTPUT_DMG"

# Remove staging
rm -rf "$BUILD_DIR"

echo ""
echo "✓ DMG created: $OUTPUT_DMG"
echo ""
echo "To distribute: share $OUTPUT_DMG"
echo "Users: mount DMG, double-click Install.command once, then open VaniLipi.app"
