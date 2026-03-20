#!/usr/bin/env bash
# Create VaniLipi.app — a native macOS application bundle.
# Double-clicking the .app launches a native window via pywebview (WKWebView).
#
# Usage: bash scripts/create_app.sh
# Output: ~/Desktop/VaniLipi.app
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="VaniLipi"
DESKTOP_APP="$HOME/Desktop/${APP_NAME}.app"
APPLICATIONS_APP="/Applications/${APP_NAME}.app"

echo "Creating $APP_NAME.app…"

# Build the .app bundle structure
rm -rf "$DESKTOP_APP"
mkdir -p "$DESKTOP_APP/Contents/MacOS"
mkdir -p "$DESKTOP_APP/Contents/Resources"

# Write Info.plist
cat > "$DESKTOP_APP/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>VaniLipi</string>
    <key>CFBundleIdentifier</key>
    <string>com.vanilipi.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>VaniLipi</string>
    <key>CFBundleDisplayName</key>
    <string>VaniLipi</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleIconFile</key>
    <string>VaniLipi</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>VaniLipi needs microphone access to record audio for transcription.</string>
</dict>
</plist>
PLIST

# Write the launcher executable
LAUNCH_SCRIPT="$DESKTOP_APP/Contents/MacOS/$APP_NAME"
cat > "$LAUNCH_SCRIPT" << SCRIPT
#!/usr/bin/env bash
# VaniLipi native launcher — starts the backend and opens a WKWebView window.
REPO_DIR="$REPO_DIR"
exec bash "\$REPO_DIR/scripts/launch.sh"
SCRIPT

chmod +x "$LAUNCH_SCRIPT"

# Copy app icon
ICNS_SRC="$REPO_DIR/frontend/public/VaniLipi.icns"
if [[ -f "$ICNS_SRC" ]]; then
    cp "$ICNS_SRC" "$DESKTOP_APP/Contents/Resources/VaniLipi.icns"
    echo "✓ App icon added"
fi

echo "✓ Created: $DESKTOP_APP"

# Optionally copy to /Applications
if [[ -w "/Applications" ]]; then
    rm -rf "$APPLICATIONS_APP"
    cp -R "$DESKTOP_APP" "$APPLICATIONS_APP" 2>/dev/null && echo "✓ Copied to /Applications"
fi

# Remove quarantine attribute so macOS doesn't block it
xattr -cr "$DESKTOP_APP" 2>/dev/null || true

echo
echo "Done! Double-click $APP_NAME.app on your Desktop to launch VaniLipi."
echo "The app opens a native macOS window — no browser required."
echo
echo "Note: On first launch, macOS may ask for permission to run the app."
echo "If blocked: System Settings → Privacy & Security → Open Anyway"
