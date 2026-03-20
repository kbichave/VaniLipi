#!/usr/bin/env bash
# VaniLipi launcher — opens a native macOS window (pywebview + WKWebView).
# Usage: bash scripts/launch.sh
# This script is also called by VaniLipi.app.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

VENV="$REPO_DIR/venv"

# Check venv exists
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "Error: Virtual environment not found. Run: bash scripts/install.sh" >&2
    exit 1
fi

source "$VENV/bin/activate"

# Check backend is importable
if ! python -c "import backend" 2>/dev/null; then
    echo "Error: backend package not importable. Make sure you're in the project root." >&2
    exit 1
fi

echo "Launching VaniLipi…"

# Launch native window (pywebview starts server + opens WKWebView window)
exec python -m backend.native
