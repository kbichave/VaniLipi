#!/usr/bin/env bash
# VaniLipi installer for Apple Silicon Mac
# Usage: bash scripts/install.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "========================================"
echo "VaniLipi Installer"
echo "========================================"
echo

# Check macOS + Apple Silicon
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: VaniLipi requires macOS (Apple Silicon)." >&2
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "Warning: VaniLipi is optimized for Apple Silicon (M1/M2/M3/M4)."
    echo "It will run on Intel Macs but will be significantly slower."
fi

# Check Python 3.11+
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
    echo "Error: Python 3.11+ required. Found: $PYTHON_VERSION" >&2
    echo "Install via: brew install python@3.11" >&2
    exit 1
fi
echo "✓ Python $PYTHON_VERSION"

# Create venv if needed
if [[ ! -d "$REPO_DIR/venv" ]]; then
    echo "Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$REPO_DIR/venv"
fi

# Activate venv
source "$REPO_DIR/venv/bin/activate"
echo "✓ Virtual environment: $REPO_DIR/venv"

# Install/upgrade pip
python -m pip install --upgrade pip --quiet

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# Verify key packages
python -c "import mlx; print('✓ MLX:', mlx.__version__)" 2>/dev/null || echo "⚠ MLX not available (install failed or not Apple Silicon)"
python -c "import mlx_whisper; print('✓ mlx-whisper: available')" 2>/dev/null || echo "⚠ mlx-whisper not available"

# Make scripts executable
chmod +x "$REPO_DIR/scripts/launch.sh"
chmod +x "$REPO_DIR/scripts/create_app.sh"

echo
echo "========================================"
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Start the app:     bash scripts/launch.sh"
echo "  2. Create .app icon:  bash scripts/create_app.sh"
echo "========================================"
