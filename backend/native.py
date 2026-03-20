"""
VaniLipi native macOS window launcher.

Starts the FastAPI server as a separate subprocess and opens a native macOS
window (WKWebView) via pywebview. No browser chrome — the app looks and
behaves like a standard desktop application.

Architecture:
  - Server runs as a child process (separate memory space for ML models)
  - pywebview runs on the main thread (macOS requirement for AppKit)
  - When the window closes, the server subprocess is terminated

Usage:
    python -m backend.native
"""
import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error

logger = logging.getLogger("vanilipi.native")

DEFAULT_PORT = 7860
PORT_SEARCH_RANGE = 10
WINDOW_TITLE = "VaniLipi"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 820


def _find_free_port(start: int = DEFAULT_PORT, search_range: int = PORT_SEARCH_RANGE) -> int:
    """Find a free TCP port starting from `start`."""
    for port in range(start, start + search_range):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{start + search_range - 1}")


def _wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Block until the server responds to an HTTP GET / or timeout expires."""
    url = f"http://127.0.0.1:{port}/"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, OSError, ConnectionRefusedError):
            pass
        time.sleep(0.3)
    return False


def main() -> None:
    """Entry point: start server subprocess, then open a native window."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    logger.info("Starting VaniLipi server on %s …", url)

    # Start uvicorn as a separate subprocess so ML models get their own
    # memory space and don't compete with the pywebview/WKWebView process.
    # Use APP_DIR env var if set (bundled .app), else fall back to cwd
    app_dir = os.environ.get("PYTHONPATH", os.getcwd())
    # Split PYTHONPATH and use the first entry (the app dir)
    if os.pathsep in app_dir:
        app_dir = app_dir.split(os.pathsep)[0]

    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=app_dir,
        env=os.environ.copy(),  # Critical: inherit DYLD_LIBRARY_PATH, PATH (ffmpeg), REPO_MODELS_DIR
    )

    # Ensure the server subprocess is killed when this process exits
    def _cleanup():
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    if not _wait_for_server(port):
        logger.error("Server failed to start within 30 seconds.")
        _cleanup()
        sys.exit(1)

    logger.info("Server ready. Opening native window…")

    try:
        import webview
    except ImportError:
        logger.error(
            "pywebview is not installed. Install it with: pip install pywebview\n"
            "Falling back to browser mode."
        )
        import webbrowser
        webbrowser.open(url)
        # Keep running until server dies or user interrupts
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            pass
        return

    window = webview.create_window(
        WINDOW_TITLE,
        url,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=(900, 600),
        text_select=True,
    )

    # webview.start() blocks until the window is closed.
    webview.start()

    # Window closed — clean up server
    logger.info("Window closed. Shutting down server…")
    _cleanup()


if __name__ == "__main__":
    main()
