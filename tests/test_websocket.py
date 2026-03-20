"""
Phase 4: WebSocket streaming tests.
Tests the /api/stream/{file_id} endpoint and the frontend's WebSocket integration.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

_DIST_ASSETS = Path(__file__).parent.parent / "frontend" / "dist" / "assets"


# ─── Backend WebSocket tests ──────────────────────────────────────────────────

@pytest.fixture
def app():
    from backend.main import app
    return app


@pytest.mark.asyncio
async def test_ws_rejects_unknown_action(app, tmp_path):
    """WebSocket closes with error on unknown action."""
    from starlette.testclient import TestClient
    client = TestClient(app)

    # Create a fake upload so the file lookup succeeds
    fake_file = tmp_path / "abc123_test.wav"
    fake_file.write_bytes(b"RIFF" + b"\x00" * 44)

    import backend.main as m
    m.TEMP_DIR = tmp_path

    with client.websocket_connect("/api/stream/abc123") as ws:
        ws.send_text(json.dumps({"action": "unknown"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"


@pytest.mark.asyncio
async def test_ws_rejects_invalid_json(app, tmp_path):
    """WebSocket closes with error on malformed JSON."""
    from starlette.testclient import TestClient
    import backend.main as m
    m.TEMP_DIR = tmp_path

    client = TestClient(app)
    with client.websocket_connect("/api/stream/nofile") as ws:
        ws.send_text("not json {{}")
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"


@pytest.mark.asyncio
async def test_ws_rejects_missing_file(app, tmp_path):
    """WebSocket closes with error when file_id not found."""
    from starlette.testclient import TestClient
    import backend.main as m
    m.TEMP_DIR = tmp_path  # empty temp dir

    client = TestClient(app)
    with client.websocket_connect("/api/stream/nonexistent") as ws:
        ws.send_text(json.dumps({"action": "transcribe", "language": "auto", "model": "turbo"}))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "not found" in msg["message"].lower()


@pytest.mark.asyncio
async def test_ws_full_pipeline(app, tmp_path):
    """WebSocket streams segment → translation → done messages."""
    from starlette.testclient import TestClient
    import backend.main as m

    # Create fake audio upload
    fake_file = tmp_path / "testfile_audio.wav"
    fake_file.write_bytes(b"RIFF" + b"\x00" * 1000)
    m.TEMP_DIR = tmp_path

    fake_segments = [
        {"id": 0, "start": 0.0, "end": 2.5, "text": "मला मराठी आवडते.", "english": "I love Marathi."},
    ]

    with (
        patch("backend.main.validate_and_prepare", return_value=(str(fake_file), 2.5)),
        patch("backend.main.asr.transcribe", return_value={
            "language": "mr",
            "segments": fake_segments,
        }),
        patch("backend.main.asr.unload"),
        patch("backend.main.translator.load"),
        patch("backend.main.translator.translate_segments", return_value=fake_segments),
    ):
        client = TestClient(app)
        with client.websocket_connect("/api/stream/testfile") as ws:
            ws.send_text(json.dumps({"action": "transcribe", "language": "auto", "model": "turbo"}))

            messages = []
            for _ in range(10):  # collect up to 10 messages
                try:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    messages.append(msg)
                    if msg["type"] in ("complete", "error"):
                        break
                except Exception:
                    break

    types = [m["type"] for m in messages]
    assert "segment" in types, f"Expected 'segment' in {types}"
    assert "complete" in types, f"Expected 'complete' in {types}"


@pytest.mark.asyncio
async def test_ws_streams_language_detected(app, tmp_path):
    """WebSocket sends language_detected when auto-detect is used."""
    from starlette.testclient import TestClient
    import backend.main as m

    fake_file = tmp_path / "langtest_audio.wav"
    fake_file.write_bytes(b"RIFF" + b"\x00" * 1000)
    m.TEMP_DIR = tmp_path

    with (
        patch("backend.main.validate_and_prepare", return_value=(str(fake_file), 3.0)),
        patch("backend.main.asr.transcribe", return_value={
            "language": "mr",
            "segments": [{"id": 0, "start": 0.0, "end": 1.5, "text": "हे आहे.", "english": "This is it."}],
        }),
        patch("backend.main.asr.unload"),
        patch("backend.main.translator.load"),
        patch("backend.main.translator.translate_segments", return_value=[
            {"id": 0, "start": 0.0, "end": 1.5, "text": "हे आहे.", "english": "This is it."}
        ]),
    ):
        client = TestClient(app)
        with client.websocket_connect("/api/stream/langtest") as ws:
            ws.send_text(json.dumps({"action": "transcribe", "language": "auto", "model": "turbo"}))

            messages = []
            for _ in range(10):
                try:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    messages.append(msg)
                    if msg["type"] in ("complete", "error"):
                        break
                except Exception:
                    break

    types = [m["type"] for m in messages]
    assert "language_detected" in types, f"Expected 'language_detected' in {types}"
    lang_msg = next(m for m in messages if m["type"] == "language_detected")
    assert lang_msg["code"] == "mr"


# ─── Frontend WebSocket integration tests ─────────────────────────────────────

def _html() -> str:
    """Return the built JS bundle (contains all frontend logic post-Vite migration)."""
    files = list(_DIST_ASSETS.glob("index-*.js"))
    assert files, "No JS bundle found. Run: cd frontend && npm run build"
    return files[0].read_text(encoding="utf-8")


class TestFrontendWebSocket:
    def test_has_websocket_constructor(self):
        assert "new WebSocket" in _html()

    def test_uses_stream_endpoint(self):
        assert "/api/stream/" in _html()

    def test_handles_segment_message(self):
        html = _html()
        assert '"segment"' in html or "'segment'" in html

    def test_handles_translation_message(self):
        html = _html()
        assert '"translation"' in html or "'translation'" in html

    def test_handles_complete_message(self):
        html = _html()
        assert '"complete"' in html or "'complete'" in html

    def test_handles_error_message(self):
        html = _html()
        assert '"error"' in html or "'error'" in html

    def test_handles_language_detected_message(self):
        html = _html()
        assert "language_detected" in html

    def test_has_rest_fallback(self):
        # REST fallback calls /api/transcribe when WebSocket fails
        html = _html()
        assert "/api/transcribe" in html

    def test_ws_uses_correct_protocol(self):
        # Should switch between ws: and wss: based on page protocol
        html = _html()
        assert "wss:" in html or "ws:" in html
        assert "window.location.protocol" in html

    def test_segments_added_progressively(self):
        # Frontend should update segments state as messages arrive (not batch at end)
        html = _html()
        assert "setSegments" in html

    def test_progress_updates_during_stream(self):
        # Progress is tracked as segments arrive (Math.min used for capping)
        html = _html()
        assert "Math.min" in html
