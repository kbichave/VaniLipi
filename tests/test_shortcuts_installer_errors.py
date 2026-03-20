"""
Phases 11-13: Keyboard shortcuts, installer scripts, and error handling tests.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

_DIST_ASSETS = Path(__file__).parent.parent / "frontend" / "dist" / "assets"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
README = Path(__file__).parent.parent / "README.md"


def _html() -> str:
    """Return the built JS bundle (contains all frontend logic post-Vite migration)."""
    files = list(_DIST_ASSETS.glob("index-*.js"))
    assert files, "No JS bundle found. Run: cd frontend && npm run build"
    return files[0].read_text(encoding="utf-8")


# ─── Phase 11: Keyboard shortcuts ─────────────────────────────────────────────

class TestKeyboardShortcuts:
    def test_space_play_pause(self):
        assert "Space" in _html()

    def test_tab_segment_navigation(self):
        html = _html()
        assert "Tab" in html
        assert "selectedSegIdx" in html

    def test_enter_to_edit(self):
        html = _html()
        assert '"Enter"' in html or "'Enter'" in html

    def test_escape_cancel(self):
        html = _html()
        assert '"Escape"' in html or "'Escape'" in html

    def test_cmd_e_export(self):
        html = _html()
        assert '"e"' in html or "'e'" in html

    def test_cmd_f_search(self):
        html = _html()
        assert '"f"' in html or "'f'" in html

    def test_cmd_shift_t_retranslate(self):
        html = _html()
        assert '"t"' in html or "'t'" in html
        assert "shiftKey" in html

    def test_speed_presets(self):
        html = _html()
        assert '"1"' in html or "'1'" in html
        assert "playbackRate" in html or "setPlaybackRate" in html

    def test_no_intercept_when_typing(self):
        # Shortcuts should not fire when typing in textarea/input
        html = _html()
        assert "TEXTAREA" in html or "textarea" in html


# ─── Phase 12: Installer scripts ──────────────────────────────────────────────

class TestInstallerScripts:
    def test_install_sh_exists(self):
        assert (SCRIPTS_DIR / "install.sh").exists()

    def test_launch_sh_exists(self):
        assert (SCRIPTS_DIR / "launch.sh").exists()

    def test_create_app_sh_exists(self):
        assert (SCRIPTS_DIR / "create_app.sh").exists()

    def test_readme_exists(self):
        assert README.exists()

    def test_install_sh_is_bash(self):
        content = (SCRIPTS_DIR / "install.sh").read_text()
        assert content.startswith("#!/")
        assert "bash" in content

    def test_launch_sh_starts_uvicorn(self):
        content = (SCRIPTS_DIR / "launch.sh").read_text()
        assert "uvicorn" in content

    def test_create_app_sh_creates_app_bundle(self):
        content = (SCRIPTS_DIR / "create_app.sh").read_text()
        assert "Contents/MacOS" in content or ".app" in content

    def test_readme_has_quick_start(self):
        content = README.read_text()
        assert "Quick Start" in content or "quick start" in content.lower()

    def test_readme_has_install_command(self):
        content = README.read_text()
        assert "install.sh" in content

    def test_readme_has_launch_command(self):
        content = README.read_text()
        assert "launch.sh" in content

    def test_readme_has_keyboard_shortcuts(self):
        content = README.read_text()
        assert "Space" in content

    def test_readme_has_supported_languages(self):
        content = README.read_text()
        assert "Marathi" in content
        assert "Hindi" in content

    def test_readme_has_architecture_section(self):
        content = README.read_text()
        assert "Architecture" in content or "architecture" in content.lower()

    def test_readme_has_troubleshooting(self):
        content = README.read_text()
        assert "Troubleshoot" in content or "troubleshoot" in content.lower()


# ─── Phase 13: Error handling ─────────────────────────────────────────────────

@pytest.fixture
def app():
    from backend.main import app
    return app


@pytest.mark.asyncio
async def test_no_speech_returns_warning(app, tmp_path):
    """When ASR returns empty segments, API returns a warning instead of 500."""
    import io
    fake_file = tmp_path / "silent_file.wav"
    fake_file.write_bytes(b"RIFF" + b"\x00" * 100)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with (
            patch("backend.main._find_upload", return_value=str(fake_file)),
            patch("backend.main.validate_and_prepare", return_value=(str(fake_file), 2.0)),
            patch("backend.main.asr.transcribe", return_value={"language": "mr", "segments": []}),
            patch("backend.main.asr.get_model_id", return_value="test-model"),
        ):
            resp = await client.post(
                "/api/transcribe",
                data={"file_id": "silent_test", "language": "mr", "model": "best"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["segments"] == []
    assert data["warning"] is not None
    assert "speech" in data["warning"].lower() or "silent" in data["warning"].lower()


@pytest.mark.asyncio
async def test_corrupted_file_returns_422(app, tmp_path):
    """AudioValidationError from corrupted file returns HTTP 422."""
    from backend.services.audio import AudioValidationError
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with (
            patch("backend.main._find_upload", return_value="/fake/path.wav"),
            patch("backend.main.validate_and_prepare", side_effect=AudioValidationError("File corrupted")),
        ):
            resp = await client.post(
                "/api/transcribe",
                data={"file_id": "bad_file", "language": "mr", "model": "best"},
            )

    assert resp.status_code == 422
    assert "corrupted" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_translation_failure_returns_warning(app, tmp_path):
    """If translation fails, the API returns a warning but still returns the transcript."""
    fake_file = tmp_path / "audio.wav"
    fake_file.write_bytes(b"RIFF" + b"\x00" * 100)

    fake_segments = [{"id": 0, "start": 0.0, "end": 2.0, "text": "मला आवडते."}]

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with (
            patch("backend.main._find_upload", return_value=str(fake_file)),
            patch("backend.main.validate_and_prepare", return_value=(str(fake_file), 2.0)),
            patch("backend.main.asr.transcribe", return_value={"language": "mr", "segments": fake_segments}),
            patch("backend.main.asr.get_model_id", return_value="test-model"),
            patch("backend.main.asr.unload"),
            patch("backend.main.translator.load"),
            patch("backend.main.translator.translate_segments", side_effect=RuntimeError("OOM")),
            patch("backend.main.project_store.save_project"),
            patch("backend.main.file_sha256", return_value="abc123"),
        ):
            resp = await client.post(
                "/api/transcribe",
                data={"file_id": "transl_fail", "language": "mr", "model": "best"},
            )

    assert resp.status_code == 200
    data = resp.json()
    # Transcript should still be returned
    assert len(data["segments"]) > 0
    # Warning should mention translation failure
    assert data["warning"] is not None
    assert "translation" in data["warning"].lower() or "failed" in data["warning"].lower()


@pytest.mark.asyncio
async def test_unsupported_file_type_rejected(app):
    """Uploading an .exe file is rejected with 400."""
    import io
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/upload",
            files={"file": ("malware.exe", io.BytesIO(b"MZ" + b"\x00" * 100), "application/octet-stream")},
        )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_ws_no_speech_sends_warning(app, tmp_path):
    """WebSocket endpoint sends warning when no speech detected."""
    import json
    from starlette.testclient import TestClient

    # Name must match {file_id}_* pattern used by _find_upload
    fake_file = tmp_path / "silent_ws_audio.wav"
    fake_file.write_bytes(b"RIFF" + b"\x00" * 100)

    import backend.main as m
    m.TEMP_DIR = tmp_path

    with (
        patch("backend.main.validate_and_prepare", return_value=(str(fake_file), 1.0)),
        patch("backend.main.asr.transcribe", return_value={"language": "mr", "segments": []}),
        patch("backend.main.asr.get_model_id", return_value="test-model"),
    ):
        client = TestClient(app)
        with client.websocket_connect("/api/stream/silent_ws") as ws:
            ws.send_text(json.dumps({"action": "transcribe", "language": "mr", "model": "turbo"}))
            messages = []
            for _ in range(5):
                try:
                    raw = ws.receive_text()
                    msg = json.loads(raw)
                    messages.append(msg)
                    if msg["type"] in ("complete", "error"):
                        break
                except Exception:
                    break

    types = [m["type"] for m in messages]
    assert "warning" in types, f"Expected 'warning' in {types}"
    assert "complete" in types, f"Expected 'complete' in {types}"
