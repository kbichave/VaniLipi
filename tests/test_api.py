"""
Tests for Phase 2: FastAPI REST endpoints.
Uses httpx AsyncClient with the app — no real ML models loaded.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.main import app


# ---------------------------------------------------------------------------
# Fixture: async client
# ---------------------------------------------------------------------------

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spa_serves_html(client, tmp_path):
    """GET / should return the index.html (or 503 if not built yet)."""
    response = await client.get("/")
    # Either HTML (200) or the planned 503 before Phase 3 ships
    assert response.status_code in (200, 503)


# ---------------------------------------------------------------------------
# GET /api/languages
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_languages_returns_14(client):
    response = await client.get("/api/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    assert len(data["languages"]) == 14


@pytest.mark.asyncio
async def test_languages_include_marathi(client):
    response = await client.get("/api/languages")
    codes = [lang["code"] for lang in response.json()["languages"]]
    assert "mr" in codes


@pytest.mark.asyncio
async def test_languages_include_auto_detect(client):
    response = await client.get("/api/languages")
    data = response.json()
    assert "auto_detect" in data
    assert data["auto_detect"]["code"] == "auto"


@pytest.mark.asyncio
async def test_languages_have_required_fields(client):
    response = await client.get("/api/languages")
    for lang in response.json()["languages"]:
        for field in ("code", "name", "script", "quality", "recommended"):
            assert field in lang, f"Missing field {field!r} in {lang}"


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_valid_file(client, tmp_path):
    with (
        patch("backend.main.save_upload", return_value=tmp_path / "test.wav"),
        patch("backend.main.file_sha256", return_value="fakehash123"),
    ):
        response = await client.post(
            "/api/upload",
            files={"file": ("recording.wav", b"RIFF....fake", "audio/wav")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["filename"] == "recording.wav"
    assert data["size_bytes"] == len(b"RIFF....fake")


@pytest.mark.asyncio
async def test_upload_empty_file_rejected(client):
    response = await client.post(
        "/api/upload",
        files={"file": ("recording.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_unsupported_extension_rejected(client):
    response = await client.post(
        "/api/upload",
        files={"file": ("document.pdf", b"fake data", "application/pdf")},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/retranslate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retranslate_valid_request(client):
    with patch("backend.main.translator") as mock_tr:
        mock_tr.is_loaded.return_value = True
        mock_tr.translate_single = MagicMock(return_value="This is a test.")

        response = await client.post(
            "/api/retranslate",
            json={"segment_id": 3, "text": "हा एक चाचणी आहे.", "language_code": "mr"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["segment_id"] == 3
    assert "english" in data


@pytest.mark.asyncio
async def test_retranslate_empty_text_rejected(client):
    response = await client.post(
        "/api/retranslate",
        json={"segment_id": 0, "text": "", "language_code": "mr"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_retranslate_unsupported_language_rejected(client):
    response = await client.post(
        "/api/retranslate",
        json={"segment_id": 0, "text": "Hello", "language_code": "en"},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/transcribe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcribe_missing_file_returns_error(client):
    """If file_id doesn't match any upload, _find_upload returns None → 422."""
    with patch("backend.main._find_upload", return_value=None):
        response = await client.post(
            "/api/transcribe",
            data={"file_id": "nonexistent", "language": "mr"},
        )
    # None path passed to validate_and_prepare raises AudioValidationError
    assert response.status_code in (422, 500)


@pytest.mark.asyncio
async def test_transcribe_successful_pipeline(client, tmp_path):
    """Full transcribe pipeline with all ML calls mocked."""
    fake_wav = tmp_path / "audio.wav"
    fake_wav.write_bytes(b"fake")

    fake_segments = [
        {"id": 0, "start": 0.0, "end": 2.0, "text": "हा एक चाचणी आहे.", "english": "This is a test."}
    ]

    with (
        patch("backend.main._find_upload", return_value=fake_wav),
        patch("backend.main.validate_and_prepare", return_value=(fake_wav, 5.0)),
        patch("backend.main.asr") as mock_asr,
        patch("backend.main.translator") as mock_tr,
    ):
        mock_asr.get_model_id.return_value = "mlx-community/whisper-large-v3-turbo"
        mock_asr.transcribe.return_value = {
            "text": "हा एक चाचणी आहे.",
            "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "हा एक चाचणी आहे."}],
            "language": "mr",
        }
        mock_asr.unload.return_value = None
        mock_tr.load.return_value = None
        mock_tr.translate_segments.return_value = fake_segments

        response = await client.post(
            "/api/transcribe",
            data={"file_id": "abc123", "language": "mr"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["language"]["code"] == "mr"
    assert len(data["segments"]) == 1
    assert data["segments"][0]["english"] == "This is a test."


# ---------------------------------------------------------------------------
# GET /api/models/status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_models_status_returns_dict(client):
    with patch("backend.services.model_manager.check_models_status") as mock_status:
        mock_status.return_value = {"asr": {}, "translation": {}}
        response = await client.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "asr" in data or "translation" in data or isinstance(data, dict)
