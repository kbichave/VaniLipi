"""
Phase 10: Model management tests.
Tests model status checking, SSE download streaming, and frontend UI integration.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from pathlib import Path

_DIST_ASSETS = Path(__file__).parent.parent / "frontend" / "dist" / "assets"


# ─── Model manager unit tests ─────────────────────────────────────────────────

class TestCheckModelsStatus:
    @pytest.mark.asyncio
    async def test_returns_models_list(self):
        with patch("backend.services.model_manager.asyncio") as mock_asyncio:
            loop_mock = MagicMock()
            loop_mock.run_in_executor = AsyncMock(return_value=False)
            mock_asyncio.get_event_loop.return_value = loop_mock

            from backend.services.model_manager import check_models_status
            result = await check_models_status()

        assert "models" in result
        assert isinstance(result["models"], list)
        assert len(result["models"]) >= 2

    @pytest.mark.asyncio
    async def test_models_have_required_fields(self):
        with patch("backend.services.model_manager.asyncio") as mock_asyncio:
            loop_mock = MagicMock()
            loop_mock.run_in_executor = AsyncMock(return_value=False)
            mock_asyncio.get_event_loop.return_value = loop_mock

            from backend.services.model_manager import check_models_status
            result = await check_models_status()

        for m in result["models"]:
            assert "id" in m
            assert "label" in m
            assert "type" in m
            assert "downloaded" in m
            assert "required" in m

    @pytest.mark.asyncio
    async def test_ready_is_true_when_all_required_downloaded(self):
        with patch("backend.services.model_manager.asyncio") as mock_asyncio:
            loop_mock = MagicMock()
            loop_mock.run_in_executor = AsyncMock(return_value=True)
            mock_asyncio.get_event_loop.return_value = loop_mock

            from backend.services.model_manager import check_models_status
            result = await check_models_status()

        assert result["ready"] is True

    @pytest.mark.asyncio
    async def test_ready_is_false_when_models_missing(self):
        with patch("backend.services.model_manager.asyncio") as mock_asyncio:
            loop_mock = MagicMock()
            loop_mock.run_in_executor = AsyncMock(return_value=False)
            mock_asyncio.get_event_loop.return_value = loop_mock

            from backend.services.model_manager import check_models_status
            result = await check_models_status()

        assert result["ready"] is False


class TestSSEFormat:
    def test_sse_format(self):
        from backend.services.model_manager import _sse
        result = _sse({"type": "done", "model_id": "test"})
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_sse_contains_json(self):
        import json
        from backend.services.model_manager import _sse
        result = _sse({"type": "progress", "elapsed_seconds": 5})
        data_line = result.replace("data: ", "").strip()
        parsed = json.loads(data_line)
        assert parsed["type"] == "progress"
        assert parsed["elapsed_seconds"] == 5


class TestDownloadModelSSE:
    @pytest.mark.asyncio
    async def test_unknown_model_yields_error(self):
        from backend.services.model_manager import download_model_sse
        import json
        messages = []
        async for chunk in download_model_sse("unknown/model"):
            data = chunk.replace("data: ", "").strip()
            if data:
                messages.append(json.loads(data))

        assert any(m["type"] == "error" for m in messages)


# ─── API endpoint tests ───────────────────────────────────────────────────────

@pytest.fixture
def app():
    from backend.main import app
    return app


@pytest.mark.asyncio
async def test_models_status_endpoint(app):
    mock_status = {
        "models": [
            {"id": "mlx-community/whisper-large-v3-turbo", "label": "Whisper", "type": "asr", "downloaded": True, "required": True}
        ],
        "ready": True,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("backend.services.model_manager.check_models_status", new=AsyncMock(return_value=mock_status)):
            resp = await client.get("/api/models/status")

    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data or "asr" in data  # support both old and new format


@pytest.mark.asyncio
async def test_download_endpoint_exists(app):
    """The download endpoint should return 200 or streaming response."""
    from backend.services.model_manager import _sse

    async def mock_sse(model_id):
        yield _sse({"type": "done", "model_id": model_id})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("backend.services.model_manager.download_model_sse", return_value=mock_sse("test")):
            resp = await client.get("/api/models/download/test-model")

    assert resp.status_code == 200


# ─── Frontend Phase 10 tests ──────────────────────────────────────────────────

def _html() -> str:
    """Return the built JS bundle (contains all frontend logic post-Vite migration)."""
    files = list(_DIST_ASSETS.glob("index-*.js"))
    assert files, "No JS bundle found. Run: cd frontend && npm run build"
    return files[0].read_text(encoding="utf-8")


class TestFrontendModelManagement:
    def test_has_model_status_check(self):
        assert "/api/models/status" in _html()

    def test_has_model_setup_modal(self):
        # Model setup shows download progress for required models
        html = _html()
        assert "Download Models" in html or "downloaded" in html

    def test_has_recent_projects_modal(self):
        # Recent projects modal lists saved transcripts
        html = _html()
        assert "Recent Projects" in html or "/api/projects" in html

    def test_has_projects_api_call(self):
        assert "/api/projects" in _html()

    def test_has_sse_eventstream(self):
        assert "EventSource" in _html()

    def test_has_download_endpoint_reference(self):
        assert "/api/models/download" in _html()
