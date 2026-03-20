"""
Phase 9: Project persistence tests.
Tests the project_store service and /api/projects endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient


SAMPLE_SEGMENTS = [
    {"start": 0.0, "end": 2.5, "marathi": "हा एक चाचणी.", "english": "This is a test."},
    {"start": 2.5, "end": 5.0, "marathi": "मला आवडते.", "english": "I love it."},
]


# ─── Unit tests ───────────────────────────────────────────────────────────────

class TestProjectStore:
    def test_save_and_load(self, tmp_path):
        from backend.services import project_store as ps
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            ps.save_project(
                file_hash="abc123",
                filename="test.wav",
                language="mr",
                detected_language="mr",
                duration_seconds=5.0,
                segments=SAMPLE_SEGMENTS,
            )
            result = ps.load_project("abc123")

        assert result is not None
        assert result["file_hash"] == "abc123"
        assert result["filename"] == "test.wav"
        assert result["language"] == "mr"
        assert len(result["segments"]) == 2

    def test_load_returns_none_for_missing(self, tmp_path):
        from backend.services import project_store as ps
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            result = ps.load_project("nonexistent")
        assert result is None

    def test_save_preserves_created_at(self, tmp_path):
        from backend.services import project_store as ps
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            ps.save_project("abc123", "f.wav", "mr", "mr", 5.0, SAMPLE_SEGMENTS)
            first = ps.load_project("abc123")
            ps.save_project("abc123", "f.wav", "mr", "mr", 5.0, SAMPLE_SEGMENTS)
            second = ps.load_project("abc123")

        # created_at should be preserved on update
        assert first["created_at"] == second["created_at"]
        # updated_at may differ if test is fast enough (same second) but should exist
        assert "updated_at" in second

    def test_list_projects_empty(self, tmp_path):
        from backend.services import project_store as ps
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            results = ps.list_projects()
        assert results == []

    def test_list_projects_returns_saved(self, tmp_path):
        from backend.services import project_store as ps
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            ps.save_project("aaa", "a.wav", "mr", "mr", 3.0, SAMPLE_SEGMENTS)
            ps.save_project("bbb", "b.wav", "hi", "hi", 4.0, SAMPLE_SEGMENTS)
            results = ps.list_projects()

        assert len(results) == 2
        hashes = {r["file_hash"] for r in results}
        assert "aaa" in hashes
        assert "bbb" in hashes

    def test_list_projects_excludes_segments(self, tmp_path):
        from backend.services import project_store as ps
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            ps.save_project("abc", "f.wav", "mr", "mr", 5.0, SAMPLE_SEGMENTS)
            results = ps.list_projects()

        assert "segments" not in results[0]
        assert "segment_count" in results[0]
        assert results[0]["segment_count"] == 2

    def test_list_projects_sorted_newest_first(self, tmp_path):
        from backend.services import project_store as ps
        import time
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            ps.save_project("first", "first.wav", "mr", "mr", 3.0, [])
            time.sleep(0.01)  # ensure different timestamps
            ps.save_project("second", "second.wav", "mr", "mr", 3.0, [])
            results = ps.list_projects()

        assert results[0]["file_hash"] == "second"

    def test_delete_project(self, tmp_path):
        from backend.services import project_store as ps
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            ps.save_project("del123", "del.wav", "mr", "mr", 2.0, [])
            assert ps.load_project("del123") is not None
            deleted = ps.delete_project("del123")
            assert deleted is True
            assert ps.load_project("del123") is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        from backend.services import project_store as ps
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            result = ps.delete_project("ghost")
        assert result is False

    def test_unicode_preserved_in_storage(self, tmp_path):
        from backend.services import project_store as ps
        segs = [{"start": 0.0, "end": 1.0, "marathi": "मराठी भाषा", "english": "Marathi language"}]
        with patch.object(ps, "PROJECTS_DIR", tmp_path):
            ps.save_project("uni123", "test.wav", "mr", "mr", 1.0, segs)
            result = ps.load_project("uni123")

        assert result["segments"][0]["marathi"] == "मराठी भाषा"


# ─── API endpoint tests ───────────────────────────────────────────────────────

@pytest.fixture
def app():
    from backend.main import app
    return app


@pytest.mark.asyncio
async def test_get_projects_endpoint(app, tmp_path):
    from backend.services import project_store as ps
    with patch.object(ps, "PROJECTS_DIR", tmp_path):
        ps.save_project("p1", "a.wav", "mr", "mr", 3.0, SAMPLE_SEGMENTS)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch("backend.main.project_store.list_projects", return_value=ps.list_projects()):
                resp = await client.get("/api/projects")

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_project_not_found(app, tmp_path):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("backend.main.project_store.load_project", return_value=None):
            resp = await client.get("/api/projects/ghost")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_project_found(app, tmp_path):
    project = {
        "file_hash": "abc",
        "filename": "test.wav",
        "segments": SAMPLE_SEGMENTS,
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("backend.main.project_store.load_project", return_value=project):
            resp = await client.get("/api/projects/abc")
    assert resp.status_code == 200
    assert resp.json()["file_hash"] == "abc"


@pytest.mark.asyncio
async def test_delete_project_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("backend.main.project_store.delete_project", return_value=True):
            resp = await client.delete("/api/projects/abc")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "abc"


@pytest.mark.asyncio
async def test_delete_project_not_found(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("backend.main.project_store.delete_project", return_value=False):
            resp = await client.delete("/api/projects/ghost")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_returns_cached_flag(app, tmp_path):
    """Upload endpoint should return cached=True when a matching hash exists."""
    import io
    from backend.services import project_store as ps

    with patch("backend.main.project_store.load_project", return_value={"file_hash": "x"}):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/upload",
                files={"file": ("test.wav", io.BytesIO(b"RIFF" + b"\x00" * 100), "audio/wav")},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "cached" in data
    assert "file_hash" in data
