"""
Phase 8: Export system tests.
Tests the exporter service and /api/export/{format} endpoint.
"""
from __future__ import annotations

import json
import pytest
from httpx import ASGITransport, AsyncClient


SAMPLE_SEGMENTS = [
    {"start": 0.0,  "end": 2.5,  "marathi": "हा एक चाचणी आहे.", "english": "This is a test."},
    {"start": 2.5,  "end": 5.0,  "marathi": "मला मराठी आवडते.",   "english": "I love Marathi."},
    {"start": 5.0,  "end": 10.0, "marathi": "आज चांगले हवामान.",   "english": "Good weather today."},
]

EMPTY_SEGMENTS: list = []


# ─── Exporter unit tests ──────────────────────────────────────────────────────

class TestSRT:
    def test_produces_valid_srt_structure(self):
        from backend.services.exporter import to_srt
        result = to_srt(SAMPLE_SEGMENTS).decode("utf-8")
        assert "1\n" in result
        assert "-->" in result
        assert "This is a test." in result

    def test_srt_timestamp_format(self):
        from backend.services.exporter import _srt_ts
        assert _srt_ts(0.0) == "00:00:00,000"
        assert _srt_ts(3661.5) == "01:01:01,500"

    def test_srt_has_sequence_numbers(self):
        from backend.services.exporter import to_srt
        result = to_srt(SAMPLE_SEGMENTS).decode("utf-8")
        assert "\n1\n" in result or result.startswith("1\n")
        assert "\n2\n" in result
        assert "\n3\n" in result

    def test_empty_segments(self):
        from backend.services.exporter import to_srt
        result = to_srt(EMPTY_SEGMENTS)
        assert isinstance(result, bytes)

    def test_includes_both_languages(self):
        from backend.services.exporter import to_srt
        result = to_srt(SAMPLE_SEGMENTS).decode("utf-8")
        assert "हा एक चाचणी आहे." in result
        assert "This is a test." in result


class TestVTT:
    def test_starts_with_webvtt(self):
        from backend.services.exporter import to_vtt
        result = to_vtt(SAMPLE_SEGMENTS).decode("utf-8")
        assert result.startswith("WEBVTT")

    def test_vtt_uses_dot_not_comma(self):
        from backend.services.exporter import to_vtt
        result = to_vtt(SAMPLE_SEGMENTS).decode("utf-8")
        # VTT uses dots, SRT uses commas for milliseconds
        assert "00:00:00.000" in result

    def test_vtt_timestamp_format(self):
        from backend.services.exporter import _vtt_ts
        assert _vtt_ts(0.0) == "00:00:00.000"
        assert _vtt_ts(90.0) == "00:01:30.000"


class TestTXT:
    def test_txt_both_mode_includes_all(self):
        from backend.services.exporter import to_txt
        result = to_txt(SAMPLE_SEGMENTS, mode="both").decode("utf-8")
        assert "हा एक चाचणी आहे." in result
        assert "This is a test." in result

    def test_txt_marathi_only(self):
        from backend.services.exporter import to_txt
        result = to_txt(SAMPLE_SEGMENTS, mode="marathi").decode("utf-8")
        assert "हा एक चाचणी आहे." in result
        assert "This is a test." not in result

    def test_txt_english_only(self):
        from backend.services.exporter import to_txt
        result = to_txt(SAMPLE_SEGMENTS, mode="english").decode("utf-8")
        assert "This is a test." in result
        assert "हा एक चाचणी आहे." not in result

    def test_txt_has_timestamps(self):
        from backend.services.exporter import to_txt
        result = to_txt(SAMPLE_SEGMENTS).decode("utf-8")
        assert "[00:00]" in result


class TestJSON:
    def test_produces_valid_json(self):
        from backend.services.exporter import to_json
        result = json.loads(to_json(SAMPLE_SEGMENTS))
        assert "transcript" in result
        assert len(result["transcript"]) == 3

    def test_json_has_all_fields(self):
        from backend.services.exporter import to_json
        result = json.loads(to_json(SAMPLE_SEGMENTS))
        seg = result["transcript"][0]
        assert "start" in seg
        assert "end" in seg
        assert "marathi" in seg
        assert "english" in seg
        assert "index" in seg

    def test_json_preserves_unicode(self):
        from backend.services.exporter import to_json
        result = json.loads(to_json(SAMPLE_SEGMENTS))
        assert result["transcript"][0]["marathi"] == "हा एक चाचणी आहे."


class TestDispatch:
    def test_unknown_format_raises_value_error(self):
        from backend.services.exporter import export
        with pytest.raises(ValueError, match="Unknown export format"):
            export(SAMPLE_SEGMENTS, "xyz")

    def test_all_formats_dispatch(self):
        from backend.services.exporter import export, EXPORTERS
        for fmt in ("srt", "vtt", "txt", "json"):
            content, mime = export(SAMPLE_SEGMENTS, fmt)
            assert isinstance(content, bytes)
            assert isinstance(mime, str)

    def test_mime_types_correct(self):
        from backend.services.exporter import export
        _, srt_mime = export(SAMPLE_SEGMENTS, "srt")
        _, json_mime = export(SAMPLE_SEGMENTS, "json")
        assert "text" in srt_mime
        assert "json" in json_mime


# ─── API endpoint tests ───────────────────────────────────────────────────────

@pytest.fixture
def app():
    from backend.main import app
    return app


@pytest.mark.asyncio
async def test_export_srt_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/export/srt", json={
            "file_id": "test",
            "segments": SAMPLE_SEGMENTS,
        })
    assert resp.status_code == 200
    assert "text" in resp.headers["content-type"]
    assert "1\n" in resp.text or resp.text.startswith("1\n")


@pytest.mark.asyncio
async def test_export_vtt_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/export/vtt", json={
            "file_id": "test",
            "segments": SAMPLE_SEGMENTS,
        })
    assert resp.status_code == 200
    assert resp.text.startswith("WEBVTT")


@pytest.mark.asyncio
async def test_export_json_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/export/json", json={
            "file_id": "test",
            "segments": SAMPLE_SEGMENTS,
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "transcript" in data


@pytest.mark.asyncio
async def test_export_txt_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/export/txt", json={
            "file_id": "test",
            "segments": SAMPLE_SEGMENTS,
        })
    assert resp.status_code == 200
    assert "This is a test." in resp.text


@pytest.mark.asyncio
async def test_export_unknown_format_returns_400(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/export/xyz", json={
            "file_id": "test",
            "segments": SAMPLE_SEGMENTS,
        })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_export_has_content_disposition_header(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/export/srt", json={
            "file_id": "test",
            "segments": SAMPLE_SEGMENTS,
        })
    assert "content-disposition" in resp.headers
    assert "transcript.srt" in resp.headers["content-disposition"]


@pytest.mark.asyncio
async def test_export_empty_segments(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/export/json", json={
            "file_id": "test",
            "segments": [],
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["transcript"] == []
