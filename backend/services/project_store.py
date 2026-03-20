"""
Project persistence service.
Auto-saves transcripts by audio file SHA256 hash.
Provides fast cache lookup: same audio file → instant transcript retrieval.

Storage layout:
    ~/.local/share/VaniLipi/projects/<sha256>.json

Each project JSON has:
    {
        "file_hash": str,
        "filename": str,
        "created_at": str (ISO8601),
        "updated_at": str (ISO8601),
        "language": str,
        "detected_language": str | null,
        "duration_seconds": float,
        "segments": [{"start", "end", "marathi", "english"}, ...]
    }
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import APP_SUPPORT_DIR

logger = logging.getLogger(__name__)

PROJECTS_DIR = APP_SUPPORT_DIR / "projects"


def _ensure_dir() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


def _project_path(file_hash: str) -> Path:
    return PROJECTS_DIR / f"{file_hash}.json"


def save_project(
    file_hash: str,
    filename: str,
    language: str,
    detected_language: str | None,
    duration_seconds: float,
    segments: list[dict[str, Any]],
) -> None:
    """Persist a transcript. Overwrites any existing entry for this file hash."""
    _ensure_dir()
    now = datetime.now(timezone.utc).isoformat()
    path = _project_path(file_hash)

    # Preserve original created_at if updating existing project
    created_at = now
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            created_at = existing.get("created_at", now)
        except Exception:
            pass

    payload: dict[str, Any] = {
        "file_hash": file_hash,
        "filename": filename,
        "created_at": created_at,
        "updated_at": now,
        "language": language,
        "detected_language": detected_language,
        "duration_seconds": duration_seconds,
        "segments": segments,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved project %s (%d segments)", file_hash[:8], len(segments))


def load_project(file_hash: str) -> dict[str, Any] | None:
    """Load cached transcript for a given file hash. Returns None on miss."""
    path = _project_path(file_hash)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load project %s: %s", file_hash[:8], exc)
        return None


def list_projects() -> list[dict[str, Any]]:
    """
    Return all saved projects sorted by updated_at descending.
    Each entry includes all fields except segments (for fast listing).
    """
    _ensure_dir()
    projects = []
    for path in PROJECTS_DIR.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            # Summary: exclude segments to keep response small
            projects.append({
                "file_hash": raw.get("file_hash", ""),
                "filename": raw.get("filename", ""),
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
                "language": raw.get("language", ""),
                "detected_language": raw.get("detected_language"),
                "duration_seconds": raw.get("duration_seconds", 0.0),
                "segment_count": len(raw.get("segments", [])),
            })
        except Exception as exc:
            logger.warning("Skipping corrupt project file %s: %s", path.name, exc)

    projects.sort(key=lambda p: p["updated_at"], reverse=True)
    return projects


def delete_project(file_hash: str) -> bool:
    """Delete a saved project. Returns True if deleted, False if not found."""
    path = _project_path(file_hash)
    if not path.exists():
        return False
    path.unlink()
    logger.info("Deleted project %s", file_hash[:8])
    return True
