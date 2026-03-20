"""
VaniLipi FastAPI application.

Serves the React SPA at GET / and provides the ML pipeline via REST + WebSocket.

Port selection: tries DEFAULT_PORT first; increments by 1 up to PORT_SEARCH_RANGE
times if the port is already in use (supports multiple simultaneous app instances).
"""
import asyncio
import json
import logging
import socket
import uuid
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import backend.services.asr as asr
import backend.services.translator as translator
from backend.services.exporter import export as export_transcript
from backend.services import project_store
from backend.config import (
    DEFAULT_LANGUAGE,
    LANGUAGE_MAP,
    SUPPORTED_WHISPER_CODES,
    TEMP_DIR,
)
from backend.services.audio import (
    AudioValidationError,
    cleanup_chunks,
    file_sha256,
    save_upload,
    split_audio_chunks,
    validate_and_prepare,
)

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"

app = FastAPI(title="VaniLipi", version="1.0.0")

# CORS — required for pywebview's WKWebView which may use a different origin
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve hashed JS/CSS/font assets from the Vite build
_assets_dir = FRONTEND_DIR / "assets"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


# ---------------------------------------------------------------------------
# SPA entrypoint + static files from dist root (logo.png, etc.)
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_spa() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="Frontend not built yet.")
    return FileResponse(str(index), media_type="text/html")


@app.get("/logo.png")
async def serve_logo() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "logo.png"), media_type="image/png")


# ---------------------------------------------------------------------------
# /api/languages
# ---------------------------------------------------------------------------

@app.get("/api/languages")
async def get_languages() -> JSONResponse:
    """Return supported languages with quality tiers for the UI dropdown."""
    languages = [
        {
            "code": code,
            "name": info["name"],
            "script": info["script"],
            "quality": info["quality"],
            "recommended": info["recommended"],
        }
        for code, info in LANGUAGE_MAP.items()
    ]
    return JSONResponse(
        {
            "languages": languages,
            "auto_detect": {
                "code": "auto",
                "name": "Auto-detect",
                "description": "Let Whisper detect the language automatically",
            },
        }
    )


# ---------------------------------------------------------------------------
# /api/upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accept an audio file upload, validate the extension, save to TEMP_DIR.

    Returns: {"file_id": "...", "filename": "...", "size_bytes": N}
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Validate extension before saving to avoid storing garbage
    from backend.services.audio import (
        validate_extension, AudioValidationError,
        is_video_file, extract_audio_from_video,
    )
    from pathlib import Path as _Path
    try:
        validate_extension(_Path(file.filename))
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Use a UUID-prefixed filename to avoid collisions
    file_id = uuid.uuid4().hex
    saved_name = f"{file_id}_{file.filename}"
    saved_path = save_upload(data, saved_name)

    # If it's a video file, extract audio and discard the video
    if is_video_file(saved_path):
        try:
            saved_path = extract_audio_from_video(saved_path)
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # Check if we have a cached transcript for this exact file (Phase 9)
    file_hash = file_sha256(saved_path)
    cached = project_store.load_project(file_hash)
    cached_flag = cached is not None

    return JSONResponse(
        {
            "file_id": file_id,
            "file_hash": file_hash,
            "filename": file.filename,
            "size_bytes": len(data),
            "cached": cached_flag,
        }
    )


# ---------------------------------------------------------------------------
# /api/transcribe
# ---------------------------------------------------------------------------

@app.post("/api/transcribe")
async def transcribe_audio(
    file_id: Annotated[str, Form()],
    language: Annotated[str, Form()] = DEFAULT_LANGUAGE,
) -> JSONResponse:
    """
    Full pipeline: validate audio → ASR → translate.

    This is the synchronous REST endpoint (returns only when complete).
    For progressive streaming, use the WebSocket endpoint /api/stream/{file_id}.

    Body (multipart form):
        file_id:  ID returned by /api/upload
        language: "auto" or ISO code ("mr", "hi", …)
    """
    audio_path = _find_upload(file_id)
    if audio_path is None:
        raise HTTPException(status_code=422, detail=f"File {file_id!r} not found. Upload it first via /api/upload.")

    # Validate and convert to WAV
    try:
        wav_path, duration = validate_and_prepare(audio_path)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ASR
    whisper_language = None if language == "auto" else language

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asr.transcribe(wav_path, language=whisper_language),
        )
    except Exception as exc:
        logger.exception("ASR failed for file_id=%s", file_id)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")

    detected_code = result.get("language", "")
    segments = result.get("segments", [])

    # No speech detected (Phase 13)
    if not segments:
        return JSONResponse(
            {
                "file_id": file_id,
                "duration_seconds": duration,
                "language": {"code": detected_code, "name": detected_code, "detected": language == "auto"},
                "model": "whisper-large-v3",
                "segments": [],
                "warning": "No speech detected in this audio file. The file may be silent, too short, or in an unsupported language.",
            }
        )

    # Warn if auto-detect picked an unsupported language
    translation_warning = None
    translations_available = detected_code in SUPPORTED_WHISPER_CODES

    if not translations_available:
        lang_name = detected_code.upper() if detected_code else "Unknown"
        translation_warning = (
            f"Detected: {lang_name}. Translation to English is only available "
            f"for Indian languages. Showing transcript only."
        )

    # Translation
    if translations_available:
        try:
            asr.unload()
            translator.load()
            segments = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: translator.translate_segments(segments, detected_code),
            )
        except Exception as exc:
            logger.exception("Translation failed for file_id=%s", file_id)
            translation_warning = f"Translation failed: {exc}"

    language_info = LANGUAGE_MAP.get(detected_code, {})

    # Auto-save to project cache (Phase 9)
    try:
        file_hash = file_sha256(audio_path)
        # Normalize segments to {start, end, marathi, english} for storage
        normalized = [
            {
                "start": s.get("start", 0.0),
                "end": s.get("end", 0.0),
                "marathi": s.get("text", s.get("marathi", "")),
                "english": s.get("english", ""),
            }
            for s in segments
        ]
        audio_path_obj = audio_path if isinstance(audio_path, Path) else Path(audio_path)
        original_name = audio_path_obj.name.split("_", 1)[-1] if "_" in audio_path_obj.name else audio_path_obj.name
        project_store.save_project(
            file_hash=file_hash,
            filename=original_name,
            language=language,
            detected_language=detected_code or None,
            duration_seconds=duration,
            segments=normalized,
        )
    except Exception as exc:
        logger.warning("Auto-save failed: %s", exc)

    return JSONResponse(
        {
            "file_id": file_id,
            "duration_seconds": duration,
            "language": {
                "code": detected_code,
                "name": language_info.get("name", detected_code),
                "detected": language == "auto",
            },
            "model": "whisper-large-v3",
            "segments": segments,
            "warning": translation_warning,
        }
    )


# ---------------------------------------------------------------------------
# /api/retranslate
# ---------------------------------------------------------------------------

@app.post("/api/retranslate")
async def retranslate_segment(body: dict) -> JSONResponse:
    """
    Re-translate a single edited segment.

    Body JSON: {"segment_id": N, "text": "...", "language_code": "mr"}
    Returns:   {"segment_id": N, "english": "..."}
    """
    segment_id = body.get("segment_id")
    text = body.get("text", "").strip()
    language_code = body.get("language_code", "")

    if not text:
        raise HTTPException(status_code=400, detail="text is required.")
    if language_code not in SUPPORTED_WHISPER_CODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language code: {language_code!r}. "
                   f"Supported: {sorted(SUPPORTED_WHISPER_CODES)}",
        )

    if not translator.is_loaded():
        translator.load()

    try:
        english = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: translator.translate_single(text, language_code),
        )
    except Exception as exc:
        logger.exception("Re-translation failed.")
        raise HTTPException(status_code=500, detail=f"Re-translation failed: {exc}")

    return JSONResponse({"segment_id": segment_id, "english": english})


# ---------------------------------------------------------------------------
# /api/stream/{file_id} — WebSocket streaming
# ---------------------------------------------------------------------------

@app.websocket("/api/stream/{file_id}")
async def stream_transcription(websocket: WebSocket, file_id: str) -> None:
    """
    WebSocket endpoint for streaming transcription + translation results.

    For long files (>5 min), audio is split into chunks and processed in an
    interleaved pattern: ASR chunk → translate chunk → stream results → next chunk.
    This gives the user English output within ~30s instead of waiting for the
    entire file to finish.

    Client sends:
        {"action": "transcribe", "language": "auto"|"mr"|...}

    Server streams back JSON messages:
        {"type": "status",            "message": "..."}
        {"type": "language_detected", "code": "mr", "name": "Marathi"}
        {"type": "segment",           "id": N, "start": F, "end": F, "text": "..."}
        {"type": "translation",       "id": N, "english": "..."}
        {"type": "chunk_progress",    "chunk": N, "total_chunks": N}
        {"type": "warning",           "message": "..."}
        {"type": "complete",          "total_segments": N, "duration_seconds": F, "language": "mr"}
    """
    await websocket.accept()

    try:
        raw = await websocket.receive_text()
        request = json.loads(raw)
    except Exception:
        await _ws_send(websocket, {"type": "error", "message": "Invalid request format."})
        await websocket.close()
        return

    action = request.get("action")
    if action != "transcribe":
        await _ws_send(websocket, {"type": "error", "message": f"Unknown action: {action!r}"})
        await websocket.close()
        return

    language = request.get("language", "auto")

    audio_path = _find_upload(file_id)
    if audio_path is None:
        await _ws_send(websocket, {"type": "error", "message": f"File {file_id!r} not found."})
        await websocket.close()
        return

    # Validate + convert
    try:
        wav_path, duration = await asyncio.get_event_loop().run_in_executor(
            None, lambda: validate_and_prepare(audio_path)
        )
    except AudioValidationError as exc:
        await _ws_send(websocket, {"type": "error", "message": str(exc)})
        await websocket.close()
        return

    # Split into chunks for interleaved processing
    chunk_paths = await asyncio.get_event_loop().run_in_executor(
        None, lambda: split_audio_chunks(wav_path)
    )
    total_chunks = len(chunk_paths)
    whisper_language = None if language == "auto" else language

    all_segments: list[dict] = []
    detected_code = ""
    segment_id_offset = 0

    try:
        for chunk_idx, chunk_path in enumerate(chunk_paths):
            if total_chunks > 1:
                await _ws_send(websocket, {
                    "type": "chunk_progress",
                    "chunk": chunk_idx + 1,
                    "total_chunks": total_chunks,
                })

            # --- ASR for this chunk ---
            await _ws_send(websocket, {
                "type": "status",
                "message": f"Transcribing{f' (chunk {chunk_idx+1}/{total_chunks})' if total_chunks > 1 else ''}...",
            })

            try:
                result = await _run_with_heartbeat(
                    websocket,
                    lambda cp=chunk_path: asr.transcribe(cp, language=whisper_language),
                    f"Transcribing{f' chunk {chunk_idx+1}/{total_chunks}' if total_chunks > 1 else ''}…",
                )
            except Exception as exc:
                await _ws_send(websocket, {"type": "error", "message": f"Transcription failed: {exc}"})
                await websocket.close()
                return

            # Use detected language from first chunk
            if chunk_idx == 0:
                detected_code = result.get("language", "")
                if language == "auto" and detected_code:
                    lang_info = LANGUAGE_MAP.get(detected_code, {})
                    await _ws_send(websocket, {
                        "type": "language_detected",
                        "code": detected_code,
                        "name": lang_info.get("name", detected_code),
                    })

            chunk_segments = result.get("segments", [])
            if not chunk_segments:
                continue

            # Adjust segment timestamps and IDs for chunk offset
            chunk_time_offset = chunk_idx * 300.0  # matches split_audio_chunks default
            for seg in chunk_segments:
                seg["id"] = segment_id_offset + seg.get("id", 0)
                seg["start"] = seg.get("start", 0.0) + chunk_time_offset
                seg["end"] = seg.get("end", 0.0) + chunk_time_offset

            # Stream ASR segments immediately
            for seg in chunk_segments:
                await _ws_send(websocket, {
                    "type": "segment",
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg.get("text", ""),
                })

            # --- Translate this chunk's segments ---
            translations_available = detected_code in SUPPORTED_WHISPER_CODES
            if translations_available:
                await _ws_send(websocket, {
                    "type": "status",
                    "message": f"Translating{f' (chunk {chunk_idx+1}/{total_chunks})' if total_chunks > 1 else ''}...",
                })
                try:
                    asr.unload()
                    await _run_with_heartbeat(
                        websocket,
                        translator.load,
                        "Loading translation model…",
                    )
                    chunk_segments = await _run_with_heartbeat(
                        websocket,
                        lambda segs=chunk_segments: translator.translate_segments(segs, detected_code),
                        f"Translating{f' chunk {chunk_idx+1}/{total_chunks}' if total_chunks > 1 else ''}…",
                    )
                    for seg in chunk_segments:
                        await _ws_send(websocket, {
                            "type": "translation",
                            "id": seg.get("id", 0),
                            "english": seg.get("english", ""),
                        })
                except Exception as exc:
                    await _ws_send(websocket, {
                        "type": "warning",
                        "message": f"Translation failed for chunk {chunk_idx+1}: {exc}",
                    })

            all_segments.extend(chunk_segments)
            segment_id_offset += len(chunk_segments)

        # Warn if language not supported for translation
        if detected_code and detected_code not in SUPPORTED_WHISPER_CODES:
            lang_display = detected_code.upper()
            await _ws_send(websocket, {
                "type": "warning",
                "message": f"Detected: {lang_display}. Translation not available for this language.",
            })

        # No speech in entire file
        if not all_segments:
            await _ws_send(websocket, {
                "type": "warning",
                "message": "No speech detected in this audio file.",
            })

        await _ws_send(websocket, {
            "type": "complete",
            "total_segments": len(all_segments),
            "duration_seconds": duration,
            "language": detected_code,
        })
    finally:
        cleanup_chunks(chunk_paths)

    await websocket.close()


# ---------------------------------------------------------------------------
# /api/export/{format}
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    file_id: str = ""
    segments: list[dict] = []


@app.post("/api/export/{fmt}")
async def export_endpoint(fmt: str, body: ExportRequest) -> Response:
    """
    Export transcript segments to the requested format.
    Supported: srt, vtt, txt, json, docx, pdf
    """
    try:
        content, mime = await asyncio.get_event_loop().run_in_executor(
            None, lambda: export_transcript(body.segments, fmt)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    filename_map = {
        "srt": "transcript.srt",
        "vtt": "transcript.vtt",
        "txt": "transcript.txt",
        "json": "transcript.json",
        "docx": "transcript.docx",
        "pdf": "transcript.pdf",
    }
    filename = filename_map.get(fmt, f"transcript.{fmt}")

    return Response(
        content=content,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# /api/projects — Recent projects (Phase 9)
# ---------------------------------------------------------------------------

@app.get("/api/projects")
async def get_projects() -> JSONResponse:
    """List all saved projects, newest first."""
    return JSONResponse(project_store.list_projects())


@app.get("/api/projects/{file_hash}")
async def get_project(file_hash: str) -> JSONResponse:
    """Load a single saved project by file hash (includes segments)."""
    project = project_store.load_project(file_hash)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {file_hash!r} not found.")
    return JSONResponse(project)


@app.delete("/api/projects/{file_hash}")
async def delete_project(file_hash: str) -> JSONResponse:
    """Delete a saved project."""
    deleted = project_store.delete_project(file_hash)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Project {file_hash!r} not found.")
    return JSONResponse({"deleted": file_hash})


# ---------------------------------------------------------------------------
# /api/models/status
# ---------------------------------------------------------------------------

@app.get("/api/models/status")
async def models_status() -> JSONResponse:
    """Check which bundled models are present and ready."""
    from backend.services.model_manager import check_models_status
    return JSONResponse(check_models_status())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_upload(file_id: str) -> Path | None:
    """Find the uploaded file in TEMP_DIR by its file_id prefix."""
    matches = list(TEMP_DIR.glob(f"{file_id}_*"))
    return matches[0] if matches else None


async def _ws_send(ws: WebSocket, data: dict) -> None:
    try:
        await ws.send_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass  # client disconnected; swallow silently


async def _run_with_heartbeat(ws: WebSocket, executor_fn, status_msg: str, interval: float = 5.0):
    """
    Run a blocking function in an executor while sending periodic heartbeat
    messages to keep the WebSocket alive. WKWebView may kill idle connections
    after ~60s, so we ping every 5s during model loading / transcription.
    """
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, executor_fn)

    tick = 0
    while not task.done():
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except asyncio.TimeoutError:
            tick += 1
            await _ws_send(ws, {"type": "status", "message": f"{status_msg} ({tick * interval:.0f}s)"})

    return await task
