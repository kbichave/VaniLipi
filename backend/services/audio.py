"""
Audio file validation, duration detection, and format conversion.

Invariants:
- All output audio is mono 16kHz WAV (what Whisper expects)
- We never modify the original uploaded file
- Errors are raised as ValueError with user-friendly messages
"""
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.config import (
    MAX_AUDIO_DURATION_SECONDS,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_UPLOAD_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    TEMP_DIR,
)


class AudioValidationError(ValueError):
    """Raised when audio cannot be processed."""


def validate_extension(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise AudioValidationError(
            f"Unsupported file format '{path.suffix}'. Supported: {supported}"
        )


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def extract_audio_from_video(video_path: Path) -> Path:
    """
    Extract audio track from a video file as 16kHz mono WAV.
    The video file is deleted after extraction — only audio is kept.
    """
    audio_out = TEMP_DIR / f"{video_path.stem}_extracted.wav"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vn",          # discard video stream
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            str(audio_out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioValidationError(
            f"Could not extract audio from video. Details: {result.stderr.strip()}"
        )
    # Delete the video — we only store audio
    try:
        video_path.unlink()
    except OSError:
        pass
    return audio_out


def get_audio_duration(path: Path) -> float:
    """Return duration in seconds. Tries ffprobe first, falls back to librosa/soundfile."""
    # Try ffprobe (fast, handles all formats)
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except FileNotFoundError:
        pass  # ffprobe not installed — fall through to Python-only method

    # Fallback: use soundfile (works for WAV, FLAC, OGG) or librosa (handles MP3, M4A etc.)
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.duration
    except Exception:
        pass

    try:
        import librosa
        duration = librosa.get_duration(path=str(path))
        return duration
    except Exception:
        pass

    raise AudioValidationError(
        "Could not determine audio duration. "
        "Install ffmpeg (brew install ffmpeg) for best format support."
    )


def convert_to_wav(path: Path) -> Path:
    """
    Convert any supported audio file to 16kHz mono WAV in the app temp dir.
    Returns path to the converted WAV. The caller is responsible for cleanup.
    """
    output = TEMP_DIR / f"{path.stem}_converted.wav"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",           # overwrite without asking
            "-i", str(path),
            "-ar", "16000", # 16kHz sample rate (Whisper requirement)
            "-ac", "1",     # mono
            "-f", "wav",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioValidationError(
            f"Could not convert audio file. Details: {result.stderr.strip()}"
        )
    return output


def validate_and_prepare(path: Path) -> tuple[Path, float]:
    """
    Full pipeline: validate extension, check duration, convert to WAV.

    Returns:
        (wav_path, duration_seconds)

    The returned wav_path is in TEMP_DIR. The caller owns cleanup.
    """
    validate_extension(path)

    duration = get_audio_duration(path)
    if duration > MAX_AUDIO_DURATION_SECONDS:
        hours = MAX_AUDIO_DURATION_SECONDS / 3600
        raise AudioValidationError(
            f"Audio is too long ({duration/3600:.1f}h). Maximum supported: {hours:.0f}h."
        )
    if duration < 0.1:
        raise AudioValidationError("Audio file appears to be empty (duration < 0.1s).")

    # If already a 16kHz mono wav, skip conversion to save time
    if path.suffix.lower() == ".wav" and _is_16k_mono(path):
        return path, duration

    wav_path = convert_to_wav(path)
    return wav_path, duration


def _is_16k_mono(path: Path) -> bool:
    """Check if a WAV is already 16kHz mono to avoid unnecessary re-encoding."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    lines = result.stdout.strip().splitlines()
    info = dict(line.split("=") for line in lines if "=" in line)
    return info.get("sample_rate") == "16000" and info.get("channels") == "1"


def file_sha256(path: Path) -> str:
    """SHA-256 of file contents. Used as the project cache key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def split_audio_chunks(wav_path: Path, chunk_seconds: float = 300.0) -> list[Path]:
    """
    Split a WAV file into time-based chunks for interleaved ASR + translation.

    This enables progressive processing of long files: transcribe a chunk,
    translate it, stream results, then move to the next chunk — instead of
    waiting for the entire file to finish ASR before any translation begins.

    Args:
        wav_path:       Path to 16kHz mono WAV.
        chunk_seconds:  Duration of each chunk in seconds (default 5 minutes).

    Returns:
        List of chunk file paths in TEMP_DIR, ordered by time.
        If the file is shorter than chunk_seconds, returns [wav_path] unchanged.
    """
    duration = get_audio_duration(wav_path)
    if duration <= chunk_seconds:
        return [wav_path]

    chunk_paths: list[Path] = []
    start = 0.0
    chunk_idx = 0

    while start < duration:
        chunk_path = TEMP_DIR / f"{wav_path.stem}_chunk{chunk_idx:03d}.wav"
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(wav_path),
                "-ss", str(start),
                "-t", str(chunk_seconds),
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                str(chunk_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AudioValidationError(
                f"Failed to split audio at {start:.0f}s: {result.stderr.strip()}"
            )
        chunk_paths.append(chunk_path)
        start += chunk_seconds
        chunk_idx += 1

    return chunk_paths


def cleanup_chunks(chunk_paths: list[Path]) -> None:
    """Remove temporary chunk files created by split_audio_chunks."""
    for p in chunk_paths:
        try:
            if p.exists() and "_chunk" in p.name:
                p.unlink()
        except OSError:
            pass


def save_upload(data: bytes, filename: str) -> Path:
    """Persist uploaded bytes to TEMP_DIR with original filename. Returns path."""
    dest = TEMP_DIR / filename
    dest.write_bytes(data)
    return dest
