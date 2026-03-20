"""
Audio file validation, duration detection, and format conversion.

Invariants:
- All output audio is mono 16kHz WAV (what Whisper expects)
- We never modify the original uploaded file
- Errors are raised as ValueError with user-friendly messages
"""
import hashlib
import json
import logging
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
    Applies the same preprocessing as convert_to_wav (noise reduction,
    normalization) since video audio is often noisier than dedicated recordings.
    The video file is deleted after extraction — only audio is kept.
    """
    audio_out = TEMP_DIR / f"{video_path.stem}_extracted.wav"

    # Step 1: Extract raw audio (no filters yet — we need it for analysis)
    raw_out = TEMP_DIR / f"{video_path.stem}_raw.wav"
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn", "-ar", "16000", "-ac", "1", "-f", "wav",
            str(raw_out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AudioValidationError(
            f"Could not extract audio from video. Details: {result.stderr.strip()}"
        )

    # Step 2: Analyze and apply adaptive filters
    try:
        profile = _analyze_audio(raw_out)
        filter_chain = _build_filter_chain(profile)
    except Exception:
        logger.warning("Audio analysis failed — using raw extraction")
        filter_chain = None

    if filter_chain:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(raw_out),
                "-af", filter_chain,
                "-ar", "16000", "-ac", "1", "-f", "wav",
                str(audio_out),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raw_out.unlink(missing_ok=True)
        else:
            logger.warning("Filtered extraction failed, using raw: %s", result.stderr[:200])
            raw_out.rename(audio_out)
    else:
        # Clean audio — just rename
        raw_out.rename(audio_out)
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


logger = logging.getLogger("vanilipi.audio")


# ---------------------------------------------------------------------------
# Adaptive audio analysis
# ---------------------------------------------------------------------------

def _analyze_audio(path: Path) -> dict:
    """
    Probe audio characteristics using ffmpeg's astats filter.
    Returns metrics used to decide which cleanup filters to apply.

    Key metrics:
      rms_level_db:  average loudness (dB). Quiet = below -30, loud = above -15.
      peak_level_db: max peak. Clipping risk above -1.
      noise_floor_db: estimated noise floor from the quietest 10% of frames.
      dynamic_range_db: difference between peak and noise floor.
      crest_factor_db: peak-to-rms ratio. High = spiky (transient noise).
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", str(path),
            "-af", "astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )

    # Parse per-frame RMS values to estimate noise floor
    rms_values = []
    for line in result.stderr.splitlines() + result.stdout.splitlines():
        if "RMS_level" in line and "=" in line:
            try:
                val = float(line.split("=")[-1].strip())
                if val > -100:  # ignore -inf
                    rms_values.append(val)
            except ValueError:
                pass

    # Fallback: use astats summary from stderr
    stats = _parse_astats_summary(result.stderr)

    if rms_values:
        rms_values.sort()
        n = len(rms_values)
        # Noise floor = average of quietest 10% of frames
        quiet_count = max(1, n // 10)
        noise_floor = sum(rms_values[:quiet_count]) / quiet_count
        overall_rms = sum(rms_values) / n
        peak = max(rms_values)
    else:
        noise_floor = stats.get("rms_trough", -60.0)
        overall_rms = stats.get("rms_level", -25.0)
        peak = stats.get("peak_level", -3.0)

    dynamic_range = peak - noise_floor
    crest_factor = peak - overall_rms

    profile = {
        "rms_level_db": round(overall_rms, 1),
        "peak_level_db": round(peak, 1),
        "noise_floor_db": round(noise_floor, 1),
        "dynamic_range_db": round(dynamic_range, 1),
        "crest_factor_db": round(crest_factor, 1),
    }
    logger.info("Audio profile: %s", profile)
    return profile


def _parse_astats_summary(stderr: str) -> dict:
    """Extract key stats from ffmpeg astats stderr output."""
    stats: dict = {}
    for line in stderr.splitlines():
        lower = line.lower().strip()
        if "rms level" in lower:
            try:
                stats["rms_level"] = float(lower.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "peak level" in lower:
            try:
                stats["peak_level"] = float(lower.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "rms trough" in lower:
            try:
                stats["rms_trough"] = float(lower.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return stats


def _build_filter_chain(profile: dict) -> str | None:
    """
    Choose ffmpeg audio filter chain based on the noise profile.

    Returns an -af filter string, or None if no processing is needed.

    Classification:
      CLEAN:  noise_floor < -45 dB, dynamic_range > 30 dB
              → no filters needed, Whisper handles this well
      MILD:   noise_floor -45 to -30 dB
              → gentle highpass + light noise reduction
      NOISY:  noise_floor -30 to -20 dB (indoor crowd, AC, fan)
              → highpass + moderate noise reduction
      HEAVY:  noise_floor > -20 dB (outdoor traffic, wind, construction)
              → aggressive highpass + strong noise reduction + dynamic normalization
    """
    noise_floor = profile.get("noise_floor_db", -60.0)
    dynamic_range = profile.get("dynamic_range_db", 40.0)

    if noise_floor < -50 and dynamic_range > 35:
        # Clean studio audio — don't touch it
        logger.info("Audio classification: CLEAN (noise=%.1f dB, DR=%.1f dB) — no filters", noise_floor, dynamic_range)
        return None

    if noise_floor < -40 and dynamic_range > 25:
        # Mild noise (quiet room, light AC) — gentle cleanup
        logger.info("Audio classification: MILD (noise=%.1f dB, DR=%.1f dB)", noise_floor, dynamic_range)
        return "highpass=f=80,afftdn=nf=-30"

    if noise_floor < -30 or dynamic_range > 15:
        # Moderate noise (crowd, outdoor with some traffic) — stronger cleanup
        logger.info("Audio classification: NOISY (noise=%.1f dB, DR=%.1f dB)", noise_floor, dynamic_range)
        return "highpass=f=100,lowpass=f=7500,afftdn=nf=-25"

    # Heavy noise (busy road, construction, strong wind) — full treatment
    logger.info("Audio classification: HEAVY (noise=%.1f dB, DR=%.1f dB)", noise_floor, dynamic_range)
    return (
        "highpass=f=120,"
        "lowpass=f=7000,"
        "afftdn=nf=-20:tn=1,"
        "dynaudnorm=f=200:g=15:p=0.9"
    )


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_to_wav(path: Path) -> Path:
    """
    Convert any supported audio file to 16kHz mono WAV.
    Adaptively applies audio cleanup based on the file's noise profile.
    """
    output = TEMP_DIR / f"{path.stem}_converted.wav"

    # Analyze noise characteristics
    try:
        profile = _analyze_audio(path)
        filter_chain = _build_filter_chain(profile)
    except Exception:
        logger.warning("Audio analysis failed — using plain conversion")
        filter_chain = None

    if filter_chain:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(path),
                "-af", filter_chain,
                "-ar", "16000", "-ac", "1", "-f", "wav",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return output
        logger.warning("Filtered conversion failed, falling back to plain: %s", result.stderr[:200])

    # Plain conversion (fallback or clean audio)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(path),
            "-ar", "16000", "-ac", "1", "-f", "wav",
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
        except OSError as exc:
            logger.warning("Failed to clean up chunk %s: %s", p, exc)


def save_upload(data: bytes, filename: str) -> Path:
    """Persist uploaded bytes to TEMP_DIR with original filename. Returns path."""
    dest = TEMP_DIR / filename
    dest.write_bytes(data)
    return dest
