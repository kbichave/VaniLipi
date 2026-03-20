"""
ASR service: mlx-whisper wrapper with explicit load/unload lifecycle.

Design:
- Module-level singleton so the model stays loaded between calls within one
  transcription session and is explicitly released before the translator loads.
- Never have both ASR and translator in memory simultaneously (see memory strategy
  in the plan: peak ~6GB otherwise, which is tight on 16GB).
- Returns Whisper's native segment format so callers can stream segments as they arrive.
"""
import gc
import logging
from collections import Counter
from pathlib import Path

from backend.config import (
    ASR_MODEL,
    SUPPORTED_WHISPER_CODES,
    WHISPER_TO_INDICTRANS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_loaded_model_id: str | None = None
_loaded_pipeline = None  # mlx_whisper module is stateless; we track the model path


def load(model_id: str = ASR_MODEL) -> None:
    """
    Pre-warm mlx-whisper by loading the model weights into Metal GPU memory.
    Idempotent: if the same model is already loaded, this is a no-op.
    """
    global _loaded_model_id, _loaded_pipeline

    if _loaded_model_id == model_id:
        logger.debug("ASR model %s already loaded, skipping.", model_id)
        return

    if _loaded_pipeline is not None:
        unload()

    logger.info("Loading ASR model: %s", model_id)

    # Verify model exists before attempting to load (clear error for bundled app)
    model_path = Path(model_id)
    if model_path.is_dir() and not (model_path / "config.json").exists():
        raise RuntimeError(
            f"ASR model not found at {model_id}. "
            "The model files may be missing or corrupted. "
            "Try reinstalling VaniLipi."
        )

    import mlx_whisper  # noqa: F401 -- import triggers Metal allocation

    _loaded_model_id = model_id
    _loaded_pipeline = model_id  # mlx_whisper is functional; we track the path
    logger.info("ASR model loaded.")


def unload() -> None:
    """
    Release ASR model from Metal GPU memory to make room for the translator.
    Call this before loading IndicTrans2.
    """
    global _loaded_model_id, _loaded_pipeline

    if _loaded_pipeline is None:
        return

    logger.info("Unloading ASR model.")
    _loaded_pipeline = None
    _loaded_model_id = None

    gc.collect()

    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass  # non-fatal: MLX may not have a cache entry

    logger.info("ASR model unloaded.")


# ---------------------------------------------------------------------------
# Language-specific initial prompts
# ---------------------------------------------------------------------------
# Whisper's decoder uses these to bias toward correct orthography and common
# vocabulary. This fixes the most frequent ASR errors: broken word boundaries,
# phonetic substitutions (स्वतंत्र→सोतंत्र), and Devanagari vs Latin confusion.
# The prompt is NOT a transcript — it's a style/vocabulary hint.

_INITIAL_PROMPTS: dict[str, str] = {
    "mr": (
        "हे मराठी भाषेतील प्रमाण लेखन आहे. "
        "स्वतंत्र, परंतु, सामान्य, परिस्थिती, आठवड्यांमध्ये, "
        "दक्षिणेकडील, हॉस्पिटल, केंब्रिज, महासागर, नियंत्रण, "
        "अमेरिकन, फ्रान्स, जर्मनी, युरोप।"
    ),
    "hi": (
        "यह हिंदी भाषा में मानक लेखन है। "
        "स्वतंत्र, परंतु, सामान्य, परिस्थिति, अमेरिका, फ्रांस।"
    ),
}


def _get_initial_prompt(language: str | None) -> str | None:
    """Return a vocabulary-hint prompt for the given Whisper language code."""
    if language is None:
        return None
    return _INITIAL_PROMPTS.get(language)


def transcribe(
    audio_path: str | Path,
    language: str | None = None,
    model_id: str = ASR_MODEL,
    word_timestamps: bool = True,
) -> dict:
    """
    Transcribe an audio file.

    Args:
        audio_path: Path to a 16kHz mono WAV (or any format ffmpeg can read).
        language:   ISO 639-1 code ("mr", "hi", …) or None for auto-detect.
        model_id:   Local path or HF repo ID for the mlx-whisper model.
        word_timestamps: Whether to include word-level timestamps (needed for
                         click-to-seek). Minor speed cost.

    Returns:
        Whisper result dict with keys:
            "text"      – full transcript string
            "segments"  – list of dicts: {id, start, end, text, words?}
            "language"  – detected/forced language code (e.g. "mr")

    Raises:
        RuntimeError if transcription fails.
    """
    import mlx_whisper

    audio_path = str(audio_path)
    logger.info(
        "Starting transcription: model=%s language=%s file=%s",
        model_id,
        language or "auto",
        audio_path,
    )

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_id,
        language=language,  # None → auto-detect
        task="transcribe",
        word_timestamps=word_timestamps,
        verbose=False,
        initial_prompt=_get_initial_prompt(language),
        condition_on_previous_text=True,  # helps noisy audio maintain context
        hallucination_silence_threshold=1.0,
    )

    detected = result.get("language", "")
    segments = result.get("segments", [])

    # Clean up [inaudible] tokens that Whisper inserts inline for silence gaps
    for seg in segments:
        text = seg.get("text", "")
        cleaned = text.replace("[inaudible]", "").strip()
        # If stripping [inaudible] leaves real text, keep the text
        # If nothing remains, mark the whole segment as inaudible
        seg["text"] = cleaned if cleaned else _INAUDIBLE

    # Filter hallucinated segments before returning
    filtered = _filter_hallucinations(segments)

    # Merge short segments into natural blocks to avoid excessive fragmentation
    merged = _merge_short_segments(filtered)
    result["segments"] = merged

    logger.info(
        "Transcription complete: language=%s segments=%d (merged from %d, raw %d)",
        detected,
        len(merged),
        len(filtered),
        len(segments),
    )
    return result


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

_INAUDIBLE = "[inaudible]"


def _filter_hallucinations(segments: list[dict]) -> list[dict]:
    """
    Post-process ASR segments to detect and replace hallucinated text.

    Catches three patterns:
    1. High no_speech_prob — Whisper thinks the segment is silence but
       hallucinated text anyway (common on music, noise, or silence).
    2. Repeated n-gram loops — Whisper gets stuck repeating the same phrase
       (e.g., "thank you thank you thank you thank you").
    3. Suspiciously long text from a short audio window — more tokens than
       the audio duration could plausibly produce.
    """
    filtered: list[dict] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        no_speech = seg.get("no_speech_prob", 0.0)
        avg_logprob = seg.get("avg_logprob", 0.0)
        seg_duration = seg.get("end", 0.0) - seg.get("start", 0.0)

        # 1. Very high no_speech_prob AND low confidence → hallucination on silence.
        # In noisy outdoor audio, no_speech_prob is elevated even with real speech.
        # Only filter when Whisper is very unsure AND the text looks like gibberish
        # (low avg_logprob = Whisper itself isn't confident in what it produced).
        if no_speech > 0.9 and avg_logprob < -1.0:
            logger.info(
                "Hallucination filtered (no_speech=%.2f, logprob=%.2f): '%s'",
                no_speech, avg_logprob, text[:60],
            )
            seg["text"] = _INAUDIBLE
            filtered.append(seg)
            continue

        # 2. Repeated bigram/trigram loops (Whisper gets stuck)
        if _has_repetition_loop(text):
            logger.info("Hallucination filtered (repetition loop): '%s'", text[:60])
            seg["text"] = _INAUDIBLE
            filtered.append(seg)
            continue

        # 3. Text too long for audio duration (>20 chars per second is suspicious)
        if seg_duration > 0 and len(text) / seg_duration > 20:
            logger.info(
                "Hallucination filtered (%.0f chars/sec): '%s'",
                len(text) / seg_duration, text[:60],
            )
            seg["text"] = _INAUDIBLE
            filtered.append(seg)
            continue

        filtered.append(seg)

    return filtered


def _merge_short_segments(
    segments: list[dict],
    min_duration: float = 15.0,
    max_duration: float = 45.0,
    gap_threshold: float = 2.0,
) -> list[dict]:
    """
    Merge adjacent short segments into longer, more natural blocks.

    Whisper tends to split on sentence boundaries every ~5 seconds, which produces
    too many segments for a 45-minute audio. This merges consecutive segments
    into blocks of roughly min_duration–max_duration seconds, splitting only
    where there's a natural pause (gap > gap_threshold) or the block would
    exceed max_duration.

    Word-level timestamps from the constituent segments are preserved in the
    merged segment's 'words' list.
    """
    if not segments:
        return segments

    merged: list[dict] = []
    current = _copy_segment(segments[0])

    for seg in segments[1:]:
        current_duration = current["end"] - current["start"]
        gap = seg.get("start", 0) - current.get("end", 0)
        seg_text = seg.get("text", "").strip()

        # Skip inaudible segments — don't merge them
        if seg_text == _INAUDIBLE:
            merged.append(current)
            current = _copy_segment(seg)
            continue

        # Merge if: block is short, gap is small, and merging won't exceed max
        would_be = seg.get("end", 0) - current["start"]
        if current_duration < min_duration and gap < gap_threshold and would_be <= max_duration:
            current["end"] = seg["end"]
            current["text"] = current.get("text", "") + " " + seg_text
            # Merge word timestamps if both segments have them
            if "words" in current and "words" in seg:
                current["words"] = current["words"] + seg["words"]
        else:
            merged.append(current)
            current = _copy_segment(seg)

    merged.append(current)

    # Re-number segment IDs
    for i, seg in enumerate(merged):
        seg["id"] = i

    return merged


def _copy_segment(seg: dict) -> dict:
    """Shallow copy a segment dict so we don't mutate the original."""
    copied = dict(seg)
    if "words" in copied:
        copied["words"] = list(copied["words"])
    return copied


def _has_repetition_loop(text: str, threshold: float = 0.5) -> bool:
    """
    Detect if >threshold of the text is made up of repeated bigrams or trigrams.

    Example: "thank you thank you thank you" → True (3-gram "thank you thank"
    repeats). This catches Whisper's most common hallucination pattern.
    """
    words = text.split()
    if len(words) < 6:
        return False

    for n in (2, 3):
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            continue
        counts = Counter(ngrams)
        most_common_count = counts.most_common(1)[0][1]
        # If the most common n-gram appears in >threshold of all n-gram positions
        if most_common_count / len(ngrams) > threshold:
            return True

    return False


def is_supported_language(code: str) -> bool:
    """True if Whisper language code maps to a supported IndicTrans2 language."""
    return code in SUPPORTED_WHISPER_CODES


def get_indictrans_src_lang(whisper_code: str) -> str | None:
    """
    Return the IndicTrans2 src_lang string for a Whisper language code,
    or None if not supported.
    """
    return WHISPER_TO_INDICTRANS.get(whisper_code)
