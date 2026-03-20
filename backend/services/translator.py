"""
Translation service: IndicTrans2 1B on MLX (Apple Silicon native).

Design:
- Module-level singleton with explicit load/unload (same pattern as asr.py).
- Uses the MLX-converted 1B model with beam search decoding.
- Batch inference for throughput (TRANSLATION_BATCH_SIZE segments per call).
- The IndicProcessor is the authoritative preprocessor from ai4bharat.
"""
import gc
import logging
from pathlib import Path
from typing import Sequence

import mlx.core as mx
from IndicTransToolkit.processor import IndicProcessor as _IndicProcessor

from backend.config import (
    TRANSLATION_MODEL_DIR,
    TRANSLATION_BATCH_SIZE,
    WHISPER_TO_INDICTRANS,
)
from backend.services.mlx_translator.model import IndicTrans2, IT2Config
from backend.services.mlx_translator.generate import beam_search
from backend.services.tokenization_indictrans import IndicTransTokenizer as _IndicTransTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_model: IndicTrans2 | None = None
_config: IT2Config | None = None
_tokenizer = None
_processor: _IndicProcessor | None = None
_loaded: bool = False


def load(model_dir: Path = TRANSLATION_MODEL_DIR) -> None:
    """
    Load IndicTrans2 1B MLX model + tokenizer from local directory.
    Idempotent: no-op if already loaded.
    """
    global _model, _config, _tokenizer, _processor, _loaded

    if _loaded:
        logger.debug("Translator already loaded, skipping.")
        return

    if _model is not None:
        unload()

    model_dir = Path(model_dir)
    logger.info("Loading MLX translator from %s", model_dir)

    config = IT2Config.from_model_config(model_dir / "config.json")
    model = IndicTrans2(config)
    weights = mx.load(str(model_dir / "weights.safetensors"))
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())

    # Load tokenizer using vendored class (avoids sys.path mutation)
    tokenizer = _IndicTransTokenizer(
        src_vocab_fp=str(model_dir / "dict.SRC.json"),
        tgt_vocab_fp=str(model_dir / "dict.TGT.json"),
        src_spm_fp=str(model_dir / "model.SRC"),
        tgt_spm_fp=str(model_dir / "model.TGT"),
    )

    _model = model
    _config = config
    _tokenizer = tokenizer
    _processor = _IndicProcessor(inference=True)
    _loaded = True
    logger.info("MLX translator loaded (enc=%d dec=%d d=%d).",
                config.encoder_layers, config.decoder_layers, config.d_model)


def unload() -> None:
    """Release translator from Metal GPU memory."""
    global _model, _config, _tokenizer, _processor, _loaded

    if not _loaded:
        return

    logger.info("Unloading translator.")
    _model = _config = _tokenizer = _processor = None
    _loaded = False

    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    logger.info("Translator unloaded.")


def translate_batch(texts: Sequence[str], src_lang: str) -> list[str]:
    """
    Translate a batch of source-language sentences to English.

    Args:
        texts:    List of source-language strings (e.g. Marathi).
        src_lang: IndicTrans2 source language code (e.g. "mar_Deva").

    Returns:
        List of English translations, same length as input.

    Raises:
        RuntimeError if the model is not loaded.
    """
    if _model is None or _tokenizer is None or _processor is None or _config is None:
        raise RuntimeError("Translator not loaded. Call translator.load() first.")

    texts = list(texts)
    preprocessed = _processor.preprocess_batch(
        texts, src_lang=src_lang, tgt_lang="eng_Latn"
    )

    _tokenizer._switch_to_input_mode()
    enc = _tokenizer(preprocessed, return_tensors="pt", padding=True, truncation=True)
    input_ids = mx.array(enc["input_ids"].numpy())
    attention_mask = mx.array(enc["attention_mask"].numpy())

    enc_out = _model.encode(input_ids, attention_mask=attention_mask)
    mx.eval(enc_out)

    token_ids_batch = beam_search(
        _model, enc_out, attention_mask,
        bos_token_id=_config.eos_token_id,  # decoder_start_token_id = eos_token_id = 2
        eos_token_id=_config.eos_token_id,
        pad_token_id=_config.pad_token_id,
        max_length=256,
        num_beams=5,
        length_penalty=1.0,
    )

    _tokenizer._switch_to_target_mode()
    results = []
    for token_ids in token_ids_batch:
        tokens = [_tokenizer._convert_id_to_token(t) for t in token_ids]
        raw_text = _tokenizer.convert_tokens_to_string(tokens)
        results.append(raw_text)

    return _processor.postprocess_batch(results, lang="eng_Latn")


def translate_segments(
    segments: list[dict],
    whisper_language_code: str,
    batch_size: int = TRANSLATION_BATCH_SIZE,
) -> list[dict]:
    """
    Translate all segments from a Whisper result in batches.

    Args:
        segments:              List of segment dicts from Whisper (must have "text").
        whisper_language_code: e.g. "mr" — used to look up IndicTrans2 src_lang.
        batch_size:            Sentences per forward pass.

    Returns:
        Same segments list, each dict augmented with "english" key.
    """
    src_lang = WHISPER_TO_INDICTRANS[whisper_language_code]
    texts = [seg["text"].strip() for seg in segments]
    translations: list[str] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_translations = translate_batch(batch, src_lang=src_lang)
        translations.extend(batch_translations)

    for seg, eng in zip(segments, translations):
        seg["english"] = eng

    return segments


def translate_single(text: str, whisper_language_code: str) -> str:
    """Translate one sentence. Used for segment re-translation after user edits."""
    src_lang = WHISPER_TO_INDICTRANS[whisper_language_code]
    results = translate_batch([text], src_lang=src_lang)
    return results[0]


def is_loaded() -> bool:
    return _loaded
