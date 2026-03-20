"""
Model manager: verify bundled models are present.

All models are bundled locally in the repo under models/:
    ASR:         models/asr/whisper-large-v3/          (mlx-whisper large-v3, MLX fp16)
    Translation: models/translation/indictrans2-1b/    (IndicTrans2 1B, MLX fp16)
"""
from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

from backend.config import ASR_MODEL, TRANSLATION_MODEL_DIR

logger = logging.getLogger(__name__)


def check_models_status() -> dict[str, Any]:
    """
    Return whether each bundled model is present and complete.
    """
    asr_ok = _check_asr_model(Path(ASR_MODEL))
    translation_ok = _check_translation_model(TRANSLATION_MODEL_DIR)

    return {
        "models": [
            {
                "id": "whisper-large-v3",
                "label": "Whisper large-v3 (ASR)",
                "type": "asr",
                "path": ASR_MODEL,
                "ready": asr_ok,
            },
            {
                "id": "indictrans2-1b",
                "label": "IndicTrans2 1B (Translation)",
                "type": "translation",
                "path": str(TRANSLATION_MODEL_DIR),
                "ready": translation_ok,
            },
        ],
        "ready": asr_ok and translation_ok,
    }


def _check_asr_model(model_dir: Path) -> bool:
    """ASR model needs config.json and a weights file."""
    if not model_dir.is_dir():
        logger.warning("ASR model directory missing: %s", model_dir)
        return False
    has_config = (model_dir / "config.json").is_file()
    has_weights = (
        (model_dir / "weights.npz").is_file()
        or (model_dir / "weights.safetensors").is_file()
    )
    if not (has_config and has_weights):
        logger.warning("ASR model incomplete in %s: config=%s weights=%s", model_dir, has_config, has_weights)
    return has_config and has_weights


def _check_translation_model(model_dir: Path) -> bool:
    """Translation model needs MLX weights, config, and tokenizer files."""
    if not model_dir.is_dir():
        logger.warning("Translation model directory missing: %s", model_dir)
        return False
    required = ["config.json", "weights.safetensors", "dict.SRC.json", "dict.TGT.json", "model.SRC", "model.TGT"]
    missing = [f for f in required if not (model_dir / f).is_file()]
    if missing:
        logger.warning("Translation model incomplete in %s: missing %s", model_dir, missing)
        return False
    return True
