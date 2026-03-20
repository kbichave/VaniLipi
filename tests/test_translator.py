"""
Tests for backend/services/translator.py — mocks PyTorch/IndicTrans2 to avoid
model downloads. Focus: load/unload lifecycle, batch slicing, language mapping.
"""
from unittest.mock import MagicMock, patch, call
import sys

import pytest

import backend.services.translator as translator_module
from backend.services.translator import (
    is_loaded,
    unload,
    translate_segments,
    translate_single,
)
from backend.config import TRANSLATION_BATCH_SIZE


def _reset():
    translator_module._model = None
    translator_module._tokenizer = None
    translator_module._processor = None
    translator_module._device = None
    translator_module._loaded_model_id = None


# ---------------------------------------------------------------------------
# Helpers to fake the heavy imports
# ---------------------------------------------------------------------------
def _make_torch_mock(device="mps"):
    torch_mock = MagicMock()
    torch_mock.backends.mps.is_available.return_value = (device == "mps")
    torch_mock.float16 = "float16"
    # model.to() returns the model itself
    model_mock = MagicMock()
    model_mock.to.return_value = model_mock
    return torch_mock, model_mock


class TestIsLoaded:
    def setup_method(self):
        _reset()

    def test_false_when_not_loaded(self):
        assert is_loaded() is False

    def test_true_when_loaded(self):
        translator_module._model = MagicMock()
        assert is_loaded() is True


class TestUnload:
    def setup_method(self):
        _reset()

    def test_unload_clears_all_state(self):
        translator_module._model = MagicMock()
        translator_module._tokenizer = MagicMock()
        translator_module._processor = MagicMock()
        translator_module._device = "mps"
        translator_module._loaded_model_id = "some-model"

        torch_mock = MagicMock()
        torch_mock.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": torch_mock}):
            unload()

        assert translator_module._model is None
        assert translator_module._tokenizer is None
        assert translator_module._processor is None
        assert translator_module._device is None
        assert translator_module._loaded_model_id is None

    def test_unload_is_safe_when_not_loaded(self):
        unload()  # should not raise


class TestTranslateNotLoaded:
    def setup_method(self):
        _reset()

    def test_translate_single_raises_when_not_loaded(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            translate_single("मला मराठी आवडते.", "mr")


class TestTranslateSegments:
    def setup_method(self):
        _reset()

    def _install_loaded_translator(self, translations: list[str]):
        """
        Inject a fake loaded translator that returns `translations` in sequence.
        """
        processor = MagicMock()
        processor.preprocess_batch.side_effect = lambda texts, **kw: texts
        processor.postprocess_batch.side_effect = lambda decoded, **kw: decoded

        # The tokenizer is called as tokenizer(...) which returns an object,
        # and then .to(_device) is called on that object — so tokenizer()
        # must return a MagicMock (which has .to() by default).
        inputs_mock = MagicMock()
        tokenizer = MagicMock()
        tokenizer.return_value = inputs_mock
        tokenizer.batch_decode.return_value = translations

        model = MagicMock()
        model.generate.return_value = MagicMock()

        translator_module._model = model
        translator_module._tokenizer = tokenizer
        translator_module._processor = processor
        translator_module._device = "cpu"
        translator_module._loaded_model_id = "fake-model"

    def test_augments_segments_with_english(self):
        segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "हा एक चाचणी आहे."},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "आज हवामान छान आहे."},
        ]
        translations = ["This is a test.", "The weather is nice today."]
        self._install_loaded_translator(translations)

        result = translate_segments(segments, "mr")

        for seg in result:
            assert "english" in seg

    def test_segments_count_preserved(self):
        n = 5
        segs = [{"id": i, "start": float(i), "end": float(i+1), "text": f"text{i}"} for i in range(n)]
        translations = [f"eng{i}" for i in range(n)]
        self._install_loaded_translator(translations)

        result = translate_segments(segs, "hi")

        assert len(result) == n

    def test_unsupported_language_raises_key_error(self):
        # "en" is not in WHISPER_TO_INDICTRANS
        _reset()
        translator_module._model = MagicMock()  # pretend loaded
        with pytest.raises(KeyError):
            translate_segments([], "en")
