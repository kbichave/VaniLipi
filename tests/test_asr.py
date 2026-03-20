"""Tests for backend/services/asr.py — mocks mlx_whisper to avoid model downloads."""
from unittest.mock import MagicMock, patch

import pytest

import backend.services.asr as asr_module
from backend.services.asr import (
    is_supported_language,
    get_indictrans_src_lang,
    get_model_id,
    unload,
)
from backend.config import ASR_MODEL_TURBO, ASR_MODEL_SMALL


class TestIsSupportedLanguage:
    def test_marathi_supported(self):
        assert is_supported_language("mr") is True

    def test_hindi_supported(self):
        assert is_supported_language("hi") is True

    def test_english_not_supported(self):
        # English is not an Indic language supported by IndicTrans2
        assert is_supported_language("en") is False

    def test_french_not_supported(self):
        assert is_supported_language("fr") is False

    def test_all_14_languages_supported(self):
        for code in ("mr", "hi", "bn", "ta", "te", "gu", "kn", "ml", "pa", "ur", "ne", "sd", "as", "sa"):
            assert is_supported_language(code), f"{code} should be supported"


class TestGetIndictransSrcLang:
    def test_marathi_returns_mar_deva(self):
        assert get_indictrans_src_lang("mr") == "mar_Deva"

    def test_hindi_returns_hin_deva(self):
        assert get_indictrans_src_lang("hi") == "hin_Deva"

    def test_unsupported_returns_none(self):
        assert get_indictrans_src_lang("en") is None
        assert get_indictrans_src_lang("fr") is None
        assert get_indictrans_src_lang("xyz") is None


class TestGetModelId:
    def test_best_returns_turbo(self):
        assert get_model_id("best") == ASR_MODEL_TURBO

    def test_fast_returns_small(self):
        assert get_model_id("fast") == ASR_MODEL_SMALL

    def test_unknown_defaults_to_turbo(self):
        assert get_model_id("unknown") == ASR_MODEL_TURBO


class TestLoadUnload:
    def setup_method(self):
        # Ensure clean state before each test
        asr_module._loaded_model_id = None
        asr_module._loaded_pipeline = None

    def test_load_sets_model_id(self):
        with patch.dict("sys.modules", {"mlx_whisper": MagicMock()}):
            asr_module.load(ASR_MODEL_TURBO)
        assert asr_module._loaded_model_id == ASR_MODEL_TURBO

    def test_load_is_idempotent(self):
        # Second call with same model should not reset state
        with patch.dict("sys.modules", {"mlx_whisper": MagicMock()}):
            asr_module.load(ASR_MODEL_TURBO)
            first_id = asr_module._loaded_model_id
            asr_module.load(ASR_MODEL_TURBO)
            assert asr_module._loaded_model_id == first_id

    def test_unload_clears_state(self):
        asr_module._loaded_model_id = ASR_MODEL_TURBO
        asr_module._loaded_pipeline = "some_model"
        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": MagicMock()}):
            unload()
        assert asr_module._loaded_model_id is None
        assert asr_module._loaded_pipeline is None

    def test_unload_is_idempotent_when_not_loaded(self):
        # Should not raise even when nothing is loaded
        unload()
        unload()


class TestTranscribe:
    def setup_method(self):
        asr_module._loaded_model_id = None
        asr_module._loaded_pipeline = None

    def test_transcribe_calls_mlx_whisper(self, tmp_path):
        fake_result = {
            "text": "हा एक चाचणी आहे.",
            "segments": [{"id": 0, "start": 0.0, "end": 2.3, "text": "हा एक चाचणी आहे."}],
            "language": "mr",
        }
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = fake_result

        audio = tmp_path / "test.wav"
        audio.write_bytes(b"")

        with patch.dict("sys.modules", {"mlx_whisper": mock_mlx}):
            result = asr_module.transcribe(str(audio), language="mr")

        mock_mlx.transcribe.assert_called_once()
        assert result["language"] == "mr"
        assert len(result["segments"]) == 1

    def test_transcribe_passes_none_language_for_autodetect(self, tmp_path):
        mock_mlx = MagicMock()
        mock_mlx.transcribe.return_value = {
            "text": "", "segments": [], "language": "hi"
        }
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"")

        with patch.dict("sys.modules", {"mlx_whisper": mock_mlx}):
            asr_module.transcribe(str(audio), language=None)

        call_kwargs = mock_mlx.transcribe.call_args
        assert call_kwargs.kwargs.get("language") is None
