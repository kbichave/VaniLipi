"""Tests for backend/config.py — no ML model required."""
import pytest
from backend.config import (
    LANGUAGE_MAP,
    WHISPER_TO_INDICTRANS,
    SUPPORTED_WHISPER_CODES,
    SUPPORTED_AUDIO_EXTENSIONS,
    TRANSLATION_BATCH_SIZE,
)


class TestLanguageMap:
    def test_marathi_is_default_present(self):
        assert "mr" in LANGUAGE_MAP

    def test_all_14_languages_present(self):
        expected = {"mr", "hi", "bn", "ta", "te", "gu", "kn", "ml", "pa", "ur", "ne", "sd", "as", "sa"}
        assert expected == set(LANGUAGE_MAP.keys())

    def test_each_language_has_required_fields(self):
        required = {"name", "script", "indictrans_code", "quality", "recommended"}
        for code, info in LANGUAGE_MAP.items():
            missing = required - set(info.keys())
            assert not missing, f"Language {code} missing fields: {missing}"

    def test_quality_values_are_valid(self):
        valid = {"very_good", "good", "fair", "poor"}
        for code, info in LANGUAGE_MAP.items():
            assert info["quality"] in valid, f"Language {code} has invalid quality '{info['quality']}'"

    def test_indictrans_codes_have_expected_format(self):
        # IndicTrans2 codes follow pattern like "mar_Deva", "hin_Deva"
        for code, info in LANGUAGE_MAP.items():
            it_code = info["indictrans_code"]
            parts = it_code.split("_")
            assert len(parts) == 2, f"{code}: IndicTrans2 code '{it_code}' should be 'xxx_Yyyy'"
            assert parts[0].islower(), f"{code}: IndicTrans2 lang prefix should be lowercase"
            assert parts[1][0].isupper(), f"{code}: IndicTrans2 script suffix should start uppercase"

    def test_recommended_languages_are_subset(self):
        recommended = [c for c, i in LANGUAGE_MAP.items() if i["recommended"]]
        assert set(recommended) <= SUPPORTED_WHISPER_CODES

    def test_marathi_is_recommended(self):
        assert LANGUAGE_MAP["mr"]["recommended"] is True

    def test_hindi_quality_is_very_good(self):
        assert LANGUAGE_MAP["hi"]["quality"] == "very_good"


class TestWhisperToIndictrans:
    def test_all_whisper_codes_map(self):
        assert set(WHISPER_TO_INDICTRANS.keys()) == set(LANGUAGE_MAP.keys())

    def test_marathi_maps_correctly(self):
        assert WHISPER_TO_INDICTRANS["mr"] == "mar_Deva"

    def test_hindi_maps_correctly(self):
        assert WHISPER_TO_INDICTRANS["hi"] == "hin_Deva"

    def test_no_duplicate_indictrans_codes(self):
        codes = list(WHISPER_TO_INDICTRANS.values())
        assert len(codes) == len(set(codes)), "Duplicate IndicTrans2 codes found"


class TestSupportedExtensions:
    def test_common_formats_present(self):
        for ext in (".mp3", ".wav", ".m4a", ".flac", ".ogg"):
            assert ext in SUPPORTED_AUDIO_EXTENSIONS

    def test_video_containers_present(self):
        for ext in (".mp4", ".mkv"):
            assert ext in SUPPORTED_AUDIO_EXTENSIONS

    def test_all_lowercase(self):
        for ext in SUPPORTED_AUDIO_EXTENSIONS:
            assert ext == ext.lower(), f"Extension {ext} is not lowercase"


class TestBatchSize:
    def test_batch_size_is_positive(self):
        assert TRANSLATION_BATCH_SIZE > 0

    def test_batch_size_is_reasonable(self):
        # Too small (1) is inefficient; too large (>64) risks OOM
        assert 4 <= TRANSLATION_BATCH_SIZE <= 64
