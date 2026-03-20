"""Tests for backend/services/audio.py — uses mocking to avoid ffmpeg/file deps."""
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.services.audio import (
    AudioValidationError,
    validate_extension,
    get_audio_duration,
    file_sha256,
    save_upload,
    _is_16k_mono,
)


class TestValidateExtension:
    def test_valid_mp3(self, tmp_path):
        validate_extension(tmp_path / "audio.mp3")  # should not raise

    def test_valid_wav(self, tmp_path):
        validate_extension(tmp_path / "audio.wav")

    def test_valid_m4a(self, tmp_path):
        validate_extension(tmp_path / "audio.m4a")

    def test_valid_mp4(self, tmp_path):
        validate_extension(tmp_path / "audio.mp4")

    def test_invalid_txt(self, tmp_path):
        with pytest.raises(AudioValidationError, match="Unsupported file format"):
            validate_extension(tmp_path / "audio.txt")

    def test_invalid_pdf(self, tmp_path):
        with pytest.raises(AudioValidationError, match="Unsupported file format"):
            validate_extension(tmp_path / "report.pdf")

    def test_case_insensitive(self, tmp_path):
        # Upper case extension should also be accepted
        validate_extension(tmp_path / "audio.MP3")

    def test_error_message_includes_supported_list(self, tmp_path):
        with pytest.raises(AudioValidationError, match="Supported"):
            validate_extension(tmp_path / "file.xyz")


class TestGetAudioDuration:
    def _make_ffprobe_result(self, stdout="3.5\n", returncode=0, stderr=""):
        result = MagicMock()
        result.stdout = stdout
        result.returncode = returncode
        result.stderr = stderr
        return result

    def test_returns_float_on_success(self, tmp_path):
        f = tmp_path / "a.wav"
        with patch("subprocess.run", return_value=self._make_ffprobe_result("3.5\n")):
            dur = get_audio_duration(f)
        assert dur == pytest.approx(3.5)

    def test_raises_on_ffprobe_failure(self, tmp_path):
        f = tmp_path / "bad.wav"
        with patch(
            "subprocess.run",
            return_value=self._make_ffprobe_result(returncode=1, stderr="no such file"),
        ):
            with pytest.raises(AudioValidationError, match="Could not read"):
                get_audio_duration(f)

    def test_raises_on_non_numeric_output(self, tmp_path):
        f = tmp_path / "a.wav"
        with patch("subprocess.run", return_value=self._make_ffprobe_result("N/A\n")):
            with pytest.raises(AudioValidationError, match="duration"):
                get_audio_duration(f)


class TestFileSha256:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "data.wav"
        f.write_bytes(b"hello world")
        assert file_sha256(f) == file_sha256(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        assert file_sha256(f1) != file_sha256(f2)

    def test_returns_64_char_hex_string(self, tmp_path):
        f = tmp_path / "x.wav"
        f.write_bytes(b"data")
        result = file_sha256(f)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


class TestSaveUpload:
    def test_saves_bytes_to_temp_dir(self, tmp_path):
        with patch("backend.services.audio.TEMP_DIR", tmp_path):
            path = save_upload(b"audio data", "recording.wav")
        assert path.read_bytes() == b"audio data"
        assert path.name == "recording.wav"


class TestIs16kMono:
    def _make_ffprobe_result(self, stdout="", returncode=0):
        r = MagicMock()
        r.stdout = stdout
        r.returncode = returncode
        return r

    def test_returns_true_for_16k_mono(self, tmp_path):
        f = tmp_path / "a.wav"
        with patch(
            "subprocess.run",
            return_value=self._make_ffprobe_result("sample_rate=16000\nchannels=1\n"),
        ):
            assert _is_16k_mono(f) is True

    def test_returns_false_for_44k_stereo(self, tmp_path):
        f = tmp_path / "a.wav"
        with patch(
            "subprocess.run",
            return_value=self._make_ffprobe_result("sample_rate=44100\nchannels=2\n"),
        ):
            assert _is_16k_mono(f) is False

    def test_returns_false_on_ffprobe_failure(self, tmp_path):
        f = tmp_path / "a.wav"
        with patch(
            "subprocess.run",
            return_value=self._make_ffprobe_result(returncode=1),
        ):
            assert _is_16k_mono(f) is False
