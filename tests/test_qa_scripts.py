"""
Smoke tests for the QA helper scripts.
These verify the scripts are structurally correct and their pure-Python logic
works without network access or real audio files.
"""
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestEvaluateAsr:
    def test_load_ground_truth_parses_tsv(self, tmp_path):
        from tests.evaluate_asr import load_ground_truth

        gt = tmp_path / "ground_truth.tsv"
        gt.write_text("file_id\ttranscript\nmr_001\tहा एक चाचणी आहे.\nmr_002\tआज छान आहे.\n", encoding="utf-8")

        result = load_ground_truth(str(tmp_path))
        assert result == {
            "mr_001": "हा एक चाचणी आहे.",
            "mr_002": "आज छान आहे.",
        }

    def test_load_ground_truth_returns_empty_for_missing_file(self, tmp_path):
        from tests.evaluate_asr import load_ground_truth

        result = load_ground_truth(str(tmp_path))
        assert result == {}

    def test_normalize_text_strips_punctuation(self):
        from tests.evaluate_asr import normalize_text

        assert normalize_text("हा एक चाचणी आहे।") == "हा एक चाचणी आहे"

    def test_normalize_text_lowercases(self):
        from tests.evaluate_asr import normalize_text

        assert normalize_text("Hello World") == "hello world"

    def test_normalize_text_collapses_whitespace(self):
        from tests.evaluate_asr import normalize_text

        assert normalize_text("  one   two  ") == "one two"

    def test_evaluate_dataset_skips_missing_dir(self, tmp_path, capsys):
        from tests.evaluate_asr import evaluate_dataset
        with patch("tests.evaluate_asr.TEST_DATA_DIR", str(tmp_path)):
            result = evaluate_dataset("nonexistent_dataset")
        assert result is None
        captured = capsys.readouterr()
        assert "skip" in captured.out

    def test_evaluate_dataset_skips_when_no_ground_truth(self, tmp_path, capsys):
        from tests.evaluate_asr import evaluate_dataset

        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()
        (dataset_dir / "audio").mkdir()

        with patch("tests.evaluate_asr.TEST_DATA_DIR", str(tmp_path)):
            result = evaluate_dataset("test_dataset")
        assert result is None


class TestDataDownloader:
    def test_ensure_dir_creates_directory(self, tmp_path):
        from tests.data_downloader import ensure_dir

        target = str(tmp_path / "a" / "b" / "c")
        ensure_dir(target)
        assert Path(target).is_dir()

    def test_ensure_dir_is_idempotent(self, tmp_path):
        from tests.data_downloader import ensure_dir

        target = str(tmp_path / "nested")
        ensure_dir(target)
        ensure_dir(target)  # should not raise

    def test_test_data_dir_is_inside_tests_folder(self):
        from tests.data_downloader import TEST_DATA_DIR

        assert "tests" in TEST_DATA_DIR
        assert "test_data" in TEST_DATA_DIR

    def test_download_functions_skip_if_already_downloaded(self, tmp_path, capsys):
        from tests.data_downloader import download_fleurs

        # Pre-create the ground truth file so the function skips
        out_dir = tmp_path / "fleurs_marathi"
        out_dir.mkdir(parents=True)
        (out_dir / "ground_truth.tsv").write_text("file_id\ttranscript\n")

        with patch("tests.data_downloader.TEST_DATA_DIR", str(tmp_path)):
            download_fleurs("marathi", "mr", n_samples=1)

        captured = capsys.readouterr()
        assert "skip" in captured.out
