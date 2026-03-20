"""
VaniLipi ASR Evaluation
========================
Benchmarks VaniLipi's ASR output against ground truth transcriptions.
Computes Word Error Rate (WER) and Character Error Rate (CER) per dataset.

Usage:
    pip install jiwer
    python tests/evaluate_asr.py

Requires:
    - tests/test_data/ folder populated by data_downloader.py
    - VaniLipi backend importable (run from project root with venv active)
"""

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, TypedDict, cast


class SampleResult(TypedDict):
    file_id: str
    ground_truth: str
    prediction: str
    wer: float
    cer: float
    time_seconds: float


class EvalResult(TypedDict):
    dataset: str
    language: str
    num_samples: int
    overall_wer: float
    overall_cer: float
    total_time_seconds: float
    avg_time_per_sample: float
    per_sample: list[SampleResult]

try:
    from jiwer import wer, cer  # type: ignore[import]
except ImportError:
    print("Install jiwer first: pip install jiwer")
    sys.exit(1)

_backend_transcribe: Any = None

try:
    from backend.services.asr import transcribe as _backend_transcribe  # type: ignore[import]
except ImportError:
    pass

TEST_DATA_DIR = str(Path(__file__).parent / "test_data")
RESULTS_FILE = str(Path(__file__).parent / "test_data" / "evaluation_results.json")


def load_ground_truth(dataset_dir: str) -> dict[str, str]:
    """Load ground truth TSV file. Returns {file_id: transcript}."""
    gt_path = Path(dataset_dir) / "ground_truth.tsv"
    if not gt_path.exists():
        return {}

    samples: dict[str, str] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            file_id = row["file_id"]
            transcript = row["transcript"].strip()
            samples[file_id] = transcript
    return samples


def transcribe_with_vanilipi(audio_path: str, language: str = "mr") -> str:
    """Transcribe a single audio file using VaniLipi's ASR service."""
    if _backend_transcribe is None:
        print("[error] Cannot import backend.services.asr")
        print("[info] Make sure you're running from the vanilipi project root")
        print("[info] and the venv is activated: source venv/bin/activate")
        sys.exit(1)
    raw: Any = _backend_transcribe(audio_path, language=language)
    if isinstance(raw, dict):
        result = cast(dict[str, Any], raw)
        if "segments" in result:
            return " ".join(str(seg["text"]) for seg in result["segments"])
        if "text" in result:
            return str(result["text"])
    if isinstance(raw, str):
        return raw
    return str(cast(object, raw))


def normalize_text(text: str) -> str:
    """
    Basic normalization for WER comparison.

    Whisper and ground truth may differ in:
    - Punctuation (Whisper adds it, some ground truths don't have it)
    - Danda (।) vs period (.)
    - Extra whitespace
    """
    text = re.sub(r'[।,.!?;:\-"\'\(\)]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text


def evaluate_dataset(dataset_name: str, language_code: str = "mr") -> EvalResult | None:
    """Evaluate VaniLipi on a single test dataset. Returns metrics dict or None."""
    dataset_dir = Path(TEST_DATA_DIR) / dataset_name
    audio_dir = dataset_dir / "audio"

    if not dataset_dir.exists():
        print(f"  [skip] {dataset_name} not found")
        return None

    ground_truth = load_ground_truth(str(dataset_dir))
    if not ground_truth:
        print(f"  [skip] {dataset_name} has no ground truth")
        return None

    print(f"  Evaluating {len(ground_truth)} samples...")

    references: list[str] = []
    predictions: list[str] = []
    per_sample: list[SampleResult] = []
    total_time = 0.0

    for file_id, gt_text in ground_truth.items():
        # Find audio file
        audio_path = None
        for ext in (".wav", ".mp3", ".flac"):
            candidate = audio_dir / f"{file_id}{ext}"
            if candidate.exists():
                audio_path = str(candidate)
                break

        if audio_path is None:
            print(f"    [skip] No audio for {file_id}")
            continue

        t0 = time.time()
        pred_text = transcribe_with_vanilipi(audio_path, language=language_code)
        elapsed = time.time() - t0
        total_time += elapsed

        gt_norm = normalize_text(gt_text)
        pred_norm = normalize_text(pred_text)

        if gt_norm and pred_norm:
            sample_wer: float = float(wer(gt_norm, pred_norm))  # type: ignore[arg-type]
            sample_cer: float = float(cer(gt_norm, pred_norm))  # type: ignore[arg-type]
        else:
            sample_wer = 1.0
            sample_cer = 1.0

        references.append(gt_norm)
        predictions.append(pred_norm)

        per_sample.append(
            {
                "file_id": file_id,
                "ground_truth": gt_text,
                "prediction": pred_text,
                "wer": round(sample_wer, 4),
                "cer": round(sample_cer, 4),
                "time_seconds": round(elapsed, 2),
            }
        )

        status = "OK" if sample_wer < 0.3 else "WARN" if sample_wer < 0.6 else "BAD"
        print(
            f"    [{status}] {file_id}: WER={sample_wer:.2%} CER={sample_cer:.2%} ({elapsed:.1f}s)"
        )

    if not references:
        return None

    overall_wer: float = float(wer(references, predictions))  # type: ignore[arg-type]
    overall_cer: float = float(cer(references, predictions))  # type: ignore[arg-type]

    return {
        "dataset": dataset_name,
        "language": language_code,
        "num_samples": len(references),
        "overall_wer": round(overall_wer, 4),
        "overall_cer": round(overall_cer, 4),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_sample": round(total_time / len(references), 2),
        "per_sample": per_sample,
    }


def main() -> None:
    print("=" * 60)
    print("VaniLipi ASR Evaluation")
    print("=" * 60)
    print()

    datasets_to_eval = [
        ("openslr64_marathi", "mr"),
        ("openslr103_marathi", "mr"),
        ("common_voice_marathi", "mr"),
        ("common_voice_hindi", "hi"),
        ("fleurs_marathi", "mr"),
        ("fleurs_hindi", "hi"),
    ]

    all_results: list[EvalResult] = []

    for dataset_name, lang_code in datasets_to_eval:
        print(f"\n[{dataset_name}] (language: {lang_code})")
        result = evaluate_dataset(dataset_name, lang_code)
        if result:
            all_results.append(result)
            print(f"  Overall WER: {result['overall_wer']:.2%}")
            print(f"  Overall CER: {result['overall_cer']:.2%}")
            print(f"  Avg time/sample: {result['avg_time_per_sample']:.1f}s")

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<30} {'Lang':<6} {'WER':<10} {'CER':<10} {'Samples'}")
    print("-" * 70)
    for r in all_results:
        print(
            f"{str(r['dataset']):<30} {str(r['language']):<6} "
            f"{float(r['overall_wer']):.2%}    {float(r['overall_cer']):.2%}    {r['num_samples']}"
        )

    print(f"\nDetailed results saved to: {RESULTS_FILE}")

    marathi_results: list[EvalResult] = [r for r in all_results if r["language"] == "mr"]
    if marathi_results:
        avg_marathi_wer = sum(r["overall_wer"] for r in marathi_results) / len(marathi_results)
        print(f"\nMarathi average WER: {avg_marathi_wer:.2%}")
        if avg_marathi_wer < 0.20:
            print("Quality: EXCELLENT - Beats most paid apps for Marathi")
        elif avg_marathi_wer < 0.35:
            print("Quality: GOOD - Competitive with paid apps")
        elif avg_marathi_wer < 0.50:
            print("Quality: FAIR - Usable with manual editing")
        else:
            print("Quality: NEEDS IMPROVEMENT - Consider model tuning")


if __name__ == "__main__":
    main()
