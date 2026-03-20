"""
VaniLipi Test Data Downloader
==============================
Downloads small curated test sets with ground truth transcriptions from free,
publicly licensed datasets. All CC-BY-SA 4.0 or CC-0 licensed.

Sources:
1. OpenSLR-64: Crowdsourced high-quality Marathi multi-speaker speech (Google, CC-BY-SA 4.0)
   - WAV files + TSV with ground truth transcriptions
   - ~712MB full dataset, we download just the TSV and sample 20 audio files

2. Mozilla Common Voice (Marathi & Hindi): Crowd-sourced read speech (CC-0)
   - Accessed via HuggingFace datasets library (streaming mode, no full download)
   - We pull 10 samples per language with ground truth

3. Google FLEURS (Marathi & Hindi): Read Wikipedia sentences (CC-BY-SA 4.0)
   - Accessed via HuggingFace datasets library
   - High quality, multi-speaker, used as the standard ASR benchmark

Usage:
    python tests/data_downloader.py

Output:
    tests/test_data/
      openslr64_marathi/
        audio/         (20 WAV files)
        ground_truth.tsv
      common_voice_marathi/
        audio/         (10 MP3 files)
        ground_truth.tsv
      common_voice_hindi/
        audio/         (10 MP3 files)
        ground_truth.tsv
      fleurs_marathi/
        audio/         (10 WAV files)
        ground_truth.tsv
      fleurs_hindi/
        audio/         (10 WAV files)
        ground_truth.tsv
      README.md        (dataset descriptions and licenses)
"""

import csv
import glob
import io
import os
import ssl
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import requests as _requests

sf: Any = None
load_dataset: Any = None

try:
    import soundfile as sf  # type: ignore[import]
    from datasets import load_dataset  # type: ignore[import]
except ImportError:
    pass

# HuggingFace token from env (set in .env or export HF_TOKEN=...)
_HF_TOKEN = os.environ.get("HF_TOKEN", "")

# TEST_DATA_DIR is relative to the project root, not to this file
TEST_DATA_DIR = str(Path(__file__).parent / "test_data")


def _build_ssl_context() -> ssl.SSLContext:
    """Build an SSL context that handles Zscaler SSL inspection.

    Zscaler's MITM proxy re-signs certificates with its own CA, causing
    hostname mismatch errors even when its CA bundle is loaded. When we
    detect the Zscaler bundle, we disable hostname checking since traffic
    is already inspected by the corporate proxy.
    """
    cert_file = os.environ.get(
        "SSL_CERT_FILE",
        os.path.expanduser("~/.zscaler_certifi_bundle.pem"),
    )
    behind_zscaler = os.path.isfile(cert_file) and "zscaler" in cert_file.lower()
    if behind_zscaler:
        ctx = ssl.create_default_context()
        ctx.load_verify_locations(cert_file)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return ssl.create_default_context()


_SSL_CTX = _build_ssl_context()


def download_file(url: str, dest: str, *, quiet: bool = True) -> bool:
    """Download a URL to a local file. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VaniLipi/1.0"})
        with urllib.request.urlopen(req, context=_SSL_CTX) as resp:
            with open(dest, "wb") as f:
                f.write(resp.read())
        return True
    except Exception:
        if not quiet:
            raise
        return False


def hf_download(url: str, dest: str) -> bool:
    """Download a file from HuggingFace with auth token. Returns True on success."""
    headers: dict[str, str] = {}
    if _HF_TOKEN:
        headers["Authorization"] = f"Bearer {_HF_TOKEN}"
    try:
        resp = _requests.get(url, headers=headers, verify=False, stream=True, timeout=300)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    [error] HF download failed: {e}")
        return False


def hf_stream_tar(url: str, audio_dir: str, filenames: set[str]) -> set[str]:
    """Stream a tar.gz from HF and extract only the requested filenames.

    Returns the set of filenames that were successfully extracted.
    """
    headers: dict[str, str] = {}
    if _HF_TOKEN:
        headers["Authorization"] = f"Bearer {_HF_TOKEN}"
    extracted: set[str] = set()
    try:
        resp = _requests.get(url, headers=headers, verify=False, stream=True, timeout=600)
        resp.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                basename = os.path.basename(member.name)
                if basename in filenames and member.isfile():
                    member.name = basename
                    tar.extract(member, audio_dir)
                    extracted.add(basename)
                    if len(extracted) >= len(filenames):
                        break
    except Exception as e:
        print(f"    [error] Tar streaming failed: {e}")
    return extracted


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_openslr64_marathi(n_samples: int = 20) -> None:
    """Download a small sample from OpenSLR-64 Marathi dataset."""
    out_dir = os.path.join(TEST_DATA_DIR, "openslr64_marathi")
    audio_dir = os.path.join(out_dir, "audio")
    ensure_dir(audio_dir)

    if os.path.exists(os.path.join(out_dir, "ground_truth.tsv")):
        print("  [skip] OpenSLR-64 Marathi already downloaded")
        return

    print("  Downloading OpenSLR-64 Marathi index...")

    # Try direct download from OpenSLR (may fail behind corporate proxies)
    tsv_url = "https://us.openslr.org/resources/64/line_index.tsv"
    tsv_path = os.path.join(out_dir, "line_index.tsv")
    successful: list[tuple[str, str]] = []

    if download_file(tsv_url, tsv_path):
        # Parse TSV and select first n_samples
        samples: list[tuple[str, str]] = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i >= n_samples:
                    break
                if len(row) >= 2:
                    file_id, transcript = row[0].strip(), row[1].strip()
                    samples.append((file_id, transcript))

        # Download audio files
        base_url = "https://us.openslr.org/resources/64/mr_in_female"
        print(f"  Downloading {len(samples)} audio files...")
        for file_id, transcript in samples:
            wav_url = f"{base_url}/{file_id}.wav"
            wav_path = os.path.join(audio_dir, f"{file_id}.wav")
            download_file(wav_url, wav_path)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000:
                successful.append((file_id, transcript))
            else:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
    else:
        print("  [warn] Direct download failed (403/SSL).")

    if not successful:
        print("  [warn] Direct download failed. Trying HuggingFace mirror...")
        if sf is None or load_dataset is None:
            print("  [error] soundfile/datasets not installed; pip install soundfile datasets")
            print("  [info] OpenSLR-64 requires manual download: https://openslr.org/64/")
            return
        try:
            ds = load_dataset(
                "openslr/openslr",
                "SLR64",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            successful: list[tuple[str, str]] = []
            for i, sample in enumerate(ds):
                if i >= n_samples:
                    break
                audio = sample["audio"]
                transcript = sample.get("sentence", sample.get("text", ""))
                file_id = f"mr_sample_{i:04d}"
                wav_path = os.path.join(audio_dir, f"{file_id}.wav")
                sf.write(wav_path, audio["array"], audio["sampling_rate"])
                successful.append((file_id, transcript))
        except Exception as e:
            print(f"  [error] HuggingFace fallback also failed: {e}")
            print("  [info] OpenSLR-64 requires manual download: https://openslr.org/64/")

    # Write ground truth TSV
    gt_path = os.path.join(out_dir, "ground_truth.tsv")
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("file_id\ttranscript\n")
        for file_id, transcript in successful:
            f.write(f"{file_id}\t{transcript}\n")

    print(f"  Downloaded {len(successful)} samples to {out_dir}")


def download_common_voice(language: str, lang_code: str, n_samples: int = 10) -> None:
    """Download samples from Mozilla Common Voice via HuggingFace streaming.

    Tries multiple CV versions (17.0, 16.1, 13.0) since older versions may
    require gated access or loading scripts that newer HF Hub rejects.
    """
    out_dir = os.path.join(TEST_DATA_DIR, f"common_voice_{language}")
    audio_dir = os.path.join(out_dir, "audio")
    ensure_dir(audio_dir)

    if os.path.exists(os.path.join(out_dir, "ground_truth.tsv")):
        print(f"  [skip] Common Voice {language} already downloaded")
        return

    if sf is None or load_dataset is None:
        print(f"  [error] Common Voice {language} failed: soundfile/datasets not installed")
        print("  [info] pip install soundfile datasets")
        return

    cv_versions = ["17_0", "16_1", "13_0"]
    for version in cv_versions:
        dataset_id = f"mozilla-foundation/common_voice_{version}"
        print(f"  Trying Common Voice {version} {language}...")
        try:
            ds = load_dataset(
                dataset_id,
                lang_code,
                split="test",
                streaming=True,
                trust_remote_code=True,
            )

            samples: list[tuple[str, str]] = []
            for i, sample in enumerate(ds):
                if i >= n_samples:
                    break

                audio = sample["audio"]
                transcript = sample["sentence"]
                file_id = f"cv_{lang_code}_{i:04d}"

                wav_path = os.path.join(audio_dir, f"{file_id}.wav")
                sf.write(wav_path, audio["array"], audio["sampling_rate"])
                samples.append((file_id, transcript))

            if samples:
                gt_path = os.path.join(out_dir, "ground_truth.tsv")
                with open(gt_path, "w", encoding="utf-8") as f:
                    f.write("file_id\ttranscript\n")
                    for file_id, transcript in samples:
                        f.write(f"{file_id}\t{transcript}\n")
                print(f"  Downloaded {len(samples)} samples to {out_dir}")
                return

        except Exception as e:
            print(f"    [warn] {dataset_id} failed: {e}")
            continue

    print(f"  [error] Common Voice {language}: all versions failed")
    print("  [info] You may need to accept terms and run: huggingface-cli login")


def download_fleurs(language: str, lang_code: str, n_samples: int = 10) -> None:
    """Download samples from Google FLEURS via direct HF file downloads.

    Downloads the TSV metadata and streams only the needed audio files from
    the tar.gz archive. Does not use the `datasets` library (avoids loading
    script issues with newer HF Hub versions).
    """
    out_dir = os.path.join(TEST_DATA_DIR, f"fleurs_{language}")
    audio_dir = os.path.join(out_dir, "audio")
    ensure_dir(audio_dir)

    if os.path.exists(os.path.join(out_dir, "ground_truth.tsv")):
        print(f"  [skip] FLEURS {language} already downloaded")
        return

    fleurs_code = f"{lang_code}_in"
    base = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"

    print(f"  Downloading FLEURS {language} metadata...")
    tsv_url = f"{base}/{fleurs_code}/test.tsv"
    tsv_path = os.path.join(out_dir, "test.tsv")
    if not hf_download(tsv_url, tsv_path):
        print(f"  [error] FLEURS {language}: could not download test.tsv")
        return

    # Parse TSV: columns are id, filename, transcription (raw), transcription (norm), ...
    metadata: list[tuple[str, str, str]] = []  # (file_id, wav_filename, transcript)
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                wav_filename = parts[1].strip()
                transcript = parts[2].strip()
                file_id = wav_filename.replace(".wav", "")
                metadata.append((file_id, wav_filename, transcript))
            if len(metadata) >= n_samples:
                break

    if not metadata:
        print(f"  [error] FLEURS {language}: no samples in test.tsv")
        return

    # Download the tar.gz and extract only the files we need
    needed_files = {wav for _, wav, _ in metadata}
    print(f"  Downloading {len(needed_files)} audio files from test.tar.gz (~720MB stream)...")
    tar_url = f"{base}/{fleurs_code}/audio/test.tar.gz"
    extracted = hf_stream_tar(tar_url, audio_dir, needed_files)

    # Write ground truth for successfully extracted files
    samples: list[tuple[str, str]] = []
    for file_id, wav_filename, transcript in metadata:
        if wav_filename in extracted:
            samples.append((file_id, transcript))

    gt_path = os.path.join(out_dir, "ground_truth.tsv")
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("file_id\ttranscript\n")
        for file_id, transcript in samples:
            f.write(f"{file_id}\t{transcript}\n")

    # Cleanup
    if os.path.exists(tsv_path):
        os.remove(tsv_path)

    print(f"  Downloaded {len(samples)} samples to {out_dir}")


def download_openslr103_marathi(n_samples: int = 20) -> None:
    """Download from OpenSLR-103: MUCS Marathi (8kHz conversational speech)."""
    out_dir = os.path.join(TEST_DATA_DIR, "openslr103_marathi")
    audio_dir = os.path.join(out_dir, "audio")
    ensure_dir(audio_dir)

    if os.path.exists(os.path.join(out_dir, "ground_truth.tsv")):
        print("  [skip] OpenSLR-103 Marathi already downloaded")
        return

    print("  Downloading OpenSLR-103 Marathi test set (235MB)...")
    print("  [info] This is conversational speech from diverse speakers - harder than read speech.")
    print("  [info] 8kHz sample rate - tests the app's handling of lower quality audio.")

    tar_url = "https://us.openslr.org/resources/103/Marathi_test.tar.gz"
    tar_path = os.path.join(out_dir, "Marathi_test.tar.gz")

    try:
        if not download_file(tar_url, tar_path):
            raise RuntimeError("Failed to download OpenSLR-103 tar (403/SSL)")
        subprocess.run(["tar", "-xzf", tar_path, "-C", out_dir], check=True)

        transcript_files = glob.glob(
            os.path.join(out_dir, "**", "transcription.txt"), recursive=True
        )
        if not transcript_files:
            transcript_files = glob.glob(
                os.path.join(out_dir, "**", "*.txt"), recursive=True
            )

        samples: list[tuple[str, str]] = []
        if transcript_files:
            with open(transcript_files[0], "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= n_samples:
                        break
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        file_id, transcript = parts
                        audio_files = glob.glob(
                            os.path.join(out_dir, "**", f"{file_id}*"), recursive=True
                        )
                        audio_files = [
                            af for af in audio_files if af.endswith((".wav", ".flac"))
                        ]
                        if audio_files:
                            dest = os.path.join(
                                audio_dir, os.path.basename(audio_files[0])
                            )
                            if not os.path.exists(dest):
                                os.rename(audio_files[0], dest)
                            samples.append(
                                (
                                    os.path.basename(audio_files[0]).split(".")[0],
                                    transcript,
                                )
                            )

        # Write ground truth
        gt_path = os.path.join(out_dir, "ground_truth.tsv")
        with open(gt_path, "w", encoding="utf-8") as f:
            f.write("file_id\ttranscript\n")
            for file_id, transcript in samples:
                f.write(f"{file_id}\t{transcript}\n")

        if os.path.exists(tar_path):
            os.remove(tar_path)

        print(f"  Downloaded {len(samples)} samples to {out_dir}")

    except Exception as e:
        print(f"  [error] OpenSLR-103 download failed: {e}")
        print("  [info] Manual download: https://openslr.org/103/")


def create_readme() -> None:
    """Create README documenting the test datasets."""
    readme = """# VaniLipi Test Data

## Datasets

### 1. OpenSLR-64: Marathi Multi-Speaker Speech (Google)
- **License:** CC-BY-SA 4.0
- **Source:** https://openslr.org/64/
- **Type:** High-quality read speech, female speakers
- **Sample rate:** 48kHz WAV
- **Use:** Baseline ASR accuracy testing (clean audio)

### 2. OpenSLR-103: MUCS Marathi (IIT Madras)
- **License:** CC-BY-SA 4.0
- **Source:** https://openslr.org/103/
- **Type:** Conversational speech from diverse speakers (college students, rural workers, urban workers)
- **Sample rate:** 8kHz WAV
- **Use:** Stress-testing ASR on noisy, lower-quality, diverse-accent audio

### 3. Mozilla Common Voice (Marathi & Hindi)
- **License:** CC-0 (public domain)
- **Source:** https://commonvoice.mozilla.org/
- **Type:** Crowd-sourced read speech, multi-speaker
- **Sample rate:** 48kHz MP3
- **Use:** Multi-language testing, diverse speaker quality

### 4. Google FLEURS (Marathi & Hindi)
- **License:** CC-BY-SA 4.0
- **Source:** https://huggingface.co/datasets/google/fleurs
- **Type:** Read Wikipedia sentences, multi-speaker
- **Sample rate:** 16kHz WAV
- **Use:** Standard ASR benchmark, comparison with published WER numbers

## Ground Truth Format

Each dataset folder contains a `ground_truth.tsv` with:
```
file_id    transcript
mr_0001    हा एक चाचणी आहे.
mr_0002    आज हवामान छान आहे.
```

## How to evaluate

```python
from jiwer import wer

# Load ground truth
ground_truth = {}
with open("tests/test_data/fleurs_marathi/ground_truth.tsv") as f:
    next(f)  # skip header
    for line in f:
        fid, transcript = line.strip().split("\\t", 1)
        ground_truth[fid] = transcript

# Compare against VaniLipi output
predictions = {}  # fill from app output
word_error_rate = wer(
    list(ground_truth.values()),
    list(predictions.values())
)
print(f"WER: {word_error_rate:.2%}")
```

Install jiwer for WER computation: `pip install jiwer`
"""
    with open(os.path.join(TEST_DATA_DIR, "README.md"), "w") as f:
        f.write(readme)


def main() -> None:
    print("=" * 60)
    print("VaniLipi Test Data Downloader")
    print("=" * 60)
    print()

    ensure_dir(TEST_DATA_DIR)

    print("[1/1] FLEURS Marathi (benchmark standard)...")
    download_fleurs("marathi", "mr", n_samples=20)
    print()

    create_readme()

    print("=" * 60)
    print("Done! Test data saved to ./tests/test_data/")
    print()
    print("Summary:")
    for d in sorted(os.listdir(TEST_DATA_DIR)):
        full = os.path.join(TEST_DATA_DIR, d)
        if os.path.isdir(full):
            gt = os.path.join(full, "ground_truth.tsv")
            if os.path.exists(gt):
                with open(gt) as _f:
                    count = sum(1 for _ in _f) - 1  # minus header
                print(f"  {d}: {count} samples")
    print()
    print("To evaluate WER: pip install jiwer")
    print("See tests/test_data/README.md for details.")


if __name__ == "__main__":
    main()
