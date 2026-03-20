"""
VaniLipi configuration: paths, model registry, language maps, and defaults.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# App directories
# ---------------------------------------------------------------------------
APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "VaniLipi"
PROJECTS_DIR = APP_SUPPORT_DIR / "projects"
MODELS_DIR = APP_SUPPORT_DIR / "models"
TEMP_DIR = APP_SUPPORT_DIR / "tmp"

for _d in (APP_SUPPORT_DIR, PROJECTS_DIR, MODELS_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model paths (local, bundled in repo under models/)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent

# REPO_MODELS_DIR can be overridden by the launcher in the macOS app bundle,
# where models live at Contents/Resources/models/ rather than next to the code.
import os as _os
_MODELS_ROOT = Path(_os.environ.get("REPO_MODELS_DIR", str(_REPO_ROOT / "models")))

# ASR — mlx-whisper accepts a local directory path in place of a HF repo ID
ASR_MODEL = str(_MODELS_ROOT / "asr" / "whisper-large-v3")

# Translation — local MLX weights + tokenizer
TRANSLATION_MODEL_DIR = _MODELS_ROOT / "translation" / "indictrans2-1b"

# ---------------------------------------------------------------------------
# Supported languages
# Whisper code -> display name, IndicTrans2 code, quality tier
# ---------------------------------------------------------------------------
LANGUAGE_MAP = {
    "mr": {
        "name": "Marathi",
        "script": "Devanagari",
        "indictrans_code": "mar_Deva",
        "quality": "good",
        "recommended": True,
    },
    "hi": {
        "name": "Hindi",
        "script": "Devanagari",
        "indictrans_code": "hin_Deva",
        "quality": "very_good",
        "recommended": True,
    },
    "bn": {
        "name": "Bengali",
        "script": "Bengali",
        "indictrans_code": "ben_Beng",
        "quality": "good",
        "recommended": True,
    },
    "ta": {
        "name": "Tamil",
        "script": "Tamil",
        "indictrans_code": "tam_Taml",
        "quality": "good",
        "recommended": True,
    },
    "te": {
        "name": "Telugu",
        "script": "Telugu",
        "indictrans_code": "tel_Telu",
        "quality": "good",
        "recommended": True,
    },
    "gu": {
        "name": "Gujarati",
        "script": "Gujarati",
        "indictrans_code": "guj_Gujr",
        "quality": "fair",
        "recommended": False,
    },
    "kn": {
        "name": "Kannada",
        "script": "Kannada",
        "indictrans_code": "kan_Knda",
        "quality": "fair",
        "recommended": False,
    },
    "ml": {
        "name": "Malayalam",
        "script": "Malayalam",
        "indictrans_code": "mal_Mlym",
        "quality": "fair",
        "recommended": False,
    },
    "pa": {
        "name": "Punjabi",
        "script": "Gurmukhi",
        "indictrans_code": "pan_Guru",
        "quality": "fair",
        "recommended": False,
    },
    "ur": {
        "name": "Urdu",
        "script": "Perso-Arabic",
        "indictrans_code": "urd_Arab",
        "quality": "good",
        "recommended": False,
    },
    "ne": {
        "name": "Nepali",
        "script": "Devanagari",
        "indictrans_code": "npi_Deva",
        "quality": "fair",
        "recommended": False,
    },
    "sd": {
        "name": "Sindhi",
        "script": "Devanagari",
        "indictrans_code": "snd_Deva",
        "quality": "poor",
        "recommended": False,
    },
    "as": {
        "name": "Assamese",
        "script": "Bengali",
        "indictrans_code": "asm_Beng",
        "quality": "poor",
        "recommended": False,
    },
    "sa": {
        "name": "Sanskrit",
        "script": "Devanagari",
        "indictrans_code": "san_Deva",
        "quality": "poor",
        "recommended": False,
    },
}

# Whisper code -> IndicTrans2 src_lang (convenience shortcut)
WHISPER_TO_INDICTRANS: dict[str, str] = {
    code: info["indictrans_code"] for code, info in LANGUAGE_MAP.items()
}

SUPPORTED_WHISPER_CODES: frozenset[str] = frozenset(LANGUAGE_MAP.keys())

# ---------------------------------------------------------------------------
# Audio constraints
# ---------------------------------------------------------------------------
MAX_AUDIO_DURATION_SECONDS = 3 * 60 * 60  # 3 hours
SUPPORTED_AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
)
SUPPORTED_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mkv", ".webm", ".mov", ".avi", ".wmv"}
)
SUPPORTED_UPLOAD_EXTENSIONS = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

# ---------------------------------------------------------------------------
# Translation batching
# ---------------------------------------------------------------------------
TRANSLATION_BATCH_SIZE = 12

# ---------------------------------------------------------------------------
# Server defaults
# ---------------------------------------------------------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
PORT_SEARCH_RANGE = 10  # try 7860..7869

# ---------------------------------------------------------------------------
# Default app settings
# ---------------------------------------------------------------------------
DEFAULT_LANGUAGE = "auto"
DEFAULT_ASR_MODEL = ASR_MODEL
DEFAULT_TRANSLATION_MODEL = TRANSLATION_MODEL_DIR
DEFAULT_THEME = "system"
DEFAULT_AUTO_TRANSLATE = True
