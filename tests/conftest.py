"""
Pre-mock heavy ML dependencies so modules can be imported in the test suite
without requiring the actual packages (torch, mlx, mlx_whisper) to be installed.
These mocks are installed before any test module imports the backend code.
"""
import sys
from unittest.mock import MagicMock

_HEAVY_DEPS = [
    "torch",
    "torch.backends",
    "torch.backends.mps",
    "mlx_whisper",
    "transformers",
    "sentencepiece",
    "IndicTransToolkit",
    "IndicTransToolkit.processor",
    "soundfile",
    "librosa",
    # datasets imports torch via importlib.util.find_spec, which fails on MagicMock
    "datasets",
    "datasets.config",
]

for _mod in _HEAVY_DEPS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make torch.backends.mps.is_available() return False by default (no GPU in tests)
import torch  # noqa: E402 — already mocked above
torch.backends.mps.is_available.return_value = False
torch.float16 = "float16"

# Make torch.no_grad() work as a context manager
_no_grad_ctx = MagicMock()
_no_grad_ctx.__enter__ = MagicMock(return_value=None)
_no_grad_ctx.__exit__ = MagicMock(return_value=False)
torch.no_grad.return_value = _no_grad_ctx
