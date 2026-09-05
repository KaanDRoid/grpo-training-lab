"""Triton learning kernels and their PyTorch-facing wrappers."""

import os
from pathlib import Path


# Keep generated compiler artifacts inside the project by default. This is portable across
# restricted/sandboxed Windows accounts and can still be overridden by the caller.
os.environ.setdefault(
    "TRITON_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".triton_cache")
)
