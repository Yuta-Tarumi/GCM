"""Pytest configuration for lightweight test settings.

The default T42L60 configuration is expensive for constrained CI runners.
These environment overrides reduce the grid and truncation so unit tests
complete quickly while preserving the production defaults when the
environment variables are unset.
"""

import os
import sys
from pathlib import Path


# Ensure the repository root is importable without an editable install.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


os.environ.setdefault("AFES_VENUS_JAX_LMAX", "10")
os.environ.setdefault("AFES_VENUS_JAX_NLAT", "32")
os.environ.setdefault("AFES_VENUS_JAX_NLON", "64")
os.environ.setdefault("AFES_VENUS_JAX_L", "10")
os.environ.setdefault("AFES_VENUS_JAX_FAST_TESTS", "1")
# Use higher precision in tests to avoid underflow in scale-sensitive operators
# such as hyperdiffusion when the truncation (Lmax) is small.
os.environ.setdefault("AFES_VENUS_JAX_ENABLE_X64", "True")

# Avoid JIT compilation overhead in the test environment to shorten wall time.
os.environ.setdefault("JAX_DISABLE_JIT", "1")
