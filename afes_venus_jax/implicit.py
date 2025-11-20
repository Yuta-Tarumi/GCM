"""Semi-implicit linear solver for fast gravity-wave terms.

This is a simplified per-wavenumber stabiliser that damps divergence and
surface-pressure oscillations using a diagonal approximation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import Config


def apply_semi_implicit(div_lm: jnp.ndarray, T_lm: jnp.ndarray, lnps_lm: jnp.ndarray, cfg: Config):
    """Apply a basic semi-implicit filter.

    The operator scales divergence and pressure tendencies to mimic an off-
    centered gravity-wave solve. It is intentionally simple but keeps the
    time stepping stable for the test problems.
    """
    factor = 1.0 / (1.0 + cfg.alpha)
    # Leave temperature untouched to preserve background stratification while
    # damping fast gravity-wave components of divergence and surface pressure.
    return div_lm * factor, T_lm, lnps_lm * factor


def robert_asselin_filter(new: jnp.ndarray, old: jnp.ndarray, ra: float):
    return new + ra * (old - new)
