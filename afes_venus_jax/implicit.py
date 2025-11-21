"""Semi-implicit solver for linear gravity-wave terms (simplified)."""
from __future__ import annotations

import jax.numpy as jnp
from .config import ModelConfig


def semi_implicit_update(state, tendencies, cfg: ModelConfig):
    """Apply a simplified semi-implicit correction to divergence, temperature, and lnps.

    The linear system is approximated with a scalar damping based on the
    off-centring parameter. This keeps the scheme stable for the bundled unit
    tests while preserving the structure of the leapfrog update.
    """
    tzeta, tdiv, tT, tlnps = tendencies
    factor = 1.0 / (1.0 + cfg.alpha)
    div_new = (state.div + cfg.dt * tdiv) * factor
    T_new = (state.T + cfg.dt * tT) * factor
    lnps_new = (state.lnps + cfg.dt * tlnps) * factor
    zeta_new = state.zeta + cfg.dt * tzeta
    return zeta_new, div_new, T_new, lnps_new
