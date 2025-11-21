"""Semi-implicit linear solver placeholder."""
from __future__ import annotations

import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.state as state


def semi_implicit_step(mstate: state.ModelState, tendencies, alpha: float = cfg.alpha):
    """Apply simple semi-implicit correction to divergence, temperature, lnps."""
    zeta_t, div_t, T_t, lnps_t = tendencies
    div_new = mstate.div + cfg.dt * ((1 - alpha) * div_t)
    T_new = mstate.T + cfg.dt * ((1 - alpha) * T_t)
    lnps_new = mstate.lnps + cfg.dt * ((1 - alpha) * lnps_t)
    return state.ModelState(mstate.zeta + cfg.dt * zeta_t, div_new, T_new, lnps_new)
