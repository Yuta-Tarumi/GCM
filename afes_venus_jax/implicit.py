"""Semi-implicit solver for linear gravity-wave terms."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from . import config


@jax.jit
def implicit_correction(div_t, T_t, lnps_t, alpha: float = config.alpha):
    """Apply a simple off-centered damping to mimic SI correction."""
    fac = 1.0 / (1.0 + alpha)
    return div_t * fac, T_t * fac, lnps_t * fac
