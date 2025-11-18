"""Semi-implicit off-centering of fast linear gravity-wave terms."""
from __future__ import annotations
import jax
from .. import config


@jax.jit
def apply(div_t, T_t, lnps_t, alpha: float | None = None):
    """Apply a simple off-centered damping to mimic SI correction."""
    if alpha is None:
        alpha = config.numerics.semi_implicit_alpha
    fac = 1.0 / (1.0 + alpha)
    return div_t * fac, T_t * fac, lnps_t * fac
