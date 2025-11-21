"""Model state container."""
from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp

import afes_venus_jax.config as cfg


@dataclass
class ModelState:
    """Spectral prognostic fields.

    Attributes
    ----------
    zeta : complex array (L, lmax+1, lmax+1)
        Vorticity coefficients.
    div : complex array (L, lmax+1, lmax+1)
        Divergence coefficients.
    T : complex array (L, lmax+1, lmax+1)
        Temperature coefficients.
    lnps : complex array (lmax+1, lmax+1)
        Log surface pressure.
    """

    zeta: jnp.ndarray
    div: jnp.ndarray
    T: jnp.ndarray
    lnps: jnp.ndarray


jax.tree_util.register_pytree_node(
    ModelState,
    lambda s: ((s.zeta, s.div, s.T, s.lnps), None),
    lambda _, data: ModelState(*data),
)


def zeros_state(L: int = cfg.L, lmax: int = cfg.Lmax):
    shape = (L, lmax + 1, lmax + 1)
    z = jnp.zeros(shape, dtype=jnp.complex128)
    return ModelState(z, z, z, jnp.zeros((lmax + 1, lmax + 1), dtype=jnp.complex128))
