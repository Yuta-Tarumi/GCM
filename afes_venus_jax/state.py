"""Model prognostic state container."""
from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp

from .config import ModelConfig
from .spharm import spectral_shape


@dataclasses.dataclass
class ModelState:
    """Spectral model state.

    Attributes
    ----------
    zeta : complex array, shape (L, Lmax+1, 2Lmax+1)
    div : complex array, shape (L, Lmax+1, 2Lmax+1)
    T : complex array, shape (L, Lmax+1, 2Lmax+1)
    lnps : complex array, shape (Lmax+1, 2Lmax+1)
    """

    zeta: jnp.ndarray
    div: jnp.ndarray
    T: jnp.ndarray
    lnps: jnp.ndarray

    def copy(self):
        return ModelState(self.zeta.copy(), self.div.copy(), self.T.copy(), self.lnps.copy())


jax.tree_util.register_pytree_node(
    ModelState,
    lambda s: ((s.zeta, s.div, s.T, s.lnps), None),
    lambda _, xs: ModelState(*xs),
)


def zeros_state(cfg: ModelConfig) -> ModelState:
    shp = (cfg.L, *spectral_shape(cfg))
    spec_zero = jnp.zeros(shp, dtype=jnp.complex128)
    lnps = jnp.zeros(spectral_shape(cfg), dtype=jnp.complex128)
    return ModelState(spec_zero, spec_zero, spec_zero, lnps)
