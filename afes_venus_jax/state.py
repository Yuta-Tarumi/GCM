"""Model prognostic state containers."""
from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
from .config import Config


@dataclasses.dataclass
class ModelState:
    """Spectral prognostic variables.

    Attributes
    ----------
    zeta: complex array, shape (L, nlat, nlon)
        Relative vorticity in spectral space.
    div: complex array, shape (L, nlat, nlon)
        Divergence in spectral space.
    T: complex array, shape (L, nlat, nlon)
        Temperature in spectral space.
    lnps: complex array, shape (nlat, nlon)
        Log surface pressure spectral coefficients.
    """

    zeta: jnp.ndarray
    div: jnp.ndarray
    T: jnp.ndarray
    lnps: jnp.ndarray


@jax.tree_util.register_pytree_node_class
class StateTree(ModelState):
    def tree_flatten(self):
        return (self.zeta, self.div, self.T, self.lnps), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        zeta, div, T, lnps = children
        return cls(zeta=zeta, div=div, T=T, lnps=lnps)


def zeros_state(cfg: Config) -> StateTree:
    shape = (cfg.L, cfg.nlat, cfg.nlon)
    z = jnp.zeros(shape, dtype=jnp.complex128)
    lnps = jnp.zeros((cfg.nlat, cfg.nlon), dtype=jnp.complex128)
    return StateTree(zeta=z, div=z, T=z, lnps=lnps)
