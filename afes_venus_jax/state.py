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
    zeta: complex array, shape (L, Lmax + 1, Lmax + 1)
        Relative vorticity in spherical-harmonic space.
    div: complex array, shape (L, Lmax + 1, Lmax + 1)
        Divergence in spherical-harmonic space.
    T: complex array, shape (L, Lmax + 1, Lmax + 1)
        Temperature in spherical-harmonic space.
    lnps: complex array, shape (Lmax + 1, Lmax + 1)
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
    shape = (cfg.L, cfg.Lmax + 1, cfg.Lmax + 1)
    z = jnp.zeros(shape, dtype=jnp.complex128)
    lnps = jnp.zeros((cfg.Lmax + 1, cfg.Lmax + 1), dtype=jnp.complex128)
    return StateTree(zeta=z, div=z, T=z, lnps=lnps)
