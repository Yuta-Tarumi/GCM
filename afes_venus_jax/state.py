"""Model state container."""
from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp
from typing import Any
from . import config


@dataclasses.dataclass
class ModelState:
    """Spectral prognostic variables.

    Attributes
    ----------
    zeta : (L, nlat, nlon) complex array
    div : (L, nlat, nlon) complex array
    T : (L, nlat, nlon) complex array
    lnps : (nlat, nlon) complex array
    """
    zeta: jnp.ndarray
    div: jnp.ndarray
    T: jnp.ndarray
    lnps: jnp.ndarray


jax.tree_util.register_pytree_node(
    ModelState,
    lambda s: ((s.zeta, s.div, s.T, s.lnps), None),
    lambda _, xs: ModelState(*xs),
)


def empty_state(L: int = config.L, nlat: int = config.nlat, nlon: int = config.nlon) -> ModelState:
    zeros3d = jnp.zeros((L, nlat, nlon), dtype=jnp.complex128)
    zeros2d = jnp.zeros((nlat, nlon), dtype=jnp.complex128)
    return ModelState(zeta=zeros3d, div=zeros3d, T=zeros3d, lnps=zeros2d)


def initial_isothermal(T0: float = 240.0, ps: float = config.ps_ref,
                       L: int = config.L, nlat: int = config.nlat, nlon: int = config.nlon) -> ModelState:
    base = empty_state(L, nlat, nlon)
    base.T = jnp.ones_like(base.T) * jnp.array(T0, dtype=jnp.complex128)
    base.lnps = jnp.ones_like(base.lnps) * jnp.log(jnp.array(ps))
    return base
