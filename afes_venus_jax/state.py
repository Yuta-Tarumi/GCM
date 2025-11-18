"""Model state container and initialisation utilities."""
from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp
from . import config, spectral


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


def empty_state(
    L: int | None = None,
    nlat: int | None = None,
    nlon: int | None = None,
) -> ModelState:
    cfg = config.DEFAULT
    if L is None:
        L = cfg.numerics.nlev
    if nlat is None:
        nlat = cfg.numerics.nlat
    if nlon is None:
        nlon = cfg.numerics.nlon
    zeros3d = jnp.zeros((L, nlat, nlon), dtype=jnp.complex64)
    zeros2d = jnp.zeros((nlat, nlon), dtype=jnp.complex64)
    return ModelState(zeta=zeros3d, div=zeros3d, T=zeros3d, lnps=zeros2d)


def initial_isothermal(
    T0: float = 240.0,
    ps: float | None = None,
    L: int | None = None,
    nlat: int | None = None,
    nlon: int | None = None,
) -> ModelState:
    cfg = config.DEFAULT
    if ps is None:
        ps = cfg.planet.surface_pressure
    base = empty_state(L, nlat, nlon)
    base.T = jnp.ones_like(base.T) * jnp.array(T0, dtype=jnp.complex64)
    base.lnps = jnp.ones_like(base.lnps) * jnp.log(jnp.array(ps))
    return base


def as_grid(state: ModelState):
    """Return grid-space versions of the prognostic variables."""
    return dataclasses.replace(
        state,
        zeta=spectral.synthesis_spec_to_grid(state.zeta),
        div=spectral.synthesis_spec_to_grid(state.div),
        T=spectral.synthesis_spec_to_grid(state.T),
        lnps=spectral.synthesis_spec_to_grid(state.lnps),
    )
