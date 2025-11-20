"""Model state container and initialisation utilities."""
from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp
from . import config, grid, spectral


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


def initial_solid_body_rotation(
    angular_speed: float | None = None,
    equatorial_speed: float | None = None,
    base_temperature: float = 240.0,
    ps: float | None = None,
    L: int | None = None,
    nlat: int | None = None,
    nlon: int | None = None,
) -> ModelState:
    """Initialise a solid-body westward circulation.

    Parameters
    ----------
    angular_speed : float, optional
        Angular velocity of the initial solid-body flow in rad s⁻¹. Defaults to
        the planetary rotation rate so that the atmosphere co-rotates with the
        surface.
    equatorial_speed : float, optional
        Target equatorial wind speed in m s⁻¹. If provided, overrides
        ``angular_speed`` using ``equatorial_speed / radius``; negative values
        correspond to westward flow.
    base_temperature : float
        Isothermal temperature used for all levels.
    ps : float, optional
        Reference surface pressure; defaults to the configured surface
        pressure.
    L, nlat, nlon : int, optional
        Vertical levels and horizontal resolution; defaults are pulled from the
        active configuration.
    """

    cfg = config.DEFAULT
    if equatorial_speed is not None:
        angular_speed = equatorial_speed / cfg.planet.radius
    if angular_speed is None:
        angular_speed = cfg.planet.rotation_rate
    if ps is None:
        ps = cfg.planet.surface_pressure
    if L is None:
        L = cfg.numerics.nlev
    if nlat is None:
        nlat = cfg.numerics.nlat
    if nlon is None:
        nlon = cfg.numerics.nlon

    lats, _, _ = grid.gaussian_grid(nlat=nlat, nlon=nlon)
    cosphi = grid.cosine_latitudes(lats)[:, None]
    u = (angular_speed * cfg.planet.radius) * cosphi
    v = jnp.zeros_like(u)

    dphi = lats[1] - lats[0]
    ucosp = u * cosphi
    d_ucosp_dphi = (jnp.roll(ucosp, -1, axis=0) - jnp.roll(ucosp, 1, axis=0)) / (2 * dphi)
    zeta_grid = -(1.0 / (cfg.planet.radius * cosphi)) * d_ucosp_dphi
    div_grid = jnp.zeros_like(v)

    base = initial_isothermal(T0=base_temperature, ps=ps, L=L, nlat=nlat, nlon=nlon)
    base = ModelState(
        zeta=spectral.analysis_grid_to_spec(jnp.broadcast_to(zeta_grid[None, ...], (L, nlat, nlon))),
        div=spectral.analysis_grid_to_spec(jnp.broadcast_to(div_grid[None, ...], (L, nlat, nlon))),
        T=base.T,
        lnps=base.lnps,
    )
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
