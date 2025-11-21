"""Initial-condition helpers for idealised Venus experiments."""
from __future__ import annotations

import jax.numpy as jnp

from .config import Config
from .grid import gaussian_grid
from .spharm import analysis_grid_to_spec, vort_div_from_uv
from .state import StateTree
from .vertical import sigma_levels, vertical_coordinates


def vira_temperature_profile(cfg: Config) -> jnp.ndarray:
    """Return a smoothed VIRA-inspired temperature column.

    The profile loosely follows the Venus International Reference Atmosphere
    with a weakly stratified layer near 55–60 km and a gradual transition to
    the cold upper atmosphere. Temperatures are clipped by ``cfg.T_cap`` to
    avoid unrealistically cold values aloft.
    """

    _, z_full = vertical_coordinates(cfg)
    alt_knots = jnp.array([0.0, 55e3, 60e3, 70e3, 120e3])
    temp_knots = jnp.array(
        [cfg.T_surface_ref, 380.0, 375.0, 260.0, cfg.T_cap]
    )
    return jnp.maximum(jnp.interp(z_full, alt_knots, temp_knots), cfg.T_cap)


def _solid_body_wind_profile(cfg: Config, max_speed: float = 100.0, ramp_height: float = 70e3) -> jnp.ndarray:
    """Height-varying equatorial wind that ramps to a target speed."""

    _, z_full = vertical_coordinates(cfg)
    ramp = jnp.clip(z_full / ramp_height, 0.0, 1.0)
    return jnp.where(z_full < ramp_height, max_speed * ramp, max_speed)


def _gradient_wind_geopotential_offsets(u_eq: jnp.ndarray, lats: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Integrate gradient-wind balance to obtain geopotential offsets.

    Parameters
    ----------
    u_eq: array_like
        Equatorial zonal wind profile [m s^-1] defined on full levels.
    lats: array_like
        Gaussian latitudes [rad].
    cfg: Config
        Model configuration for planetary constants.

    Returns
    -------
    offsets: jnp.ndarray
        Geopotential offsets relative to the equator with shape (L, nlat).
    """

    cos_lat = jnp.cos(lats)
    sin_lat = jnp.sin(lats)
    tan_lat = jnp.tan(lats)
    u_level = u_eq[:, None] * cos_lat[None, :]
    coriolis = 2 * cfg.Omega * sin_lat[None, :]
    grad_phi = cfg.a * coriolis * u_level + u_level**2 * tan_lat[None, :]

    offsets = jnp.zeros_like(grad_phi)
    equator_idx = int(jnp.argmin(jnp.abs(lats)))

    # March northward from the equator
    for j in range(equator_idx + 1, lats.size):
        dphi = lats[j] - lats[j - 1]
        mean_grad = 0.5 * (grad_phi[:, j] + grad_phi[:, j - 1])
        offsets = offsets.at[:, j].set(offsets[:, j - 1] + mean_grad * dphi)

    # March southward from the equator
    for j in range(equator_idx - 1, -1, -1):
        dphi = lats[j + 1] - lats[j]
        mean_grad = 0.5 * (grad_phi[:, j] + grad_phi[:, j + 1])
        offsets = offsets.at[:, j].set(offsets[:, j + 1] - mean_grad * dphi)

    return offsets


def superrotating_initial_state(cfg: Config) -> StateTree:
    """Construct a balanced super-rotating atmosphere.

    The flow follows solid-body rotation, ramping linearly from the surface to
    70 km and capping at 100 m s^-1 thereafter. Temperatures use a
    VIRA-inspired vertical column with a weakly stratified 55–60 km layer, and
    a gradient-wind-derived meridional structure that balances the imposed
    zonal flow. Surface pressure is set uniformly to ``cfg.ps_ref``.
    """

    lats, _, _ = gaussian_grid(cfg)
    cos_lat = jnp.cos(lats)

    u_eq = _solid_body_wind_profile(cfg)
    # Broadcast the zonal-mean solid-body wind across all longitudes. Without
    # the explicit longitude dimension the FFT-based spectral analysis sees a
    # single sample in longitude and rescales the field, collapsing the wind
    # amplitude toward zero. Repeating the field along ``nlon`` preserves the
    # intended cos(latitude) structure and 100 m s^-1 equatorial speed.
    u_grid = u_eq[:, None, None] * cos_lat[None, :, None]
    u_grid = jnp.broadcast_to(u_grid, (cfg.L, cfg.nlat, cfg.nlon))
    v_grid = jnp.zeros_like(u_grid)
    zeta_spec, div_spec = vort_div_from_uv(u_grid, v_grid, cfg)

    T_equator = vira_temperature_profile(cfg)
    sigma_half, _ = sigma_levels(cfg)
    p_half = sigma_half * cfg.ps_ref

    # Hydrostatic geopotential at the equator
    phi_half_eq = jnp.zeros(cfg.L + 1)
    for k in range(cfg.L):
        delta_phi = cfg.R_gas * T_equator[k] * jnp.log(p_half[k] / p_half[k + 1])
        phi_half_eq = phi_half_eq.at[k + 1].set(phi_half_eq[k] + delta_phi)
    phi_full_eq = 0.5 * (phi_half_eq[:-1] + phi_half_eq[1:])

    # Meridional geopotential structure from gradient-wind balance
    phi_offset_full = _gradient_wind_geopotential_offsets(u_eq, lats, cfg)
    phi_offset_half = jnp.zeros((cfg.L + 1, cfg.nlat))
    phi_offset_half = phi_offset_half.at[0].set(phi_offset_full[0])
    phi_offset_half = phi_offset_half.at[1:-1].set(0.5 * (phi_offset_full[:-1] + phi_offset_full[1:]))
    phi_offset_half = phi_offset_half.at[-1].set(phi_offset_full[-1])
    phi_half = phi_half_eq[:, None] + phi_offset_half

    # Recover hydrostatic temperatures at each latitude
    log_p_ratio = jnp.log(p_half[:-1] / p_half[1:])[:, None]
    T_bal = (phi_half[1:] - phi_half[:-1]) / (cfg.R_gas * log_p_ratio)
    T_bal = jnp.maximum(T_bal, cfg.T_cap)
    T_grid = T_bal[:, :, None].repeat(cfg.nlon, axis=2)

    T_spec = analysis_grid_to_spec(T_grid, cfg)
    lnps_spec = analysis_grid_to_spec(jnp.log(cfg.ps_ref) * jnp.ones((cfg.nlat, cfg.nlon)), cfg)

    return StateTree(zeta=zeta_spec, div=div_spec, T=T_spec, lnps=lnps_spec)

