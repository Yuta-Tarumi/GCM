"""Nonlinear tendency calculations in grid space."""
from __future__ import annotations

import jax
import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.grid as grid
import afes_venus_jax.spharm as sph
import afes_venus_jax.state as state
import afes_venus_jax.vertical as vertical


# Simple radiative forcing parameters
HEATING_PEAK_K_PER_DAY = 25.0  # peak shortwave heating in the 50–80 km layer
HEATING_CENTER_M = 65_000.0
HEATING_WIDTH_M = 8_000.0
TAU_NEWTONIAN = 20.0 * 86400.0
T_BOTTOM = 730.0
T_TOP = 170.0


def _reference_temperature_profile(L: int = cfg.L):
    z_full, _ = vertical.level_altitudes(L)
    lapse = (T_TOP - T_BOTTOM) / vertical.Z_TOP
    return T_BOTTOM + lapse * z_full


def _solar_heating_profile(L: int = cfg.L):
    z_full, _ = vertical.level_altitudes(L)
    peak = HEATING_PEAK_K_PER_DAY / 86400.0
    return peak * jnp.exp(-((z_full - HEATING_CENTER_M) ** 2) / (2 * HEATING_WIDTH_M**2))


LAT_GRID, LON_GRID, _ = grid.grid_arrays(cfg.nlat, cfg.nlon)


def _diurnal_heating_mask(time_seconds: float, nlat: int = cfg.nlat, nlon: int = cfg.nlon):
    """Return a day–night heating mask normalised to unit mean.

    The mask follows ``max(0, cos(lat) * cos(hour_angle))`` with the sub-solar
    longitude set by the planetary rotation rate. Normalising by the global
    mean keeps the net column heating comparable to the previous uniform
    prescription while introducing a diurnal cycle.
    """

    if nlat == LAT_GRID.shape[0] and nlon == LON_GRID.shape[1]:
        lat_grid = LAT_GRID
        lon_grid = LON_GRID
    else:
        lat_grid, lon_grid, _ = grid.grid_arrays(nlat, nlon)

    subsolar_lon = jnp.mod(cfg.Omega * time_seconds, 2 * jnp.pi)
    hour_angle = lon_grid - subsolar_lon
    cos_zenith = jnp.cos(lat_grid) * jnp.cos(hour_angle)
    daylight = jnp.where(cos_zenith > 0.0, cos_zenith, 0.0)
    mean_mask = jnp.mean(daylight)
    return jnp.where(mean_mask > 0.0, daylight / mean_mask, daylight)


def compute_nonlinear_tendencies(
    mstate: state.ModelState, time_seconds: float = 0.0, nlat: int = cfg.nlat, nlon: int = cfg.nlon
):
    """Compute Eulerian nonlinear tendencies.

    For this first-cut implementation the tendencies are limited to
    advective forms with simplified metrics to maintain stability while
    preserving the ζ–D structure. The formulation synthesises the grid
    winds from the streamfunction and velocity potential before
    constructing divergence and vorticity tendencies.
    """
    psi, chi = sph.psi_chi_from_zeta_div(mstate.zeta, mstate.div)
    u, v = sph.uv_from_psi_chi(psi, chi, nlat, nlon)
    zeta = sph.synthesis_spec_to_grid(mstate.zeta, nlat, nlon)
    div = sph.synthesis_spec_to_grid(mstate.div, nlat, nlon)
    T = sph.synthesis_spec_to_grid(mstate.T, nlat, nlon)
    lnps = sph.synthesis_spec_to_grid(mstate.lnps, nlat, nlon)

    # Simple nonlinear advection using flux form; zero metric corrections
    def advect(field, u_field, v_field):
        dlon = 2 * jnp.pi / nlon
        dlat = (jnp.linspace(-1, 1, nlat)[1] - jnp.linspace(-1, 1, nlat)[0]) * jnp.pi / 2
        flux_lon = u_field * field
        flux_lat = v_field * field
        dfdlon = (jnp.roll(flux_lon, -1, axis=-1) - jnp.roll(flux_lon, 1, axis=-1)) / (2 * dlon)
        dfdlat = (jnp.roll(flux_lat, -1, axis=-2) - jnp.roll(flux_lat, 1, axis=-2)) / (2 * dlat)
        return -(dfdlon + dfdlat)

    zeta_tend = jnp.stack([advect(zeta[k], u[k], v[k]) for k in range(zeta.shape[0])])
    div_tend = jnp.stack([advect(div[k], u[k], v[k]) for k in range(div.shape[0])])

    # Thermodynamics: advection + prescribed heating/cooling
    T_adv = jnp.stack([advect(T[k], u[k], v[k]) for k in range(T.shape[0])])
    T_eq = _reference_temperature_profile(T.shape[0])[:, None, None]
    heating_profile = _solar_heating_profile(T.shape[0])[:, None, None]
    diurnal = _diurnal_heating_mask(time_seconds, nlat=nlat, nlon=nlon)[None, :, :]
    heating = heating_profile * diurnal
    cooling = -(T - T_eq) / TAU_NEWTONIAN
    T_tend = T_adv + heating + cooling

    lnps_tend = advect(lnps, u[0], v[0])

    zeta_spec = sph.analysis_grid_to_spec(zeta_tend)
    div_spec = sph.analysis_grid_to_spec(div_tend)
    T_spec = sph.analysis_grid_to_spec(T_tend)
    lnps_spec = sph.analysis_grid_to_spec(lnps_tend)
    return zeta_spec, div_spec, T_spec, lnps_spec
