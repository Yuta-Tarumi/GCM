"""Nonlinear tendency calculations in grid space."""
from __future__ import annotations

import jax
import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.grid as grid
import afes_venus_jax.spharm as sph
import afes_venus_jax.state as state
import afes_venus_jax.vertical as vertical


# Reference thermodynamic structure: Venus-like radiative equilibrium
def _reference_temperature_profile(L: int = cfg.L):
    profile = cfg.T_eq_profile
    if profile.shape[0] != L:
        # simple resample if the config was not initialised with this L
        sigma_full = jnp.linspace(0.5 / L, 1.0 - 0.5 / L, L)
        base_sigma = jnp.linspace(0.5 / profile.shape[0], 1.0 - 0.5 / profile.shape[0], profile.shape[0])
        profile = jnp.interp(sigma_full, base_sigma, profile)
    return profile


def _tau_rad_profile(L: int = cfg.L):
    profile = cfg.tau_rad_profile
    if profile.shape[0] != L:
        sigma_full = jnp.linspace(0.5 / L, 1.0 - 0.5 / L, L)
        base_sigma = jnp.linspace(0.5 / profile.shape[0], 1.0 - 0.5 / profile.shape[0], profile.shape[0])
        profile = jnp.interp(sigma_full, base_sigma, profile)
    return profile


LAT_GRID, LON_GRID, _ = grid.grid_arrays(cfg.nlat, cfg.nlon)
LAT_AXIS = LAT_GRID[:, 0]
COS_LAT = jnp.cos(LAT_AXIS)[:, None]


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


def _vertical_laplacian(field: jnp.ndarray, coord_half: jnp.ndarray):
    """Second derivative along a monotonic vertical coordinate.

    The diffusion coefficient ``KZ`` in :mod:`afes_venus_jax.config` is
    specified in ``m^2 s^-1``, so the coordinate passed here **must** be in
    metres. Previously this operator used non-dimensional ``sigma`` levels,
    which artificially amplified diffusion near the model top (where
    ``Δsigma`` becomes tiny) and immediately blew up temperature tendencies.
    """

    L = field.shape[0]
    dz = coord_half[1:] - coord_half[:-1]
    dz_full = 0.5 * (dz[1:] + dz[:-1])
    grad_half = jnp.zeros((L + 1,) + field.shape[1:], dtype=field.dtype)
    grad_half = grad_half.at[1:-1].set((field[1:] - field[:-1]) / dz_full[:, None, None])
    lap = jnp.zeros_like(field)
    lap = lap.at[0].set((grad_half[1] - grad_half[0]) / dz[0])
    lap = lap.at[1:-1].set((grad_half[2:-1] - grad_half[1:-2]) / dz[1:-1, None, None])
    lap = lap.at[-1].set((grad_half[-1] - grad_half[-2]) / dz[-1])
    return lap


def _bottom_rayleigh(u_field: jnp.ndarray, tau: float, ramp_levels: int):
    levels = jnp.arange(u_field.shape[0])
    weights = jnp.where(levels < ramp_levels, (ramp_levels - levels) / ramp_levels, 0.0)
    return -(weights[:, None, None] / tau) * u_field


def _upper_sponge(field: jnp.ndarray, tau_min: float, tau_base: float, top_levels: int):
    L = field.shape[0]
    level_idx = jnp.arange(L)
    start = max(0, L - top_levels)
    frac = jnp.clip((level_idx - start) / max(top_levels - 1, 1), 0.0, 1.0)
    tau = tau_base * (tau_min / tau_base) ** frac
    zonal_mean = jnp.mean(field, axis=-1, keepdims=True)
    eddy = field - zonal_mean
    return -(1.0 / tau[:, None, None]) * eddy


def compute_nonlinear_tendencies(
    mstate: state.ModelState, time_seconds: float = 0.0, nlat: int = cfg.nlat, nlon: int = cfg.nlon
):
    """Compute Eulerian nonlinear tendencies for the primitive equations.

    The formulation prognoses vorticity–divergence, temperature, and log
    surface pressure with explicit Coriolis and pressure-gradient
    forcings. Horizontal advection is handled in flux form with basic
    spherical metrics to maintain stability while capturing the dominant
    wave–mean-flow interactions.
    """

    psi, chi = sph.psi_chi_from_zeta_div(mstate.zeta, mstate.div)
    u, v = sph.uv_from_psi_chi(psi, chi, nlat, nlon)
    zeta = sph.synthesis_spec_to_grid(mstate.zeta, nlat, nlon)
    div = sph.synthesis_spec_to_grid(mstate.div, nlat, nlon)
    T = sph.synthesis_spec_to_grid(mstate.T, nlat, nlon)
    lnps = sph.synthesis_spec_to_grid(mstate.lnps, nlat, nlon)

    if nlat == LAT_AXIS.shape[0] and nlon == LON_GRID.shape[1]:
        lat_grid = LAT_GRID
        lon_grid = LON_GRID
        lat_axis = LAT_AXIS
        cos_lat = COS_LAT
    else:
        lat_grid, lon_grid, _ = grid.grid_arrays(nlat, nlon)
        lat_axis = jnp.array(grid.spectral_grid(nlat, nlon)[0])
        cos_lat = jnp.cos(lat_axis)[:, None]
    cos_safe = jnp.clip(cos_lat, 1e-6, None)
    dlon = 2 * jnp.pi / nlon

    def advect(field, u_field, v_field):
        if cfg.use_semi_lagrangian_advection:
            return _semi_lagrangian_advect(field, u_field, v_field, lat_axis, lat_grid, lon_grid, cos_safe)

        lon_flux = u_field * field * cos_lat
        dfdlon = (jnp.roll(lon_flux, -1, axis=-1) - jnp.roll(lon_flux, 1, axis=-1)) / (2 * dlon)

        lat_flux = v_field * field
        dfdlat = jnp.gradient(lat_flux, lat_axis, axis=-2)

        return -(dfdlon / (cfg.a * cos_safe) + dfdlat / cfg.a)

    def horizontal_grad(field):
        dfdlon = (jnp.roll(field, -1, axis=-1) - jnp.roll(field, 1, axis=-1)) / (2 * dlon)
        dfdlat = jnp.gradient(field, lat_axis, axis=-2)
        return dfdlon / (cfg.a * cos_safe), dfdlat / cfg.a

    def curl_from_uv(u_field, v_field):
        dvd_lon = (jnp.roll(v_field, -1, axis=-1) - jnp.roll(v_field, 1, axis=-1)) / (2 * dlon)
        ucos = u_field * cos_lat
        ducos_dlat = jnp.gradient(ucos, lat_axis, axis=-2)
        return (dvd_lon - ducos_dlat) / (cfg.a * cos_safe)

    def div_from_uv(u_field, v_field):
        dud_lon = (jnp.roll(u_field, -1, axis=-1) - jnp.roll(u_field, 1, axis=-1)) / (2 * dlon)
        vcos = v_field * cos_lat
        dvcos_dlat = jnp.gradient(vcos, lat_axis, axis=-2)
        return (dud_lon + dvcos_dlat) / (cfg.a * cos_safe)

    coriolis = 2.0 * cfg.Omega * jnp.sin(lat_axis)[:, None]
    _, sigma_half = vertical.sigma_levels(T.shape[0])
    z_full, z_half = vertical.level_altitudes(T.shape[0])
    ps_grid = cfg.ps_ref * jnp.exp(lnps)
    geopotential = vertical.hydrostatic_geopotential(T, ps_grid, sigma_half)
    grad_lnps_lon, grad_lnps_lat = horizontal_grad(lnps)

    # Vertical diffusion and drag terms (grid space)
    vdiff_u = cfg.vertical_diffusion_kz * _vertical_laplacian(u, z_half)
    vdiff_v = cfg.vertical_diffusion_kz * _vertical_laplacian(v, z_half)
    vdiff_T = cfg.vertical_diffusion_kz * _vertical_laplacian(T, z_half)
    rayleigh_u = _bottom_rayleigh(u, cfg.bottom_rayleigh_tau, cfg.bottom_rayleigh_ramp)
    rayleigh_v = _bottom_rayleigh(v, cfg.bottom_rayleigh_tau, cfg.bottom_rayleigh_ramp)
    sponge = cfg.sponge_config
    sponge_u = _upper_sponge(u, sponge.tau_min, sponge.tau_base, sponge.top_levels) if "u" in sponge.apply_to else 0.0
    sponge_v = _upper_sponge(v, sponge.tau_min, sponge.tau_base, sponge.top_levels) if "v" in sponge.apply_to else 0.0
    sponge_T = _upper_sponge(T, sponge.tau_min, sponge.tau_base, sponge.top_levels) if "T" in sponge.apply_to else 0.0

    zeta_tend_levels = []
    div_tend_levels = []
    T_adv = []

    for k in range(T.shape[0]):
        phi = geopotential[k]
        grad_phi_lon, grad_phi_lat = horizontal_grad(phi)

        u_adv = advect(u[k], u[k], v[k])
        v_adv = advect(v[k], u[k], v[k])

        pressure_u = grad_phi_lon + cfg.R_gas * T[k] * grad_lnps_lon
        pressure_v = grad_phi_lat + cfg.R_gas * T[k] * grad_lnps_lat

        u_tend = u_adv + coriolis * v[k] - pressure_u + vdiff_u[k] + rayleigh_u[k] + sponge_u[k]
        v_tend = v_adv - coriolis * u[k] - pressure_v + vdiff_v[k] + rayleigh_v[k] + sponge_v[k]

        zeta_tend_levels.append(curl_from_uv(u_tend, v_tend))
        div_tend_levels.append(div_from_uv(u_tend, v_tend))
        T_adv.append(advect(T[k], u[k], v[k]))

    zeta_tend = jnp.stack(zeta_tend_levels)
    div_tend = jnp.stack(div_tend_levels)
    T_adv = jnp.stack(T_adv)

    # Thermodynamics: advection + compressional heating + prescribed Newtonian cooling
    T_eq = _reference_temperature_profile(T.shape[0])[:, None, None]
    tau_rad = _tau_rad_profile(T.shape[0])[:, None, None]
    kappa = cfg.R_gas / cfg.cp
    cooling = -(T - T_eq) / tau_rad
    T_tend = T_adv - kappa * T * div + cooling + vdiff_T + sponge_T

    # Surface pressure tendency from mean divergence and advection by lowest-level winds
    lnps_tend = advect(lnps, u[0], v[0]) - jnp.mean(div, axis=0)

    zeta_spec = sph.analysis_grid_to_spec(zeta_tend)
    div_spec = sph.analysis_grid_to_spec(div_tend)
    T_spec = sph.analysis_grid_to_spec(T_tend)
    lnps_spec = sph.analysis_grid_to_spec(lnps_tend)
    return zeta_spec, div_spec, T_spec, lnps_spec


def _semi_lagrangian_advect(
    field: jnp.ndarray,
    u_field: jnp.ndarray,
    v_field: jnp.ndarray,
    lat_axis: jnp.ndarray,
    lat_grid: jnp.ndarray,
    lon_grid: jnp.ndarray,
    cos_lat: jnp.ndarray,
):
    """First-order semi-Lagrangian advection on the regular lat–lon grid.

    Departure points are estimated with a single Euler step and bilinear
    interpolation. The returned tendency represents ``(field_d - field)/dt``
    so it can replace the Eulerian flux-form advection in the explicit time
    update without altering the surrounding forcing terms.
    """

    cos_safe = jnp.clip(cos_lat, 1e-6, None)
    lat_depart = jnp.clip(lat_grid - cfg.dt * v_field / cfg.a, -jnp.pi / 2, jnp.pi / 2)
    lon_depart = lon_grid - cfg.dt * u_field / (cfg.a * cos_safe)
    lon_depart = jnp.mod(lon_depart, 2 * jnp.pi)

    field_depart = _bilinear_sample(field, lat_depart, lon_depart, lat_axis, lon_grid[0], 2 * jnp.pi / field.shape[-1])
    return (field_depart - field) / cfg.dt


def _bilinear_sample(field, lat_dep, lon_dep, lat_axis, lon0, dlon):
    """Bilinear interpolation on a latitude–longitude grid."""

    nlat, nlon = field.shape[-2:]
    lon_idx_float = (lon_dep - lon0) / dlon
    lon_idx0 = jnp.floor(lon_idx_float).astype(int) % nlon
    lon_w = lon_idx_float - lon_idx0
    lon_idx1 = (lon_idx0 + 1) % nlon

    lat_idx1 = jnp.clip(jnp.searchsorted(lat_axis, lat_dep), 1, nlat - 1)
    lat_idx0 = lat_idx1 - 1
    lat_w = (lat_dep - lat_axis[lat_idx0]) / jnp.maximum(lat_axis[lat_idx1] - lat_axis[lat_idx0], 1e-12)

    f00 = field[lat_idx0, lon_idx0]
    f01 = field[lat_idx0, lon_idx1]
    f10 = field[lat_idx1, lon_idx0]
    f11 = field[lat_idx1, lon_idx1]

    return (
        (1 - lat_w) * ((1 - lon_w) * f00 + lon_w * f01)
        + lat_w * ((1 - lon_w) * f10 + lon_w * f11)
    )
