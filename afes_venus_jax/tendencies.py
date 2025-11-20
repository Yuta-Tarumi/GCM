"""Explicit nonlinear tendencies evaluated on the grid."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .spharm import (
    synthesis_spec_to_grid,
    psi_chi_from_vort_div,
    uv_from_psi_chi,
    analysis_grid_to_spec,
    scalar_gradients,
)
from .config import Config
from .grid import gaussian_grid
from .vertical import reference_temperature_profile, sigma_levels, vertical_coordinates


def nonlinear_tendencies(state, cfg: Config):
    """Compute advective tendencies for zeta, div, temperature, and log-ps."""
    psi, chi = psi_chi_from_vort_div(state.zeta, state.div, cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    zeta = synthesis_spec_to_grid(state.zeta, cfg)
    div = synthesis_spec_to_grid(state.div, cfg)
    T = synthesis_spec_to_grid(state.T, cfg)
    lnps = synthesis_spec_to_grid(state.lnps, cfg)

    lats, _, _ = gaussian_grid(cfg)
    cos_lat = jnp.cos(lats)[None, :, None]

    def advect(field):
        dlon, dlat = scalar_gradients(field, cfg)
        metric_u = dlon / (cfg.a * cos_lat)
        metric_v = dlat / cfg.a
        return -(u * metric_u + v * metric_v)

    _, z_full = vertical_coordinates(cfg)
    laplace_zeta = vertical_laplacian(zeta, z_full)
    laplace_div = vertical_laplacian(div, z_full)

    vort_tend = advect(zeta) + cfg.nu_vert * laplace_zeta
    div_tend = advect(div) + cfg.nu_vert * laplace_div
    T_tend = advect(T) + heating_tendency(T, cfg, z_full)
    lnps_tend = -jnp.mean(div, axis=0)

    vort_tend = apply_surface_rayleigh(vort_tend, zeta, cfg)
    div_tend = apply_surface_rayleigh(div_tend, div, cfg)

    vort_tend, div_tend, T_tend = apply_upper_sponge(
        vort_tend, div_tend, T_tend, zeta, div, T, z_full, cfg
    )

    return {
        "zeta": analysis_grid_to_spec(vort_tend, cfg),
        "div": analysis_grid_to_spec(div_tend, cfg),
        "T": analysis_grid_to_spec(T_tend, cfg),
        "lnps": analysis_grid_to_spec(lnps_tend, cfg),
    }


def heating_tendency(T: jnp.ndarray, cfg: Config, z_full: jnp.ndarray) -> jnp.ndarray:
    """Thermodynamic tendency from diabatic processes."""

    _, sigma_full = sigma_levels(cfg)
    T_eq = reference_temperature_profile(cfg)[:, None, None]

    dz = jnp.diff(z_full).mean()

    dT_dz = jnp.gradient(T, dz, axis=0)
    d2T_dz2 = jnp.gradient(dT_dz, dz, axis=0)
    vertical_diffusion = cfg.nu_vert * d2T_dz2

    # Prescribed shortwave heating focused near the cloud tops
    solar_shape = jnp.exp(-((sigma_full - cfg.solar_heating_peak_sigma) / cfg.solar_heating_width) ** 2)
    solar_heating = cfg.solar_heating_rate * solar_shape[:, None, None]

    # Newtonian cooling toward a reference profile
    newtonian_cooling = -(T - T_eq) / cfg.tau_newtonian

    return vertical_diffusion + solar_heating + newtonian_cooling


def vertical_laplacian(field: jnp.ndarray, z_full: jnp.ndarray) -> jnp.ndarray:
    """Second derivative with respect to height for vertically stacked fields."""

    dz = jnp.diff(z_full).mean()

    d_dz = jnp.gradient(field, dz, axis=0)
    return jnp.gradient(d_dz, dz, axis=0)


def apply_surface_rayleigh(tendency: jnp.ndarray, field: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Apply Rayleigh friction to the lowest model level."""

    damping = -field[0] / cfg.tau_rayleigh_surface
    return tendency.at[0].add(damping)


def apply_upper_sponge(
    vort_tend: jnp.ndarray,
    div_tend: jnp.ndarray,
    T_tend: jnp.ndarray,
    zeta: jnp.ndarray,
    div: jnp.ndarray,
    T: jnp.ndarray,
    z_full: jnp.ndarray,
    cfg: Config,
):
    """Damp eddy components above the sponge start altitude."""

    ramp = jnp.clip((z_full - cfg.sponge_start_alt) / (z_full[-1] - cfg.sponge_start_alt), 0.0, 1.0)
    sponge_coeff = (ramp**cfg.sponge_exponent) / cfg.tau_sponge_top

    def damp(field_tend, field):
        zonal_mean = field.mean(axis=-1, keepdims=True)
        eddy = field - zonal_mean
        return field_tend - sponge_coeff[:, None, None] * eddy

    vort_tend = damp(vort_tend, zeta)
    div_tend = damp(div_tend, div)
    T_tend = damp(T_tend, T)
    return vort_tend, div_tend, T_tend
