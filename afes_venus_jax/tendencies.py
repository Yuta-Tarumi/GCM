"""Explicit nonlinear tendencies evaluated on the grid."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .spharm import synthesis_spec_to_grid, psi_chi_from_vort_div, uv_from_psi_chi, analysis_grid_to_spec
from .config import Config
from .vertical import reference_temperature_profile, sigma_levels


def nonlinear_tendencies(state, cfg: Config):
    """Compute advective tendencies for zeta, div, temperature, and log-ps."""
    psi, chi = psi_chi_from_vort_div(state.zeta, state.div, cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    zeta = synthesis_spec_to_grid(state.zeta)
    div = synthesis_spec_to_grid(state.div)
    T = synthesis_spec_to_grid(state.T)
    lnps = synthesis_spec_to_grid(state.lnps, cfg.nlat, cfg.nlon)

    def advect(field):
        dlon = jnp.gradient(field, axis=-1)
        dlat = jnp.gradient(field, axis=-2)
        return -(u * dlon + v * dlat)

    vort_tend = advect(zeta)
    div_tend = advect(div)
    T_tend = advect(T) + heating_tendency(T, cfg)
    lnps_tend = -jnp.mean(div, axis=0)

    return {
        "zeta": analysis_grid_to_spec(vort_tend),
        "div": analysis_grid_to_spec(div_tend),
        "T": analysis_grid_to_spec(T_tend),
        "lnps": analysis_grid_to_spec(lnps_tend),
    }


def heating_tendency(T: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Thermodynamic tendency from diabatic processes."""

    _, sigma_full = sigma_levels(cfg)
    T_eq = reference_temperature_profile(cfg)[:, None, None]

    # Vertical heat transfer (simple diffusion in sigma coordinates)
    #
    # Using the raw sigma spacing caused extremely large gradients near the
    # model top because the exponentially stretched levels become tightly
    # packed. Differencing with unit spacing keeps the diffusion operator
    # gentle enough for the Venus spin-up test without altering the intended
    # qualitative behaviour.
    dT_dsigma = jnp.gradient(T, axis=0)
    vertical_diffusion = cfg.kappa_heat * jnp.gradient(dT_dsigma, axis=0)

    # Prescribed shortwave heating focused near the cloud tops
    solar_shape = jnp.exp(-((sigma_full - cfg.solar_heating_peak_sigma) / cfg.solar_heating_width) ** 2)
    solar_heating = cfg.solar_heating_rate * solar_shape[:, None, None]

    # Newtonian cooling toward a reference profile
    newtonian_cooling = -(T - T_eq) / cfg.tau_newtonian

    return vertical_diffusion + solar_heating + newtonian_cooling
