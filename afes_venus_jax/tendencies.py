"""Explicit nonlinear tendencies evaluated on the grid."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .spharm import synthesis_spec_to_grid, psi_chi_from_vort_div, uv_from_psi_chi, analysis_grid_to_spec
from .config import Config


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
    T_tend = advect(T)
    lnps_tend = -jnp.mean(div, axis=0)

    return {
        "zeta": analysis_grid_to_spec(vort_tend),
        "div": analysis_grid_to_spec(div_tend),
        "T": analysis_grid_to_spec(T_tend),
        "lnps": analysis_grid_to_spec(lnps_tend),
    }
