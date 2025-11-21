"""Nonlinear tendencies on the Gaussian grid."""
from __future__ import annotations

import jax.numpy as jnp

from .config import ModelConfig
from .spharm import (
    analysis_grid_to_spec,
    psi_chi_from_zeta_div,
    synthesis_spec_to_grid,
    uv_from_psi_chi,
    _lat_deltas,
)


def compute_tendencies(state, cfg: ModelConfig):
    psi, chi = psi_chi_from_zeta_div(state.zeta, state.div, cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)

    T_grid = synthesis_spec_to_grid(state.T, cfg)
    lnps_grid = synthesis_spec_to_grid(state.lnps, cfg)

    dphi = 2 * jnp.pi / cfg.nlon
    lats = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, cfg.nlat)
    dlat = _lat_deltas(lats)

    def grad(field):
        dlon = (jnp.roll(field, -1, axis=-1) - jnp.roll(field, 1, axis=-1)) / (2 * dphi)
        dlat_field = (jnp.roll(field, -1, axis=-2) - jnp.roll(field, 1, axis=-2)) / (2 * dlat[:, None])
        return dlon, dlat_field

    tzeta_list = []
    tdiv_list = []
    tT_list = []
    for k in range(cfg.L):
        ug = u[k]
        vg = v[k]
        zeta_g = synthesis_spec_to_grid(state.zeta[k], cfg)
        div_g = synthesis_spec_to_grid(state.div[k], cfg)
        dlon_zeta, dlat_zeta = grad(zeta_g)
        dlon_div, dlat_div = grad(div_g)
        adv_zeta = -(ug * dlon_zeta + vg * dlat_zeta)
        adv_div = -(ug * dlon_div + vg * dlat_div)
        tzeta_list.append(analysis_grid_to_spec(adv_zeta, cfg))
        tdiv_list.append(analysis_grid_to_spec(adv_div, cfg))
        Tg = T_grid[k]
        dlon_T, dlat_T = grad(Tg)
        adv_T = -(ug * dlon_T + vg * dlat_T)
        tT_list.append(analysis_grid_to_spec(adv_T, cfg))
    tzeta = jnp.stack(tzeta_list)
    tdiv = jnp.stack(tdiv_list)
    tT = jnp.stack(tT_list)

    dlon_lnps, dlat_lnps = grad(lnps_grid)
    mass_tendency = -analysis_grid_to_spec(u[0] * dlon_lnps + v[0] * dlat_lnps, cfg)
    return tzeta, tdiv, tT, mass_tendency
