"""Nonlinear Eulerian tendencies."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from . import spharm, grid, config


def _advect_scalar(q, u, v, lats, lons):
    dlat, dlon = lats[1] - lats[0], lons[1] - lons[0]
    dq_dlon = (jnp.roll(q, -1, axis=-1) - jnp.roll(q, 1, axis=-1)) / (2 * dlon)
    dq_dlat = (jnp.roll(q, -1, axis=-2) - jnp.roll(q, 1, axis=-2)) / (2 * dlat)
    cosphi = grid.cosine_latitudes(lats)[:, None]
    return -(u * dq_dlon / (config.a * cosphi) + v * dq_dlat / config.a)


@jax.jit
def nonlinear_tendencies(state):
    """Compute nonlinear tendencies in grid space then project to spectral."""
    lats, lons, _ = grid.gaussian_grid()
    psi, chi = spharm.psi_chi_from_vort_div(state.zeta, state.div)
    u, v = spharm.uv_from_psi_chi(psi, chi)

    def level_tend(zeta_l, div_l, T_l, u_l, v_l):
        dzeta = _advect_scalar(zeta_l.real, u_l, v_l, lats, lons)
        ddiv = _advect_scalar(div_l.real, u_l, v_l, lats, lons)
        dT = _advect_scalar(T_l.real, u_l, v_l, lats, lons)
        return dzeta, ddiv, dT

    dzeta, ddiv, dT = jax.vmap(level_tend)(state.zeta, state.div, state.T, u, v)
    dlnps = spharm.analysis_grid_to_spec(jnp.zeros_like(state.lnps.real))
    return spharm.analysis_grid_to_spec(dzeta), spharm.analysis_grid_to_spec(ddiv), spharm.analysis_grid_to_spec(dT), dlnps
