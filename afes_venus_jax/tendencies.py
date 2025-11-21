"""Nonlinear tendency calculations in grid space."""
from __future__ import annotations

import jax
import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.spharm as sph
import afes_venus_jax.state as state


def compute_nonlinear_tendencies(mstate: state.ModelState, nlat: int = cfg.nlat, nlon: int = cfg.nlon):
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
    T_tend = jnp.stack([advect(T[k], u[k], v[k]) for k in range(T.shape[0])])
    lnps_tend = advect(lnps, u[0], v[0])

    zeta_spec = sph.analysis_grid_to_spec(zeta_tend)
    div_spec = sph.analysis_grid_to_spec(div_tend)
    T_spec = sph.analysis_grid_to_spec(T_tend)
    lnps_spec = sph.analysis_grid_to_spec(lnps_tend)
    return zeta_spec, div_spec, T_spec, lnps_spec
