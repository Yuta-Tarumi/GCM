"""Tests for the illustrative initial condition."""

from __future__ import annotations

import numpy as np

import afes_venus_jax.config as cfg
import afes_venus_jax.grid as grid
import afes_venus_jax.spharm as sph
import afes_venus_jax.vertical as vertical
from afes_venus_jax.examples.t42l60_venus_dry_spinup import initial_condition, _zonal_wind_profile


def _zonal_wind_expectation():
    z_full, _ = vertical.level_altitudes()
    return np.array(_zonal_wind_profile(z_full))


def test_initial_condition_produces_smooth_rotating_atmosphere():
    mstate = initial_condition()

    psi, chi = sph.psi_chi_from_zeta_div(mstate.zeta, mstate.div)
    u, v = sph.uv_from_psi_chi(psi, chi, cfg.nlat, cfg.nlon)
    T_grid = sph.synthesis_spec_to_grid(mstate.T, cfg.nlat, cfg.nlon)
    lnps_grid = sph.synthesis_spec_to_grid(mstate.lnps, cfg.nlat, cfg.nlon)

    # No NaNs or infs in any field
    for field in (u, v, T_grid, lnps_grid):
        assert np.isfinite(np.array(field)).all()

    lats, _, _ = grid.spectral_grid(cfg.nlat, cfg.nlon)
    cos_lats = np.cos(np.array(lats))
    lon_mean = lambda arr: np.mean(np.array(arr), axis=-1)
    u_mean_lon = lon_mean(u)

    expected_profile = _zonal_wind_expectation()

    # Select the latitude closest to the equator to assess the vertical jet profile
    equator_idx = int(np.argmin(np.abs(np.array(lats))))
    np.testing.assert_allclose(
        u_mean_lon[:, equator_idx], expected_profile, atol=1.0, rtol=5e-3
    )

    # Winds follow cos(lat) scaling away from the equator at a mid-level altitude
    mid_level = cfg.L // 2
    scaled_profile = expected_profile[mid_level] * cos_lats
    midlat_mask = np.abs(np.array(lats)) < np.deg2rad(80.0)
    np.testing.assert_allclose(
        u_mean_lon[mid_level][midlat_mask], scaled_profile[midlat_mask], atol=2.0, rtol=5e-3
    )

    # The flow should be effectively axisymmetric; longitudinal structure
    # indicates numerical noise or an unintended perturbation in the base
    # state. Allow a tiny tolerance relative to the jet maximum to avoid
    # flakiness on extremely coarse grids.
    u_anom = u - u_mean_lon[..., None]
    max_u = np.max(np.abs(u_mean_lon))
    max_allowed_anom = max(1e-2, 1e-3 * max_u)
    assert np.max(np.abs(u_anom)) < max_allowed_anom

    # Divergence-free flow keeps meridional winds negligible
    assert np.max(np.abs(v)) < 5e-6

    # Temperature and surface pressure are horizontally uniform on each level
    for level in range(cfg.L):
        assert np.max(T_grid[level]) - np.min(T_grid[level]) < 1e-2
    assert np.max(lnps_grid) - np.min(lnps_grid) < 1e-12
