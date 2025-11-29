"""T42L60 Venus balanced random-wind spin-up demo."""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp

jax_enable_x64 = os.getenv("AFES_VENUS_JAX_ENABLE_X64", "false").lower() == "true"
if jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import afes_venus_jax.config as cfg
import afes_venus_jax.grid as grid
import afes_venus_jax.spharm as sph
import afes_venus_jax.state as state
import afes_venus_jax.tendencies as tend
from afes_venus_jax.examples import t42l60_venus_dry_spinup as base_example


def _vorticity_divergence(u: jnp.ndarray, v: jnp.ndarray):
    lats, lons, _ = grid.spectral_grid(cfg.nlat, cfg.nlon)
    lat_axis = jnp.array(lats)
    dlon = 2 * jnp.pi / cfg.nlon
    cosphi = jnp.cos(lat_axis)[:, None]
    cosphi_safe = jnp.clip(cosphi, 1e-8, None)

    d_v_dlon = (jnp.roll(v, -1, axis=-1) - jnp.roll(v, 1, axis=-1)) / (2 * dlon)
    d_ucosphi_dlat = jnp.gradient(u * cosphi, lat_axis, axis=-2)
    zeta = (d_v_dlon - d_ucosphi_dlat) / (cfg.a * cosphi_safe)

    d_u_dlon = (jnp.roll(u, -1, axis=-1) - jnp.roll(u, 1, axis=-1)) / (2 * dlon)
    d_v_dlat = jnp.gradient(v, lat_axis, axis=-2)
    div = (d_u_dlon / cosphi_safe + d_v_dlat) / cfg.a
    return zeta, div


def balanced_random_initial_condition(seed: int = 0, wind_std: float = 5.0):
    """Balanced initial condition with random winds and reference thermodynamics."""

    base = state.zeros_state()

    tref = tend._reference_temperature_profile()
    T_grid = tref[:, None, None] * jnp.ones((cfg.L, cfg.nlat, cfg.nlon))
    base.T = sph.analysis_grid_to_spec(T_grid)
    base.lnps = base.lnps.at[0, 0].set(0.0)

    key = jax.random.PRNGKey(seed)
    u_balanced = wind_std * jax.random.normal(key, (cfg.L, cfg.nlat, cfg.nlon))
    key, subkey = jax.random.split(key)
    v_balanced = wind_std * jax.random.normal(subkey, (cfg.L, cfg.nlat, cfg.nlon))

    def _remove_zonal_mean(field: jnp.ndarray) -> jnp.ndarray:
        zonal_mean = jnp.mean(field, axis=-1, keepdims=True)
        return field - zonal_mean

    def _rescale(field: jnp.ndarray, target: float) -> jnp.ndarray:
        std = jnp.std(field)
        return jnp.where(std > 0.0, field * (target / std), 0.0)

    target_u = 1.3 * wind_std
    target_v = 2.5 * wind_std

    for _ in range(5):
        u_balanced = _rescale(_remove_zonal_mean(u_balanced), target_u)
        v_balanced = _rescale(_remove_zonal_mean(v_balanced), target_v)

        zeta_grid, div_grid = _vorticity_divergence(u_balanced, v_balanced)
        psi, chi = sph.psi_chi_from_zeta_div(
            sph.analysis_grid_to_spec(zeta_grid), sph.analysis_grid_to_spec(div_grid)
        )
        u_balanced, v_balanced = sph.uv_from_psi_chi(psi, chi, cfg.nlat, cfg.nlon)

    u_balanced = _rescale(_remove_zonal_mean(u_balanced), wind_std)
    v_balanced = _rescale(_remove_zonal_mean(v_balanced), wind_std)

    final_zeta, final_div = _vorticity_divergence(u_balanced, v_balanced)
    zeta_spec = sph.analysis_grid_to_spec(final_zeta)
    div_spec = sph.analysis_grid_to_spec(final_div)

    for _ in range(3):
        psi, chi = sph.psi_chi_from_zeta_div(zeta_spec, div_spec)
        u_final, v_final = sph.uv_from_psi_chi(psi, chi, cfg.nlat, cfg.nlon)

        u_final = _rescale(_remove_zonal_mean(u_final), wind_std)
        v_final = _rescale(_remove_zonal_mean(v_final), wind_std)

        final_zeta, final_div = _vorticity_divergence(u_final, v_final)
        zeta_spec = sph.analysis_grid_to_spec(final_zeta)
        div_spec = sph.analysis_grid_to_spec(final_div)

    psi, chi = sph.psi_chi_from_zeta_div(zeta_spec, div_spec)
    u_out, v_out = sph.uv_from_psi_chi(psi, chi, cfg.nlat, cfg.nlon)

    factor_u = jnp.where(jnp.std(u_out) > 0.0, wind_std / jnp.std(u_out), 0.0)
    factor_v = jnp.where(jnp.std(v_out) > 0.0, wind_std / jnp.std(v_out), 0.0)
    u_out = _remove_zonal_mean(u_out * factor_u)
    v_out = _remove_zonal_mean(v_out * factor_v)

    final_zeta, final_div = _vorticity_divergence(u_out, v_out)
    zeta_spec = sph.analysis_grid_to_spec(final_zeta)
    div_spec = sph.analysis_grid_to_spec(final_div)

    base.zeta = zeta_spec
    base.div = div_spec
    return base


def sanity_check_balanced_state(mstate: state.ModelState, target_std: float = 5.0):
    """Ensure random-wind initial state is finite and thermodynamically uniform."""

    psi, chi = sph.psi_chi_from_zeta_div(mstate.zeta, mstate.div)
    u, v = sph.uv_from_psi_chi(psi, chi, cfg.nlat, cfg.nlon)
    T_grid = sph.synthesis_spec_to_grid(mstate.T, cfg.nlat, cfg.nlon)
    lnps_grid = sph.synthesis_spec_to_grid(mstate.lnps, cfg.nlat, cfg.nlon)

    for field in (u, v, T_grid, lnps_grid):
        if not jnp.isfinite(field).all():
            raise ValueError("Initial condition contains non-finite values.")

    std_u = float(jnp.std(u))
    std_v = float(jnp.std(v))
    lo, hi = 0.4 * target_std, 1.6 * target_std
    if not (lo <= std_u <= hi and lo <= std_v <= hi):
        raise ValueError(
            f"Wind standard deviations deviate from target: std_u={std_u:.2f}, std_v={std_v:.2f}"
        )

    for level in range(cfg.L):
        if (jnp.max(T_grid[level]) - jnp.min(T_grid[level])) >= 1e-2:
            raise ValueError("Temperature should be horizontally uniform on each level.")
    if (jnp.max(lnps_grid) - jnp.min(lnps_grid)) >= 1e-12:
        raise ValueError("Surface pressure should be horizontally uniform.")


def main():
    base_example.run_t42l60_venus_spinup(
        initial_condition_fn=balanced_random_initial_condition,
        sanity_check_fn=sanity_check_balanced_state,
    )


if __name__ == "__main__":
    main()
