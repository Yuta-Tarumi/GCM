"""T42L60 Venus dry spin-up demo."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import afes_venus_jax.config as cfg
import afes_venus_jax.grid as grid
import afes_venus_jax.spharm as sph
import afes_venus_jax.state as state
import afes_venus_jax.tendencies as tend
import afes_venus_jax.timestep as timestep
import afes_venus_jax.vertical as vertical


def _zonal_wind_profile(z_full: jnp.ndarray, u_max: float = 100.0, z_peak: float = 70_000.0):
    """Return the target zonal wind amplitude for each full level."""

    ramp = jnp.clip(z_full / z_peak, 0.0, 1.0)
    return u_max * ramp


def initial_condition(option: int = 1):
    base = state.zeros_state()
    if option == 1:
        # Solid-body rotation with altitude-dependent amplitude.
        lats, lons, _ = grid.gaussian_grid(cfg.nlat, cfg.nlon)
        lat_component = jnp.array(lats) / 2 + jnp.sin(2 * jnp.array(lats)) / 4
        lon_axis = jnp.ones((cfg.nlon,))
        z_full, _ = vertical.level_altitudes()
        u_profile = _zonal_wind_profile(z_full)
        psi_levels = -cfg.a * u_profile[:, None, None] * lat_component[None, :, None] * lon_axis
        for k in range(cfg.L):
            psi_spec = sph.analysis_grid_to_spec(psi_levels[k])
            zeta_spec = sph.lap_spec(psi_spec)
            # Keep only the zonal-mean component to ensure purely zonal flow.
            zeta_spec = zeta_spec.at[:, 1:].set(0.0)
            base.zeta = base.zeta.at[k].set(zeta_spec)
    else:
        key = jax.random.PRNGKey(0)
        noise = 1e-6 * (jax.random.normal(key, base.zeta.shape) + 1j * jax.random.normal(key, base.zeta.shape))
        base.zeta = base.zeta + noise

    # Hydrostatic temperature structure (730 K at the bottom, 170 K at the top)
    tref = tend._reference_temperature_profile()
    T_grid = tref[:, None, None] * jnp.ones((cfg.L, cfg.nlat, cfg.nlon))
    base.T = sph.analysis_grid_to_spec(T_grid)

    # Uniform reference surface pressure
    base.lnps = base.lnps.at[0, 0].set(0.0)
    return base


def sanity_check_initial_condition(mstate: state.ModelState):
    """Confirm the illustrative initial condition matches the expected profiles."""

    psi, chi = sph.psi_chi_from_zeta_div(mstate.zeta, mstate.div)
    u, v = sph.uv_from_psi_chi(psi, chi, cfg.nlat, cfg.nlon)
    T_grid = sph.synthesis_spec_to_grid(mstate.T, cfg.nlat, cfg.nlon)
    lnps_grid = sph.synthesis_spec_to_grid(mstate.lnps, cfg.nlat, cfg.nlon)

    for field in (u, v, T_grid, lnps_grid):
        if not np.isfinite(np.array(field)).all():
            raise ValueError("Initial condition contains non-finite values.")

    lats, _, _ = grid.gaussian_grid(cfg.nlat, cfg.nlon)
    cos_lats = np.cos(np.array(lats))
    lon_mean = lambda arr: np.mean(np.array(arr), axis=-1)
    u_mean_lon = lon_mean(u)

    expected_profile = np.array(_zonal_wind_profile(vertical.level_altitudes()[0]))
    expected_zonal = expected_profile[:, None] * cos_lats[None, :]
    equator_idx = int(np.argmin(np.abs(np.array(lats))))
    if not np.allclose(u_mean_lon, expected_zonal, atol=1.0, rtol=5e-3):
        raise ValueError("Zonal-mean jet does not match expected cos(latitude) structure.")

    mid_level = cfg.L // 2
    scaled_profile = expected_profile[mid_level] * cos_lats
    midlat_mask = np.abs(np.array(lats)) < np.deg2rad(80.0)
    if not np.allclose(
        u_mean_lon[mid_level][midlat_mask], scaled_profile[midlat_mask], atol=2.0, rtol=5e-3
    ):
        raise ValueError("Mid-level zonal wind does not follow cos(lat) structure.")

    if np.max(np.abs(v)) >= 5e-6:
        raise ValueError("Meridional wind should be negligible for the initial state.")

    for level in range(cfg.L):
        if (np.max(T_grid[level]) - np.min(T_grid[level])) >= 1e-2:
            raise ValueError("Temperature should be horizontally uniform on each level.")
    if (np.max(lnps_grid) - np.min(lnps_grid)) >= 1e-12:
        raise ValueError("Surface pressure should be horizontally uniform.")


def plot_initial_snapshot(mstate: state.ModelState, levels: list[int] | None = None, filename: str = "initial_snapshot.png"):
    """Save a quick-look of u, v, T, and p for representative levels."""

    import matplotlib.pyplot as plt

    psi, chi = sph.psi_chi_from_zeta_div(mstate.zeta, mstate.div)
    u, v = sph.uv_from_psi_chi(psi, chi, cfg.nlat, cfg.nlon)
    T_grid = sph.synthesis_spec_to_grid(mstate.T, cfg.nlat, cfg.nlon)
    lnps_grid = sph.synthesis_spec_to_grid(mstate.lnps, cfg.nlat, cfg.nlon)
    sigma_full, _ = vertical.sigma_levels()
    p_grid = sigma_full[:, None, None] * cfg.ps_ref * jnp.exp(lnps_grid)

    if levels is None:
        candidates = [0, cfg.L // 4, cfg.L // 2, (3 * cfg.L) // 4, cfg.L - 1]
        levels = sorted({min(cfg.L - 1, max(0, lev)) for lev in candidates})

    lats, lons, _ = grid.gaussian_grid(cfg.nlat, cfg.nlon)
    lon2d, lat2d = np.meshgrid(np.rad2deg(np.array(lons)), np.rad2deg(np.array(lats)))

    fields = [(u, "u [m/s]"), (v, "v [m/s]"), (T_grid, "T [K]"), (p_grid, "p [Pa]")]

    fig, axes = plt.subplots(len(levels), len(fields), figsize=(4 * len(fields), 3 * len(levels)), constrained_layout=True)
    if len(levels) == 1:
        axes = axes[None, :]

    for row_idx, level in enumerate(levels):
        for col_idx, (field, label) in enumerate(fields):
            data = np.array(field[min(level, field.shape[0] - 1)])
            pcm = axes[row_idx, col_idx].pcolormesh(lon2d, lat2d, data, shading="auto")
            axes[row_idx, col_idx].set_title(f"{label} (level {level})")
            axes[row_idx, col_idx].set_xlabel("Longitude [deg]")
            axes[row_idx, col_idx].set_ylabel("Latitude [deg]")
            fig.colorbar(pcm, ax=axes[row_idx, col_idx], orientation="vertical", shrink=0.8)

    fig.suptitle("Initial snapshot of prognostic fields")
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    mstate = initial_condition()
    sanity_check_initial_condition(mstate)
    plot_initial_snapshot(mstate)
    nsteps = int(2 * 86400 / cfg.dt)
    for step_idx in range(nsteps):
        mstate = timestep.step(mstate)
        if step_idx % int(3 * 3600 / cfg.dt) == 0:
            psi, chi = mstate.zeta, mstate.div
            print(f"step {step_idx}: max|zeta|={jnp.max(jnp.abs(psi)).item():.3e}")


if __name__ == "__main__":
    main()
