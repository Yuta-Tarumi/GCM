"""T42L60 Venus dry spin-up demo."""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

jax_enable_x64 = os.getenv("AFES_VENUS_JAX_ENABLE_X64", "false").lower() == "true"
# Keep single precision by default to avoid overwhelming limited GPUs. Opt-in to
# 64-bit via AFES_VENUS_JAX_ENABLE_X64=true when needed for validation runs.
if jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

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
    # Impose a plateau above ``z_peak`` so upper levels share the same
    # structure as the 70 km level instead of tapering to rest.
    plateau = jnp.where(z_full >= z_peak, 1.0, ramp)
    profile = u_max * plateau
    # Keep only the lowest level at rest to avoid ringing from sharp
    # truncation when constructing purely zonal flows for tests.
    profile = profile.at[0].set(0.0)
    profile = profile.at[-1].set(0.0)
    return profile


def initial_condition(option: int = 1):
    base = state.zeros_state()
    if option == 1:
        # Solid-body rotation with altitude-dependent amplitude. Build the
        # streamfunction directly on the latitude grid using the same
        # quadrature spacing that ``uv_from_psi_chi`` differentiates over so we
        # recover the desired ``u = u_profile * cos(lat)`` structure even on
        # coarse grids.
        lats, lons, _ = grid.spectral_grid(cfg.nlat, cfg.nlon)
        lat_axis = jnp.array(lats)
        lon_axis = jnp.ones((cfg.nlon,))
        z_full, _ = vertical.level_altitudes()
        u_profile = _zonal_wind_profile(z_full)

        # Integrate psi(lat) such that discrete meridional derivatives match the
        # target zonal wind on the Gaussian grid. Start integration at the South
        # Pole to avoid introducing an arbitrary constant and broadcast across
        # longitude because the flow is axisymmetric.
        cos_lats = jnp.cos(lat_axis)
        target_zonal = u_profile[:, None] * cos_lats[None, :]
        dlat = jnp.diff(lat_axis)
        # Use cumulative trapezoidal integration for each level.
        def integrate_streamfunction(u_row):
            # psi[0] = 0 at the pole.
            increments = -cfg.a * 0.5 * (u_row[1:] * cos_lats[1:] + u_row[:-1] * cos_lats[:-1]) * dlat
            psi_lat = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(increments)])
            return psi_lat

        psi_lat_levels = jax.vmap(integrate_streamfunction)(target_zonal)
        psi_levels = psi_lat_levels[:, :, None] * lon_axis
        for k in range(cfg.L):
            psi_spec = sph.analysis_grid_to_spec(psi_levels[k])
            zeta_spec = sph.lap_spec(psi_spec)
            # Keep only the zonal-mean component to ensure purely zonal flow.
            zeta_spec = zeta_spec.at[:, 1:].set(0.0)
            base.zeta = base.zeta.at[k].set(zeta_spec)

        # The streamfunction construction already matches the desired zonal
        # winds on the Gaussian grid, so additional rescaling is unnecessary and
        # can amplify numerical noise when ``denom`` becomes tiny at low
        # truncation. Leave the vorticity amplitudes unchanged.
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

    lats, _, _ = grid.spectral_grid(cfg.nlat, cfg.nlon)
    cos_lats = np.cos(np.array(lats))
    lon_mean = lambda arr: np.mean(np.array(arr), axis=-1)
    u_mean_lon = lon_mean(u)

    expected_profile = np.array(_zonal_wind_profile(vertical.level_altitudes()[0]))
    expected_zonal = expected_profile[:, None] * cos_lats[None, :]
    equator_idx = int(np.argmin(np.abs(np.array(lats))))
    max_diff = np.max(np.abs(u_mean_lon - expected_zonal))
    if max_diff > 25.0:
        raise ValueError(
            "Zonal-mean jet does not match expected cos(latitude) structure. "
            f"max_abs_diff={max_diff:.3f} m/s"
        )

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

    lats, lons, _ = grid.spectral_grid(cfg.nlat, cfg.nlon)
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


def save_snapshot(mstate: state.ModelState, step_idx: int, levels: list[int] | None = None):
    """Save a snapshot for the provided simulation step."""

    filename = f"snapshot_step_{step_idx:05d}.png"
    plot_initial_snapshot(mstate, levels=levels, filename=filename)


def main():
    run_t42l60_venus_spinup()


def _diagnostics(mstate: state.ModelState):
    psi, chi = sph.psi_chi_from_zeta_div(mstate.zeta, mstate.div)
    u, v = sph.uv_from_psi_chi(psi, chi, cfg.nlat, cfg.nlon)
    T_grid = sph.synthesis_spec_to_grid(mstate.T, cfg.nlat, cfg.nlon)
    lnps_grid = sph.synthesis_spec_to_grid(mstate.lnps, cfg.nlat, cfg.nlon)
    ps_grid = cfg.ps_ref * jnp.exp(lnps_grid)
    T_eq = tend._reference_temperature_profile()[:, None, None]
    diag = {
        "max_u": float(jnp.max(jnp.abs(u))),
        "max_v": float(jnp.max(jnp.abs(v))),
        "max_T_prime": float(jnp.max(jnp.abs(T_grid - T_eq))),
        "ps_range": (float(jnp.min(ps_grid)), float(jnp.max(ps_grid))),
        "global_ke": float(jnp.mean(0.5 * (u**2 + v**2))),
        "mean_T": float(jnp.mean(T_grid)),
    }
    return diag


def run_t42l60_venus_spinup(nsteps: int | None = None, save_snapshots: bool = True):
    """Smoke test for AFES-like T42L60 defaults.

    Runs a short integration with conservative time step and AFES-style
    diffusion/drag/radiation intended to avoid blow-ups while spinning up
    the super-rotating circulation. This is not a production-quality
    simulation, but it should remain numerically stable for several
    Earth days with the packaged defaults.
    """

    mstate = initial_condition()
    sanity_check_initial_condition(mstate)
    if save_snapshots:
        save_snapshot(mstate, step_idx=0)
    if nsteps is None:
        nsteps = int(5 * 86400 / cfg.dt)

    for step_idx in range(1, nsteps + 1):
        time_seconds = (step_idx - 1) * cfg.dt
        mstate = timestep.step(mstate, time_seconds=time_seconds)
        if step_idx % 240 == 0:
            diag = _diagnostics(mstate)
            print(
                f"step {step_idx}: |u|={diag['max_u']:.2f} m/s, |v|={diag['max_v']:.2f} m/s, "
                f"|T'|={diag['max_T_prime']:.2f} K, ps=[{diag['ps_range'][0]:.2e},{diag['ps_range'][1]:.2e}]"
            )
        if save_snapshots and step_idx % int(12 * 3600 / cfg.dt) == 0:
            save_snapshot(mstate, step_idx=step_idx)

    return mstate


if __name__ == "__main__":
    main()
