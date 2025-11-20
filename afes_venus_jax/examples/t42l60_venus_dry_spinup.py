"""Minimal Venus dry spin-up demo using the simplified core."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from afes_venus_jax.config import DEFAULT_CFG
from afes_venus_jax.spharm import analysis_grid_to_spec, psi_chi_from_vort_div, synthesis_spec_to_grid, uv_from_psi_chi
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import jit_step
from afes_venus_jax.vertical import reference_temperature_profile


def save_wind_field(state, cfg, step):
    psi, chi = psi_chi_from_vort_div(state.zeta[0], state.div[0], cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    wind_speed = jnp.sqrt(u**2 + v**2)

    lats = jnp.linspace(-90, 90, cfg.nlat)
    lons = jnp.linspace(0, 360, cfg.nlon, endpoint=False)
    lon2d, lat2d = jnp.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(10, 4))
    speed_plot = ax.pcolormesh(lon2d, lat2d, wind_speed, shading="auto")
    skip = max(1, cfg.nlon // 32)
    ax.quiver(lon2d[::skip, ::skip], lat2d[::skip, ::skip], u[::skip, ::skip], v[::skip, ::skip], scale=300)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Wind field at step {step}")
    fig.colorbar(speed_plot, ax=ax, label="Wind speed (m/s)")
    fig.tight_layout()
    fig.savefig(f"wind_step_{step:03d}.png", dpi=150)
    plt.close(fig)


def main():
    cfg = DEFAULT_CFG
    state = zeros_state(cfg)

    # Add small vorticity perturbations in grid space
    key = jax.random.PRNGKey(0)
    grid_noise = 1e-3 * jax.random.normal(key, (cfg.L, cfg.nlat, cfg.nlon))
    noise_spec = analysis_grid_to_spec(grid_noise)
    state = state.__class__(zeta=state.zeta + noise_spec, div=state.div, T=state.T, lnps=state.lnps)

    # Initialise a Venus-like dry-adiabatic column capped by the observed cold top
    # (e.g., VIRA; Seiff et al., 1985) instead of a hand-tuned linear profile.
    T_profile = reference_temperature_profile(cfg)
    T_grid = jnp.broadcast_to(T_profile[:, None, None], (cfg.L, cfg.nlat, cfg.nlon))
    T_spec = analysis_grid_to_spec(T_grid)
    state = state.__class__(zeta=state.zeta, div=state.div, T=T_spec, lnps=state.lnps)
    snapshot_steps = {0, 120, 240, 360, 480}
    nsteps = max(int(2 * 86400 / cfg.dt), max(snapshot_steps))

    save_wind_field(state, cfg, step=0)
    for i in range(nsteps):
        state = jit_step(state, cfg)
        step_count = i + 1
        if step_count in snapshot_steps:
            save_wind_field(state, cfg, step=step_count)
        if (i + 1) % 12 == 0:
            zeta_grid = jnp.abs(synthesis_spec_to_grid(state.zeta[0]))
            T_grid = synthesis_spec_to_grid(state.T)
            col_mean_T = T_grid.mean(axis=(-2, -1))
            print(
                "Step {}: max|zeta|={:.3e}, Tsurf={:.1f} K, Tmid={:.1f} K, Ttop={:.1f} K".format(
                    step_count, zeta_grid.max(), col_mean_T[0], col_mean_T[cfg.L // 2], col_mean_T[-1]
                )
            )
    print("Spin-up complete")


if __name__ == "__main__":
    main()
