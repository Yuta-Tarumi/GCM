"""Minimal Venus dry spin-up demo using the Gaussian-grid core."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from afes_venus_jax.config import DEFAULT_CFG, Config
from afes_venus_jax.grid import gaussian_grid
from afes_venus_jax.spharm import analysis_grid_to_spec, psi_chi_from_vort_div, synthesis_spec_to_grid, uv_from_psi_chi
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import jit_step
from afes_venus_jax.vertical import reference_temperature_profile

def save_wind_field(state, cfg: Config, step: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    psi, chi = psi_chi_from_vort_div(state.zeta[0], state.div[0], cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    wind_speed = jnp.sqrt(u**2 + v**2)

    lats, lons, _ = gaussian_grid(cfg)
    lon_deg = jnp.rad2deg(lons)
    lat_deg = jnp.rad2deg(lats)
    lon2d, lat2d = jnp.meshgrid(lon_deg, lat_deg)

    fig, ax = plt.subplots(figsize=(10, 4))
    speed_plot = ax.pcolormesh(jnp.asarray(lon2d), jnp.asarray(lat2d), jnp.asarray(wind_speed), shading="auto")
    skip = max(1, cfg.nlon // 32)
    ax.quiver(
        jnp.asarray(lon2d[::skip, ::skip]),
        jnp.asarray(lat2d[::skip, ::skip]),
        jnp.asarray(u[::skip, ::skip]),
        jnp.asarray(v[::skip, ::skip]),
        scale=300,
    )
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Wind field at step {step}")
    fig.colorbar(speed_plot, ax=ax, label="Wind speed (m/s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"wind_step_{step:03d}.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=int(2 * 86400 / DEFAULT_CFG.dt), help="Number of time steps to run")
    parser.add_argument("--snapshot-every", type=int, default=60, help="Interval (steps) for saving wind plots")
    parser.add_argument("--output-dir", type=Path, default=Path("wind_plots"), help="Directory for saved wind figures")
    parser.add_argument("--nlat", type=int, default=None, help="Optional Gaussian latitude count override")
    parser.add_argument("--nlon", type=int, default=None, help="Optional longitude count override")
    parser.add_argument("--lmax", type=int, default=None, help="Optional spectral truncation override")
    parser.add_argument("--levels", type=int, default=None, help="Optional vertical level count override")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DEFAULT_CFG
    if any(x is not None for x in (args.nlat, args.nlon, args.lmax, args.levels)):
        cfg = dataclasses.replace(
            cfg,
            nlat=args.nlat or cfg.nlat,
            nlon=args.nlon or cfg.nlon,
            Lmax=args.lmax or cfg.Lmax,
            L=args.levels or cfg.L,
        )
    state = zeros_state(cfg)

    # Add small vorticity perturbations in grid space
    key = jax.random.PRNGKey(0)
    grid_noise = 1e-3 * jax.random.normal(key, (cfg.L, cfg.nlat, cfg.nlon))
    noise_spec = analysis_grid_to_spec(grid_noise, cfg)
    state = state.__class__(zeta=state.zeta + noise_spec, div=state.div, T=state.T, lnps=state.lnps)

    # Initialise a Venus-like dry-adiabatic column capped by the observed cold top
    T_profile = reference_temperature_profile(cfg)
    T_grid = jnp.broadcast_to(T_profile[:, None, None], (cfg.L, cfg.nlat, cfg.nlon))
    T_spec = analysis_grid_to_spec(T_grid, cfg)
    state = state.__class__(zeta=state.zeta, div=state.div, T=T_spec, lnps=state.lnps)
    snapshot_steps = set(range(0, args.steps + 1, max(1, args.snapshot_every)))

    save_wind_field(state, cfg, step=0, out_dir=args.output_dir)
    for i in range(args.steps):
        state = jit_step(state, cfg)
        step_count = i + 1
        if step_count in snapshot_steps:
            save_wind_field(state, cfg, step=step_count, out_dir=args.output_dir)
        if step_count % 12 == 0:
            zeta_grid = jnp.abs(synthesis_spec_to_grid(state.zeta[0], cfg))
            T_grid = synthesis_spec_to_grid(state.T, cfg)
            col_mean_T = T_grid.mean(axis=(-2, -1))
            print(
                "Step {}: max|zeta|={:.3e}, Tsurf={:.1f} K, Tmid={:.1f} K, Ttop={:.1f} K".format(
                    step_count, zeta_grid.max(), col_mean_T[0], col_mean_T[cfg.L // 2], col_mean_T[-1]
                )
            )
    print("Spin-up complete")


if __name__ == "__main__":
    main()
