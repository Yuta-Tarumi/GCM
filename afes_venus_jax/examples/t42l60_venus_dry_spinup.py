"""Minimal Venus dry spin-up demo using the Gaussian-grid core."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from afes_venus_jax.config import DEFAULT_CFG, Config
from afes_venus_jax.grid import gaussian_grid
from afes_venus_jax.initial_conditions import superrotating_initial_state
from afes_venus_jax.spharm import psi_chi_from_vort_div, synthesis_spec_to_grid, uv_from_psi_chi
from afes_venus_jax.timestep import jit_step
from afes_venus_jax.vertical import vertical_coordinates

def wind_speed_at_level(state, cfg: Config, level_idx: int):
    psi, chi = psi_chi_from_vort_div(state.zeta[level_idx], state.div[level_idx], cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    return jnp.sqrt(u**2 + v**2)


def save_wind_field(state, cfg: Config, step: int, out_dir: Path, lon2d, lat2d, level_idx: int = 0):
    out_dir.mkdir(parents=True, exist_ok=True)

    wind_speed = wind_speed_at_level(state, cfg, level_idx)

    fig, ax = plt.subplots(figsize=(10, 4))
    speed_plot = ax.pcolormesh(jnp.asarray(lon2d), jnp.asarray(lat2d), jnp.asarray(wind_speed), shading="auto")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Wind field at step {step} (level {level_idx})")
    fig.colorbar(speed_plot, ax=ax, label="Wind speed (m/s)")
    fig.tight_layout()
    fig.savefig(out_dir / f"wind_step_{step:03d}_level{level_idx:02d}.png", dpi=150)
    plt.close(fig)


def save_multiheight_wind_fields(state, cfg: Config, step: int, level_indices: Iterable[int], z_full, lon2d, lat2d, out_dir: Path):
    level_indices = list(level_indices)
    if not level_indices:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    speeds = [wind_speed_at_level(state, cfg, idx) for idx in level_indices]
    vmax = max(float(s.max()) for s in speeds)

    fig, axes = plt.subplots(
        1, len(level_indices), figsize=(4 * len(level_indices), 4), squeeze=False, constrained_layout=True
    )
    meshes = []
    for ax, speed, idx in zip(axes[0], speeds, level_indices):
        mesh = ax.pcolormesh(jnp.asarray(lon2d), jnp.asarray(lat2d), jnp.asarray(speed), shading="auto", vmin=0.0, vmax=vmax)
        alt_km = float(z_full[idx] / 1e3)
        ax.set_title(f"Level {idx} (~{alt_km:.0f} km)")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        meshes.append(mesh)
    fig.colorbar(meshes[-1], ax=axes.ravel().tolist(), label="Wind speed (m/s)")
    fig.suptitle(f"Wind field at step {step}")
    fig.savefig(out_dir / f"wind_levels_step_{step:03d}.png", dpi=150)
    plt.close(fig)


def save_multiheight_temperature_fields(
    state, cfg: Config, step: int, level_indices: Iterable[int], z_full, lon2d, lat2d, out_dir: Path
):
    level_indices = list(level_indices)
    if not level_indices:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    temperature_grid = jnp.asarray(synthesis_spec_to_grid(state.T, cfg))
    temps = [temperature_grid[idx] for idx in level_indices]
    vmin = min(float(temp.min()) for temp in temps)
    vmax = max(float(temp.max()) for temp in temps)

    fig, axes = plt.subplots(
        1, len(level_indices), figsize=(4 * len(level_indices), 4), squeeze=False, constrained_layout=True
    )
    meshes = []
    for ax, temp, idx in zip(axes[0], temps, level_indices):
        mesh = ax.pcolormesh(jnp.asarray(lon2d), jnp.asarray(lat2d), temp, shading="auto", vmin=vmin, vmax=vmax)
        alt_km = float(z_full[idx] / 1e3)
        ax.set_title(f"Level {idx} (~{alt_km:.0f} km)")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        meshes.append(mesh)
    fig.colorbar(meshes[-1], ax=axes.ravel().tolist(), label="Temperature (K)")
    fig.suptitle(f"Temperature field at step {step}")
    fig.savefig(out_dir / f"temp_levels_step_{step:03d}.png", dpi=150)
    plt.close(fig)


def levels_from_heights(target_heights_m: Iterable[float], z_full) -> list[int]:
    """Return nearest full-level indices for requested heights in meters."""

    indices = []
    for height_m in target_heights_m:
        idx = int(jnp.argmin(jnp.abs(z_full - height_m)))
        if idx not in indices:
            indices.append(idx)
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=int(2 * 86400 / DEFAULT_CFG.dt), help="Number of time steps to run")
    parser.add_argument("--snapshot-every", type=int, default=60, help="Interval (steps) for saving wind plots")
    parser.add_argument("--output-dir", type=Path, default=Path("out_plots"), help="Directory for saved output figures")
    parser.add_argument("--nlat", type=int, default=None, help="Optional Gaussian latitude count override")
    parser.add_argument("--nlon", type=int, default=None, help="Optional longitude count override")
    parser.add_argument("--lmax", type=int, default=None, help="Optional spectral truncation override")
    parser.add_argument("--levels", type=int, default=None, help="Optional vertical level count override")
    parser.add_argument(
        "--solar-diurnal-contrast",
        type=float,
        default=DEFAULT_CFG.solar_diurnal_contrast,
        help="Fractional day-night contrast for shortwave heating (0=uniform, 1=zero at local midnight)",
    )
    parser.add_argument(
        "--subsolar-longitude-deg",
        type=float,
        default=0.0,
        help="Longitude (deg) of the subsolar point where heating peaks",
    )
    parser.add_argument(
        "--plot-heights-km",
        type=str,
        default="0,30,60,90",
        help="Comma-separated target altitudes (km) for multi-height wind plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = dataclasses.replace(
        DEFAULT_CFG,
        nlat=args.nlat or DEFAULT_CFG.nlat,
        nlon=args.nlon or DEFAULT_CFG.nlon,
        Lmax=args.lmax or DEFAULT_CFG.Lmax,
        L=args.levels or DEFAULT_CFG.L,
        solar_diurnal_contrast=args.solar_diurnal_contrast,
        subsolar_longitude=float(jnp.deg2rad(args.subsolar_longitude_deg)),
    )
    state = superrotating_initial_state(cfg)
    snapshot_steps = set(range(0, args.steps + 1, max(1, args.snapshot_every)))

    lats, lons, _ = gaussian_grid(cfg)
    lon_deg = jnp.rad2deg(lons)
    lat_deg = jnp.rad2deg(lats)
    lon2d, lat2d = jnp.meshgrid(lon_deg, lat_deg)
    _, z_full = vertical_coordinates(cfg)

    plot_heights_km = [float(val) for val in args.plot_heights_km.split(",") if val.strip()]
    level_indices = levels_from_heights([h * 1e3 for h in plot_heights_km], z_full)

    save_wind_field(state, cfg, step=0, out_dir=args.output_dir, lon2d=lon2d, lat2d=lat2d)
    save_multiheight_wind_fields(state, cfg, step=0, level_indices=level_indices, z_full=z_full, lon2d=lon2d, lat2d=lat2d, out_dir=args.output_dir)
    save_multiheight_temperature_fields(state, cfg, step=0, level_indices=level_indices, z_full=z_full, lon2d=lon2d, lat2d=lat2d, out_dir=args.output_dir)
    for i in range(args.steps):
        state = jit_step(state, cfg)
        step_count = i + 1
        if step_count in snapshot_steps:
            save_wind_field(state, cfg, step=step_count, out_dir=args.output_dir, lon2d=lon2d, lat2d=lat2d)
            save_multiheight_wind_fields(state, cfg, step=step_count, level_indices=level_indices, z_full=z_full, lon2d=lon2d, lat2d=lat2d, out_dir=args.output_dir)
            save_multiheight_temperature_fields(state, cfg, step=step_count, level_indices=level_indices, z_full=z_full, lon2d=lon2d, lat2d=lat2d, out_dir=args.output_dir)
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
