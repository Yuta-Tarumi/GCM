"""Reproduce the AFES-Venus superrotation experiment with paper-like settings.

This script mirrors the numerical setup described in the AFES-Venus superrotation
study by loading a dedicated YAML configuration, running a moderately long
integration, and writing both a diagnostic figure and a movie that highlight the
spin-up of the equatorial jet.  It reuses the superrotation demo helpers but
exposes the configuration so the experiment aligns with the paper's forcings and
diffusion choices.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import jax
import matplotlib.pyplot as plt
import numpy as np

from .. import config
from .superrotation_demo import collect_superrotation_history, render_movie


def _stack_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(frame) for frame in frames], axis=0)


def _band_mean_zonal_wind(u_fields: np.ndarray, lats: np.ndarray, half_width_deg: float) -> np.ndarray:
    lat_deg = np.rad2deg(lats)
    mask = np.abs(lat_deg) <= half_width_deg
    return u_fields[:, :, mask, :].mean(axis=(2, 3))


def _preferred_device() -> jax.Device:
    for dev in jax.devices():
        if dev.platform == "gpu":
            return dev
    return jax.devices()[0]


def plot_equatorial_spinup(
    u_fields: np.ndarray,
    lats: np.ndarray,
    heights: np.ndarray,
    times: np.ndarray,
    cfg: config.ModelConfig,
    output: Path,
    half_width_deg: float,
) -> tuple[Path, np.ndarray]:
    band_mean = _band_mean_zonal_wind(u_fields, lats, half_width_deg)
    time_days = times / cfg.rotation_period
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        band_mean.T,
        origin="lower",
        aspect="auto",
        extent=[time_days.min(), time_days.max(), heights.min() / 1000.0, heights.max() / 1000.0],
        cmap="RdBu_r",
        vmin=-band_mean.max(),
        vmax=band_mean.max(),
    )
    ax.set_xlabel("Time (Venus days)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Equatorial-mean zonal wind spin-up")
    fig.colorbar(im, ax=ax, label="m s$^{-1}$")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output, band_mean


def plot_snapshot(u_fields: np.ndarray, lats: np.ndarray, lons: np.ndarray, heights: np.ndarray, level_km: float, output: Path) -> Path:
    level_index = int(np.argmin(np.abs((heights / 1000.0) - level_km)))
    lon_deg = np.rad2deg(lons)
    lat_deg = np.rad2deg(lats)
    target = u_fields[-1, level_index]
    vmax = np.max(np.abs(target))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(
        target,
        origin="lower",
        extent=[lon_deg.min(), lon_deg.max(), lat_deg.min(), lat_deg.max()],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Zonal wind snapshot at {level_km:.0f} km (final step)")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, label="m s$^{-1}$")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="afes_venus_jax/config/afes_venus_paper.yaml", help="YAML file with AFES-Venus experiment settings")
    parser.add_argument("--nsteps", type=int, default=3600, help="Number of forward time steps to integrate")
    parser.add_argument("--sample-interval", type=int, default=36, help="Stride (in steps) between stored frames")
    parser.add_argument("--seed", type=int, default=2, help="Random seed used to initialise perturbations")
    parser.add_argument("--level-height-km", type=float, default=60.0, help="Target altitude for the movie map panel")
    parser.add_argument("--equator-band", type=float, default=20.0, help="Half-width in degrees for equatorial means")
    parser.add_argument("--movie", type=str, default="figures/afes_paper_superrotation.gif", help="Filename for the rendered animation")
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="figures",
        help="Directory where diagnostic figures will be written",
    )
    args = parser.parse_args()

    device = _preferred_device()
    print(f"Using JAX device: {device.platform} (id={device.id})")

    cfg = config.load_config(args.config)
    frames, profiles, times, lats, lons, heights = collect_superrotation_history(
        nsteps=args.nsteps,
        sample_interval=args.sample_interval,
        seed=args.seed,
        equatorial_half_width=args.equator_band,
        device=device,
        cfg=cfg,
    )

    movie_path = render_movie(
        frames,
        profiles,
        heights,
        lats,
        lons,
        level_height_km=args.level_height_km,
        times=times,
        path=args.movie,
        fps=6,
        cfg=cfg,
    )

    stacked = _stack_frames(frames)
    equatorial_fig, band_mean = plot_equatorial_spinup(
        stacked,
        lats,
        heights,
        times,
        cfg,
        output=Path(args.figure_dir) / "afes_paper_equatorial_spinup.png",
        half_width_deg=args.equator_band,
    )
    snapshot_fig = plot_snapshot(
        stacked,
        lats,
        lons,
        heights,
        level_km=args.level_height_km,
        output=Path(args.figure_dir) / "afes_paper_zonal_wind_snapshot.png",
    )
    final_profile = band_mean[-1]
    target_idx = int(np.argmin(np.abs((heights / 1000.0) - args.level_height_km)))
    peak_final = float(np.nanmax(np.abs(final_profile)))
    target_level = float(final_profile[target_idx])
    expected_range = (40.0, 120.0)  # m s^-1 from AFES-Venus cloud-top jets
    within_range = expected_range[0] <= abs(target_level) <= expected_range[1]
    status = "OK" if within_range else "OFF"
    print("Paper-comparison diagnostics:")
    print(f"  peak |u| in final equatorial profile: {peak_final:5.1f} m/s")
    print(f"  u at {args.level_height_km:.0f} km (equatorial mean): {target_level:5.1f} m/s -> {status}")
    if not within_range:
        print(
            "  (Outside the 40-120 m/s envelope quoted in the AFES-Venus study; "
            "extend the integration or increase solar peak heating to tighten the match.)"
        )
    print(f"Wrote movie to {movie_path.resolve()}")
    print(f"Wrote equatorial spin-up figure to {equatorial_fig.resolve()}")
    print(f"Wrote snapshot figure to {snapshot_fig.resolve()}")


if __name__ == "__main__":
    main()
