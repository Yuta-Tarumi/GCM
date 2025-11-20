"""Generate a long Venus run highlighting the build-up of superrotation.

This example starts from a noisy vorticity field, integrates for a user-defined
number of steps, and records zonal wind snapshots that illustrate how the
equatorial atmosphere spins up over time.  The resulting animation has two
panels: the left shows the horizontal zonal wind ``u`` at a selected altitude,
while the right plots the equatorial-mean vertical profile to make the
superrotating jet evident.
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Sequence

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .. import config, grid, spectral, state, vertical
from ..dynamics import integrators


def _random_initial_state(seed: int, amplitude: float = 1e-7) -> state.ModelState:
    """Seed a weak, zero-divergence perturbation around an isothermal state.

    The previous default (``3e-5``) injected vorticity energetic enough to blow up
    the AFES-Venus paper configuration before the zonal-mean jet could form.
    Using a gentler amplitude keeps the early integration linear while the
    Newtonian/solar forcings build the circulation toward the paper-like
    superrotation.
    """

    base = state.initial_isothermal()
    key = jax.random.PRNGKey(seed)
    noise = amplitude * (jax.random.normal(key, base.zeta.shape) + 1j * 0.0)
    return state.ModelState(zeta=base.zeta + noise, div=base.div, T=base.T, lnps=base.lnps)


def initial_solid_body_superrotation(
    u_equator: float = 60.0,
    cfg: config.ModelConfig | None = None,
) -> state.ModelState:
    """Construct a solid-body rotating state that mimics cloud-top superrotation.

    The target equatorial wind speed sets an equivalent angular velocity
    (``Ω = u_equator / a``) that is converted into a streamfunction with zero
    divergence.  Applying the Laplacian recovers the vorticity field required for
    the desired zonal wind, while temperature and surface pressure remain tied to
    the default isothermal profile.
    """

    if cfg is None:
        cfg = config.DEFAULT

    base = state.initial_isothermal(
        L=cfg.numerics.nlev, nlat=cfg.numerics.nlat, nlon=cfg.numerics.nlon
    )
    lats, lons, _ = grid.gaussian_grid(cfg.numerics.nlat, cfg.numerics.nlon)
    a = cfg.planet.radius
    zeta_band = jnp.ones((cfg.numerics.nlat, 1)) * (1.1 * u_equator / a)
    zeta_grid = jnp.broadcast_to(zeta_band, (cfg.numerics.nlev, cfg.numerics.nlat, cfg.numerics.nlon))
    zeta = spectral.analysis_grid_to_spec(zeta_grid.astype(jnp.complex64))
    div = spectral.analysis_grid_to_spec(jnp.zeros_like(zeta_grid, dtype=jnp.complex64))

    return state.ModelState(zeta=zeta, div=div, T=base.T, lnps=base.lnps)


def _equatorial_mean(u_field: np.ndarray, lats: np.ndarray, half_width_deg: float) -> np.ndarray:
    lat_deg = np.rad2deg(lats)
    mask = np.abs(lat_deg) <= half_width_deg
    if not np.any(mask):
        raise ValueError("Equatorial mask is empty; increase half_width_deg")
    subset = u_field[:, mask, :]
    return subset.mean(axis=(1, 2))


def _record_frame(
    cur: state.ModelState,
    lats: np.ndarray,
    half_width_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    psi, chi = spectral.psi_chi_from_vort_div(cur.zeta, cur.div)
    u, _ = spectral.uv_from_psi_chi(psi, chi)
    u_grid = np.asarray(u.real)
    profile = _equatorial_mean(u_grid, lats, half_width_deg)
    return u_grid, profile


def collect_superrotation_history(
    nsteps: int,
    sample_interval: int,
    seed: int,
    equatorial_half_width: float,
    initial_state: state.ModelState | None = None,
    device: jax.Device | None = None,
    cfg: config.ModelConfig | None = None,
):
    if cfg is None:
        cfg = config.DEFAULT
    cur = initial_state if initial_state is not None else _random_initial_state(seed)
    if device is not None:
        cur = jax.tree_util.tree_map(lambda arr: jax.device_put(arr, device), cur)

    lats, lons, _ = grid.gaussian_grid()
    heights = np.asarray(vertical.level_heights())

    frames: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    times: list[float] = []

    u_grid, profile = _record_frame(cur, np.asarray(lats), equatorial_half_width)
    frames.append(u_grid)
    profiles.append(profile)
    times.append(0.0)

    for step in range(1, nsteps + 1):
        t = step * cfg.numerics.dt
        cur = integrators.step(cur, t, cfg)
        if step % sample_interval == 0:
            u_grid, profile = _record_frame(cur, np.asarray(lats), equatorial_half_width)
            frames.append(u_grid)
            profiles.append(profile)
            times.append(t)

    return frames, profiles, np.asarray(times), np.asarray(lats), np.asarray(lons), heights


def _render_frame(
    u_field: np.ndarray,
    profile: np.ndarray,
    heights: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    level_index: int,
    vmax: float,
    profile_max: float,
    time_s: float,
    venus_day: float,
) -> np.ndarray:
    fig, (ax_map, ax_profile) = plt.subplots(1, 2, figsize=(11, 4.5))
    lon_deg = np.rad2deg(lons)
    lat_deg = np.rad2deg(lats)
    height_km = heights[level_index] / 1000.0
    im = ax_map.imshow(
        u_field[level_index],
        origin="lower",
        extent=[lon_deg.min(), lon_deg.max(), lat_deg.min(), lat_deg.max()],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    day_fraction = time_s / venus_day if venus_day else 0.0
    ax_map.set_title(f"u @ {height_km:.0f} km | t = {day_fraction:.1f} Venus days")
    ax_map.set_xlabel("Longitude (deg)")
    ax_map.set_ylabel("Latitude (deg)")
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label="m s$^{-1}$")

    ax_profile.plot(profile, heights / 1000.0, color="k")
    ax_profile.axvline(0.0, color="0.7", linestyle="--")
    ax_profile.set_xlim(-profile_max, profile_max)
    ax_profile.set_ylim(0, heights[-1] / 1000.0)
    ax_profile.set_xlabel("Equatorial-mean u (m s$^{-1}$)")
    ax_profile.set_ylabel("Altitude (km)")
    ax_profile.set_title("Equatorial jet evolution")

    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return image


def render_movie(
    frames: Sequence[np.ndarray],
    profiles: Sequence[np.ndarray],
    heights: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    level_height_km: float,
    times: np.ndarray,
    path: str,
    fps: int = 6,
    cfg: config.ModelConfig | None = None,
):
    if cfg is None:
        cfg = config.DEFAULT
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    level_index = int(np.argmin(np.abs((heights / 1000.0) - level_height_km)))
    vmax = max(float(np.max(np.abs(frame[level_index]))) for frame in frames)
    profile_max = max(float(np.max(np.abs(profile))) for profile in profiles)
    venus_day = cfg.rotation_period

    with imageio.get_writer(output, mode="I", fps=fps) as writer:
        for u_field, profile, t in zip(frames, profiles, times):
            image = _render_frame(
                u_field,
                profile,
                heights,
                lats,
                lons,
                level_index,
                vmax,
                profile_max,
                t,
                venus_day,
            )
            writer.append_data(image)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsteps", type=int, default=2400, help="Number of forward time steps to integrate")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=24,
        help="Number of model steps between saved frames (controls movie length)",
    )
    parser.add_argument(
        "--level-height-km",
        type=float,
        default=55.0,
        help="Target altitude for the zonal-wind map panel",
    )
    parser.add_argument(
        "--equator-band",
        type=float,
        default=20.0,
        help="Half-width in degrees for the equatorial mean profile",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used to initialise the vorticity field")
    parser.add_argument("--fps", type=int, default=6, help="Frames per second in the output animation")
    parser.add_argument(
        "--output",
        type=str,
        default="venus_superrotation.gif",
        help="Filename for the rendered animation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML configuration overriding the default Venus setup",
    )
    args = parser.parse_args()

    cfg = config.load_config(args.config) if args.config is not None else config.DEFAULT

    frames, profiles, times, lats, lons, heights = collect_superrotation_history(
        nsteps=args.nsteps,
        sample_interval=args.sample_interval,
        seed=args.seed,
        equatorial_half_width=args.equator_band,
        device=None,
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
        path=args.output,
        fps=args.fps,
        cfg=cfg,
    )
    print(f"Wrote movie to {movie_path.resolve()}")


if __name__ == "__main__":
    main()
