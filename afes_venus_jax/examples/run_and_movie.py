"""Generate a short example run and write a simple movie from the output.

This script runs a brief integration of the minimal AFES-Venus core,
records level-0 relative vorticity on the Gaussian grid, and writes
an animated gif (or other format supported by :mod:`imageio` via the
file extension) visualizing the evolution. The intent is to provide a
lightweight smoketest and a ready-to-run demo without introducing heavy
runtime dependencies or large output files.
"""
from __future__ import annotations
import argparse
import pathlib
from typing import Literal, Sequence

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .. import config, grid, spharm, state, timestep


def _lon_distance(lon: np.ndarray, center: float) -> np.ndarray:
    """Compute wrapped longitudinal distance preserving sign."""
    return np.arctan2(np.sin(lon - center), np.cos(lon - center))


def _vortex_pair_state(
    amplitude: float = 3e-5,
    half_spacing: float = np.deg2rad(25.0),
    lon_offset: float = np.deg2rad(60.0),
    sigma: float = np.deg2rad(12.0),
):
    """Return a balanced vortex pair to use as a deterministic test problem."""

    base = state.initial_isothermal()
    lats, lons, _ = grid.gaussian_grid()
    lon2d, lat2d = np.meshgrid(np.array(lons), np.array(lats))

    def gaussian(lat_c: float, lon_c: float) -> np.ndarray:
        return np.exp(
            -0.5
            * (
                ((lat2d - lat_c) / sigma) ** 2
                + (_lon_distance(lon2d, lon_c) / sigma) ** 2
            )
        )

    north = gaussian(half_spacing, -lon_offset)
    south = gaussian(-half_spacing, lon_offset)
    zeta_grid = amplitude * (north - south)
    zeta_spec = spharm.analysis_grid_to_spec(jnp.asarray(zeta_grid))
    base = base.__class__(
        zeta=base.zeta.at[0].set(zeta_spec),
        div=base.div,
        T=base.T,
        lnps=base.lnps,
    )
    return base


def _random_perturbation_state(seed: int) -> state.ModelState:
    """Return a near-rest state with a reproducible random vorticity perturbation."""

    base = state.initial_isothermal()
    key = jax.random.PRNGKey(seed)
    pert = 1e-6 * (jax.random.normal(key, base.zeta.shape) + 1j * 0.0)
    return state.ModelState(zeta=base.zeta + pert, div=base.div, T=base.T, lnps=base.lnps)


def run_example(
    nsteps: int = 30,
    seed: int = 0,
    scenario: Literal["vortex_pair", "random_noise"] = "vortex_pair",
):
    """Run a short integration and collect level-0 vorticity fields.

    Parameters
    ----------
    nsteps : int
        Number of forward steps to integrate.
    seed : int
        PRNG seed used to create a small perturbation on the base state.

    Returns
    -------
    frames : list of ndarray
        Real-valued arrays with shape (nlat, nlon) for each saved step.
    lats, lons : ndarray
        Grid latitude/longitude arrays in radians.
    scenario : {"vortex_pair", "random_noise"}
        Controls the initial state. ``"vortex_pair"`` seeds a balanced pair of
        opposite-signed vortices that remain bounded while shearing apart,
        whereas ``"random_noise"`` reproduces the previous behaviour where the
        isothermal base state is perturbed with small-amplitude noise.

    """

    if scenario == "vortex_pair":
        s0 = _vortex_pair_state()
    elif scenario == "random_noise":
        s0 = _random_perturbation_state(seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Integrate and record trajectory
    _, traj = timestep.integrate(s0, nsteps=nsteps)
    traj = jax.tree_util.tree_map(lambda arr: np.array(arr), traj)
    lats, lons, _ = grid.gaussian_grid()

    # Collect level-0 vorticity in grid space for each frame
    frames = []
    for istep in range(traj.zeta.shape[0]):
        step_zeta = traj.zeta[istep]
        zeta_grid = spharm.synthesis_spec_to_grid(step_zeta[0])
        frames.append(np.array(zeta_grid))
    return frames, np.array(lats), np.array(lons)


def _render_frame(field: np.ndarray, lats: np.ndarray, lons: np.ndarray, vmax: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 3))
    lon_deg = np.rad2deg(lons)
    lat_deg = np.rad2deg(lats)
    im = ax.imshow(
        field,
        origin="lower",
        extent=[lon_deg.min(), lon_deg.max(), lat_deg.min(), lat_deg.max()],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    ax.set_title("Level-0 relative vorticity")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return image


def make_movie(frames: Sequence[np.ndarray], lats: np.ndarray, lons: np.ndarray, path: str = "vort.gif", fps: int = 10):
    """Render frames to an animated image container based on extension.

    Parameters
    ----------
    frames : sequence of ndarrays
        Fields to plot; typically output from :func:`run_example`.
    lats, lons : ndarray
        Grid coordinates in radians (matching frame shapes).
    path : str
        Output filename; extension controls container (e.g., gif or mp4).
    fps : int
        Frames per second for the movie.
    """
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    vmax = max(np.max(np.abs(f)) for f in frames)

    with imageio.get_writer(output, mode="I", fps=fps) as writer:
        for frame in frames:
            writer.append_data(_render_frame(frame, lats, lons, vmax))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsteps", type=int, default=40, help="Number of forward steps to integrate")
    parser.add_argument("--output", type=str, default="t42l60_vort.gif", help="Filename for the rendered movie")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second in the output animation")
    parser.add_argument(
        "--scenario",
        choices=["vortex_pair", "random_noise"],
        default="vortex_pair",
        help="Initial condition used for the demonstration run",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the 'random_noise' scenario")
    args = parser.parse_args()

    frames, lats, lons = run_example(nsteps=args.nsteps, seed=args.seed, scenario=args.scenario)
    movie_path = make_movie(frames, lats, lons, path=args.output, fps=args.fps)
    print(f"Wrote movie to {movie_path.resolve()}")


if __name__ == "__main__":
    main()
