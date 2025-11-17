"""Generate a short example run and write a simple movie from the output.

This script runs a brief integration of the minimal AFES-Venus core,
records level-0 relative vorticity on the Gaussian grid, and writes
an mp4 (or gif) visualizing the evolution. The intent is to provide a
lightweight smoketest and a ready-to-run demo without introducing heavy
runtime dependencies or large output files.
"""
from __future__ import annotations
import pathlib
from typing import Sequence

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .. import config, grid, spharm, state, timestep


def run_example(nsteps: int = 30, seed: int = 0):
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
    """
    base = state.initial_isothermal()
    key = jax.random.PRNGKey(seed)
    pert = 1e-6 * (jax.random.normal(key, base.zeta.shape) + 1j * 0.0)
    s0 = state.ModelState(zeta=base.zeta + pert, div=base.div, T=base.T, lnps=base.lnps)

    # Integrate and record trajectory
    _, traj = timestep.integrate(s0, nsteps=nsteps)
    lats, lons, _ = grid.gaussian_grid()

    # Collect level-0 vorticity in grid space for each frame
    frames = []
    for step_state in traj:
        zeta_grid = spharm.synthesis_spec_to_grid(step_state.zeta[0])
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
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def make_movie(frames: Sequence[np.ndarray], lats: np.ndarray, lons: np.ndarray, path: str = "vort.mp4", fps: int = 10):
    """Render frames to an mp4 (or gif based on extension).

    Parameters
    ----------
    frames : sequence of ndarrays
        Fields to plot; typically output from :func:`run_example`.
    lats, lons : ndarray
        Grid coordinates in radians (matching frame shapes).
    path : str
        Output filename; extension controls container (e.g., mp4 or gif).
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
    frames, lats, lons = run_example(nsteps=40, seed=0)
    movie_path = make_movie(frames, lats, lons, path="t42l60_vort.mp4", fps=8)
    print(f"Wrote movie to {movie_path.resolve()}")


if __name__ == "__main__":
    main()
