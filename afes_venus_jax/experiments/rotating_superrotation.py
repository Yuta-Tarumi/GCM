"""One-command runner for the rotating superrotation experiment.

The script mirrors the regression test setup while offering a few knobs to
adjust runtime and output paths. It automatically prefers a GPU if available
and writes a short animation to the requested location.
"""

import argparse
from pathlib import Path

import jax

from afes_venus_jax import config
from afes_venus_jax.examples import superrotation_demo


def _preferred_device():
    for dev in jax.devices():
        if dev.platform == "gpu":
            return dev
    return jax.devices()[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the rotating superrotation experiment with one command.",
    )
    parser.add_argument(
        "--u-equator",
        type=float,
        default=70.0,
        help="Initial cloud-top jet speed (m/s) for the solid-body superrotation.",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=48,
        help="Total time steps to integrate (sets movie length).",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=6,
        help="Interval (steps) between stored frames.",
    )
    parser.add_argument(
        "--equator-band",
        type=float,
        default=20.0,
        help="Half-width of the equatorial averaging band (degrees).",
    )
    parser.add_argument(
        "--level-height-km",
        type=float,
        default=60.0,
        help="Target height (km) highlighted in the jet profile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for initial perturbations.",
    )
    parser.add_argument(
        "--movie",
        type=str,
        default="figures/rotating_superrotation.gif",
        help="Output path for the rendered animation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = _preferred_device()
    cfg = config.DEFAULT

    initial = superrotation_demo.initial_solid_body_superrotation(
        u_equator=args.u_equator,
        cfg=cfg,
    )
    initial = jax.tree_util.tree_map(lambda arr: jax.device_put(arr, device), initial)

    frames, profiles, times, lats, lons, heights = superrotation_demo.collect_superrotation_history(
        nsteps=args.nsteps,
        sample_interval=args.sample_interval,
        seed=args.seed,
        equatorial_half_width=args.equator_band,
        initial_state=initial,
        device=device,
        cfg=cfg,
    )

    movie_path = Path(args.movie)
    movie_path.parent.mkdir(parents=True, exist_ok=True)

    movie = superrotation_demo.render_movie(
        frames,
        profiles,
        heights,
        lats,
        lons,
        level_height_km=args.level_height_km,
        times=times,
        path=str(movie_path),
    )

    print(f"Using device: {device}")
    print(f"Saved movie to: {movie}")


if __name__ == "__main__":
    main()
