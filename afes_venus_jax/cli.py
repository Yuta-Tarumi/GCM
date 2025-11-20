"""Command-line entry point for running AFES-Venus demos."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from . import driver


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Run the AFES-Venus paper demonstration with configurable runtime "
            "options."
        )
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=240,
        help="Number of timesteps to integrate (paper demo default: 240).",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=24,
        help="Stride, in timesteps, between saved history outputs.",
    )
    parser.add_argument(
        "--angular-speed",
        type=float,
        default=None,
        help=(
            "Solid-body angular speed in rad s^-1. Overrides equatorial wind if "
            "provided."
        ),
    )
    parser.add_argument(
        "--equatorial-speed",
        type=float,
        default=-100.0,
        help=(
            "Solid-body equatorial speed in m s^-1 (westward negative). Ignored "
            "when angular speed is supplied."
        ),
    )
    parser.add_argument(
        "--base-temperature",
        type=float,
        default=240.0,
        help="Base temperature for the initial condition (K).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional NetCDF file to write the resulting history dataset to.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ds = driver.run_afes_venus_paper_demo(
        nsteps=args.nsteps,
        history_stride=args.history_stride,
        angular_speed=args.angular_speed,
        equatorial_speed=args.equatorial_speed,
        base_temperature=args.base_temperature,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(args.output)
        print(f"Saved history dataset to {args.output}")

    print(ds)


if __name__ == "__main__":  # pragma: no cover
    main()
