"""Demonstrate softer hyperdiffusion settings for the Venus demo.

This script runs the same two-day spin-up as :mod:`t42l60_venus_dry_spinup`
but with weaker damping to better resolve small-scale vorticity.  The
settings are inspired by long-integration studies that target eddy-rich
superrotation (e.g., Peng et al., 2023), which typically use:

* a longer e-folding time at the truncation scale (a few Venus days instead
  of 0.1 Earth days),
* a higher-order operator to confine dissipation to the smallest resolved
  scales.

Adjust ``tau_hdiff`` and ``order_hdiff`` below to match the experiment design
you have in mind.  The ``hyperdiffusion_timescale`` helper reports the implied
e-folding time at different fractions of the spectral radius to make this more
transparent.
"""

from __future__ import annotations

import dataclasses
import jax.numpy as jnp
from afes_venus_jax.config import DEFAULT_CFG
from afes_venus_jax.diffusion import hyperdiffusion_timescale
from afes_venus_jax.initial_conditions import superrotating_initial_state
from afes_venus_jax.spharm import synthesis_spec_to_grid
from afes_venus_jax.timestep import jit_step


def describe_diffusion(cfg):
    for frac in (1.0, 0.5, 0.25):
        tau = hyperdiffusion_timescale(cfg, frac)
        print(f"Hyperdiffusion tau @ {frac:.2f}*Lmax: {tau/86400:.2f} days")


def main():
    cfg = dataclasses.replace(
        DEFAULT_CFG,
        tau_hdiff=2.0 * 86400.0,  # ~2 days at the truncation scale
        order_hdiff=8,  # confine damping to the smallest resolved scales
        ra=0.02,  # mild Robert–Asselin filter to preserve phase accuracy
    )

    print("Diffusion settings for the low-damping experiment:")
    describe_diffusion(cfg)

    state = superrotating_initial_state(cfg)

    nsteps = int(2 * 86400 / cfg.dt)
    for i in range(nsteps):
        state = jit_step(state, cfg)
        if (i + 1) % 12 == 0:
            zeta_grid = jnp.abs(synthesis_spec_to_grid(state.zeta[0], cfg))
            T_grid = synthesis_spec_to_grid(state.T, cfg)
            col_mean_T = T_grid.mean(axis=(-2, -1))
            print(
                "Step {}: max|zeta|={:.3e}, Tsurf={:.1f} K, Tmid={:.1f} K, Ttop={:.1f} K".format(
                    i + 1, zeta_grid.max(), col_mean_T[0], col_mean_T[cfg.L // 2], col_mean_T[-1]
                )
            )

    print("Low-diffusion spin-up complete")


if __name__ == "__main__":
    main()
