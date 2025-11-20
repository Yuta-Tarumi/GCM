"""Minimal Venus dry spin-up demo using the simplified core."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from afes_venus_jax.config import DEFAULT_CFG
from afes_venus_jax.spharm import analysis_grid_to_spec, synthesis_spec_to_grid
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import jit_step
from afes_venus_jax.vertical import reference_temperature_profile


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
    nsteps = int(2 * 86400 / cfg.dt)
    for i in range(nsteps):
        state = jit_step(state, cfg)
        if (i + 1) % 12 == 0:
            zeta_grid = jnp.abs(synthesis_spec_to_grid(state.zeta[0]))
            T_grid = synthesis_spec_to_grid(state.T)
            col_mean_T = T_grid.mean(axis=(-2, -1))
            print(
                "Step {}: max|zeta|={:.3e}, Tsurf={:.1f} K, Tmid={:.1f} K, Ttop={:.1f} K".format(
                    i + 1, zeta_grid.max(), col_mean_T[0], col_mean_T[cfg.L // 2], col_mean_T[-1]
                )
            )
    print("Spin-up complete")


if __name__ == "__main__":
    main()
