"""Minimal Venus dry spin-up demo using the simplified core."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from afes_venus_jax.config import DEFAULT_CFG
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import jit_step


def main():
    cfg = DEFAULT_CFG
    state = zeros_state(cfg)
    # add small perturbation
    key = jax.random.PRNGKey(0)
    noise = 1e-6 * jax.random.normal(key, state.zeta.shape)
    state = state.__class__(zeta=state.zeta + noise, div=state.div, T=state.T, lnps=state.lnps)
    nsteps = int(2 * 86400 / cfg.dt)
    for i in range(nsteps):
        state = jit_step(state, cfg)
        if (i + 1) % 12 == 0:
            zeta_grid = jnp.abs(jnp.fft.ifft2(state.zeta[0]).real)
            print(f"Step {i+1}: max|zeta|={zeta_grid.max():.3e}")
    print("Spin-up complete")


if __name__ == "__main__":
    main()
