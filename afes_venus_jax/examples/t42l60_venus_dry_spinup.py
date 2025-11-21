"""T42L60 Venus dry spin-up demo."""
from __future__ import annotations

import jax
import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.state as state
import afes_venus_jax.timestep as timestep


def initial_condition(option: int = 1):
    base = state.zeros_state()
    if option == 1:
        # solid-body rotation imprint
        ell = 1
        m = 1
        base.zeta = base.zeta.at[:, ell, m].set(1e-6 + 0j)
    else:
        key = jax.random.PRNGKey(0)
        noise = 1e-6 * (jax.random.normal(key, base.zeta.shape) + 1j * jax.random.normal(key, base.zeta.shape))
        base.zeta = base.zeta + noise
    return base


def main():
    mstate = initial_condition()
    nsteps = int(2 * 86400 / cfg.dt)
    for step_idx in range(nsteps):
        mstate = timestep.step(mstate)
        if step_idx % int(3 * 3600 / cfg.dt) == 0:
            psi, chi = mstate.zeta, mstate.div
            print(f"step {step_idx}: max|zeta|={jnp.max(jnp.abs(psi)).item():.3e}")


if __name__ == "__main__":
    main()
