"""T42L60 Venus dry spin-up demo (simplified)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from afes_venus_jax.config import DEFAULT_CFG
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import step_jit


def main():
    cfg = DEFAULT_CFG
    state = zeros_state(cfg)
    # small perturbation initial state
    state = state.__class__(
        state.zeta.at[:, 1, cfg.Lmax].set(1e-6),
        state.div,
        state.T,
        state.lnps,
    )

    def body(carry, _):
        st = carry
        new_state = step_jit(st, cfg)
        return new_state, None

    state, _ = jax.lax.scan(body, state, jnp.arange(10))
    print("Completed 10 steps; max |zeta|", jnp.abs(state.zeta).max())


if __name__ == "__main__":
    main()
