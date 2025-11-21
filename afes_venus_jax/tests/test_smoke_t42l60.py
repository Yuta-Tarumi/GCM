import jax.numpy as jnp

import afes_venus_jax.state as state
import afes_venus_jax.timestep as timestep


def test_short_integration():
    mstate = state.zeros_state()
    nsteps = 6  # 1 hour with dt=600s
    out_state = timestep.integrate(mstate, nsteps)[0]
    assert jnp.all(jnp.isfinite(out_state.zeta))
    assert jnp.abs(jnp.mean(out_state.lnps)) < 1e-10
