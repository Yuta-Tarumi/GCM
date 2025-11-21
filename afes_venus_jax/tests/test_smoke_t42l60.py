import os

import jax.numpy as jnp
import pytest

import afes_venus_jax.state as state
import afes_venus_jax.timestep as timestep


FAST = os.getenv("AFES_VENUS_JAX_FAST_TESTS", "0") == "1"


def test_short_integration():
    if FAST:
        pytest.skip("Skipping smoke integration in fast test mode")
    mstate = state.zeros_state()
    nsteps = 6  # 1 hour with dt=600s
    out_state = timestep.integrate(mstate, nsteps)[0]
    assert jnp.all(jnp.isfinite(out_state.zeta))
    assert jnp.abs(jnp.mean(out_state.lnps)) < 1e-10
