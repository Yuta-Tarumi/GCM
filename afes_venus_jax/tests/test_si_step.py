import os

import jax
import jax.numpy as jnp
import pytest

import afes_venus_jax.state as state
import afes_venus_jax.timestep as timestep


FAST = os.getenv("AFES_VENUS_JAX_FAST_TESTS", "0") == "1"


def test_small_perturbation():
    if FAST:
        pytest.skip("Skipping expensive dynamics in fast test mode")
    mstate = state.zeros_state()
    mstate = mstate.__class__(
        mstate.zeta.at[:, 1, 1].set(1e-8),
        mstate.div,
        mstate.T,
        mstate.lnps,
    )
    out = timestep.step(mstate)
    assert jnp.all(jnp.isfinite(out.zeta))
    assert jnp.all(jnp.isfinite(out.div))
