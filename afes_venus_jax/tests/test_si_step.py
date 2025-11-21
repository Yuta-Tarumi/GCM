import jax
import jax.numpy as jnp

import afes_venus_jax.state as state
import afes_venus_jax.timestep as timestep


def test_small_perturbation():
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
