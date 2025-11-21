import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.state as state
import afes_venus_jax.timestep as timestep


def test_robert_asselin_damps_computational_mode():
    # Build minimal states with a two-time-level oscillation (+1, -1).
    prev = state.ModelState(
        zeta=jnp.full((1, 2, 2), -1.0 + 0.0j),
        div=jnp.full((1, 2, 2), -1.0 + 0.0j),
        T=jnp.full((1, 2, 2), -1.0 + 0.0j),
        lnps=jnp.full((2, 2), -1.0 + 0.0j),
    )
    curr = state.ModelState(
        zeta=jnp.full((1, 2, 2), 1.0 + 0.0j),
        div=jnp.full((1, 2, 2), 1.0 + 0.0j),
        T=jnp.full((1, 2, 2), 1.0 + 0.0j),
        lnps=jnp.full((2, 2), 1.0 + 0.0j),
    )
    new = state.ModelState(
        zeta=jnp.full((1, 2, 2), -1.0 + 0.0j),
        div=jnp.full((1, 2, 2), -1.0 + 0.0j),
        T=jnp.full((1, 2, 2), -1.0 + 0.0j),
        lnps=jnp.full((2, 2), -1.0 + 0.0j),
    )

    filt_curr, filt_new = timestep._robert_asselin(prev, curr, new)

    expected = 1.0 + cfg.ra * (-1.0 - 2.0 * 1.0 - 1.0)
    assert jnp.allclose(filt_curr.zeta.real, expected)
    assert jnp.allclose(filt_curr.div.real, expected)
    assert jnp.allclose(filt_curr.T.real, expected)
    assert jnp.allclose(filt_curr.lnps.real, expected)

    # The newly stepped state should pass through untouched.
    assert jnp.array_equal(filt_new.zeta, new.zeta)
    assert jnp.array_equal(filt_new.div, new.div)
    assert jnp.array_equal(filt_new.T, new.T)
    assert jnp.array_equal(filt_new.lnps, new.lnps)
