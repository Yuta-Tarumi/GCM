import jax
import jax.numpy as jnp
from afes_venus_jax import state, config
from afes_venus_jax.dynamics import integrators


def test_small_step_stability():
    s = state.initial_isothermal()
    key = jax.random.PRNGKey(3)
    s = state.ModelState(
        zeta=s.zeta + 1e-5 * jax.random.normal(key, s.zeta.shape),
        div=s.div,
        T=s.T,
        lnps=s.lnps,
    )
    final, _ = integrators.integrate(s, nsteps=5, cfg=config.DEFAULT)
    assert jnp.all(jnp.isfinite(final.zeta))
    assert jnp.all(jnp.isfinite(final.T))
