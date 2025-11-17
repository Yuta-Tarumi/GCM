import jax
import jax.numpy as jnp
from afes_venus_jax import state, timestep, config


def test_short_run_no_nan():
    s = state.initial_isothermal()
    key = jax.random.PRNGKey(4)
    s = state.ModelState(
        zeta=s.zeta + 1e-6 * jax.random.normal(key, s.zeta.shape),
        div=s.div,
        T=s.T,
        lnps=s.lnps,
    )
    final, _ = timestep.integrate(s, nsteps=3)
    assert jnp.all(jnp.isfinite(final.zeta))
    assert jnp.all(jnp.isfinite(final.lnps))
