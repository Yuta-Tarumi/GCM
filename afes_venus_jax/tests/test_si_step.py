import jax
import jax.numpy as jnp
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import jit_step
from afes_venus_jax.config import DEFAULT_CFG


def test_stable_si():
    cfg = DEFAULT_CFG
    state = zeros_state(cfg)
    key = jax.random.PRNGKey(0)
    perturb = 1e-5 * jax.random.normal(key, state.div.shape)
    state = state.__class__(zeta=state.zeta, div=state.div + perturb, T=state.T, lnps=state.lnps)
    for _ in range(10):
        state = jit_step(state, cfg)
    assert jnp.isfinite(state.div).all()
    assert jnp.isfinite(state.lnps).all()
