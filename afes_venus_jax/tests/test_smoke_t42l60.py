import jax
import jax.numpy as jnp
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import jit_step
from afes_venus_jax.config import DEFAULT_CFG


def test_six_hour_spin():
    cfg = DEFAULT_CFG
    state = zeros_state(cfg)
    key = jax.random.PRNGKey(0)
    state = state.__class__(
        zeta=state.zeta + 1e-6 * jax.random.normal(key, state.zeta.shape),
        div=state.div,
        T=state.T,
        lnps=state.lnps,
    )
    nsteps = int(6 * 3600 / cfg.dt)
    ps0 = jnp.exp(state.lnps).mean()
    for _ in range(nsteps):
        state = jit_step(state, cfg)
    ps1 = jnp.exp(state.lnps).mean()
    rel = jnp.abs(ps1 - ps0) / ps0
    assert rel < 1e-8
    assert jnp.isfinite(state.zeta).all()
