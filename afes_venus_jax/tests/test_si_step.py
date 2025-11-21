import jax
import jax.numpy as jnp

from afes_venus_jax.config import ModelConfig
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import step_jit


def test_semi_implicit_stability():
    cfg = ModelConfig(Lmax=6, nlat=16, nlon=32, L=4)
    state = zeros_state(cfg)
    # impose tiny divergence perturbation
    state = state.__class__(
        state.zeta,
        state.div.at[:, 1, cfg.Lmax].set(1e-6),
        state.T,
        state.lnps,
    )

    def body(carry, _):
        st = carry
        new_state = step_jit(st, cfg)
        return new_state, None

    state, _ = jax.lax.scan(body, state, jnp.arange(5))
    assert jnp.all(jnp.isfinite(jnp.abs(state.div)))
