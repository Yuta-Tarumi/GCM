import jax.numpy as jnp
from afes_venus_jax.config import DEFAULT_CFG
from afes_venus_jax.state import zeros_state
from afes_venus_jax.diffusion import apply_diffusion, diffusion_operator


def test_decay_rate():
    cfg = DEFAULT_CFG
    state = zeros_state(cfg)
    mode = jnp.zeros_like(state.zeta)
    mode = mode.at[0, 0, 1].set(1.0)
    state = state.__class__(zeta=mode, div=mode, T=mode, lnps=state.lnps)
    coeff = diffusion_operator(cfg, cfg.nlat, cfg.nlon)[0, 1]
    stepped = apply_diffusion(state, cfg)
    factor = stepped.zeta[0, 0, 1].real
    expected = 1.0 + coeff * cfg.dt
    assert jnp.isclose(factor, expected)
