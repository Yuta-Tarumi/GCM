import dataclasses

import jax
import jax.numpy as jnp

from afes_venus_jax.config import DEFAULT_CFG
from afes_venus_jax.spharm import analysis_grid_to_spec, psi_chi_from_vort_div, uv_from_psi_chi
from afes_venus_jax.state import zeros_state
from afes_venus_jax.timestep import jit_step
from afes_venus_jax.vertical import reference_temperature_profile


def test_fifteen_steps_keep_winds_finite():
    cfg = dataclasses.replace(DEFAULT_CFG, nlat=16, nlon=32, Lmax=8, L=5)
    state = zeros_state(cfg)

    key = jax.random.PRNGKey(0)
    grid_noise = 1e-4 * jax.random.normal(key, (cfg.L, cfg.nlat, cfg.nlon))
    noise_spec = analysis_grid_to_spec(grid_noise, cfg)
    state = state.__class__(zeta=state.zeta + noise_spec, div=state.div, T=state.T, lnps=state.lnps)

    T_profile = reference_temperature_profile(cfg)
    T_grid = jnp.broadcast_to(T_profile[:, None, None], (cfg.L, cfg.nlat, cfg.nlon))
    T_spec = analysis_grid_to_spec(T_grid, cfg)
    state = state.__class__(zeta=state.zeta, div=state.div, T=T_spec, lnps=state.lnps)

    for _ in range(15):
        state = jit_step(state, cfg)

    psi, chi = psi_chi_from_vort_div(state.zeta[0], state.div[0], cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    speed = jnp.sqrt(u**2 + v**2)

    assert jnp.all(jnp.isfinite(speed))
    assert speed.max() < 100.0
