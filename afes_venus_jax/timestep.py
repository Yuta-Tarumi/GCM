"""Single time-step driver."""
from __future__ import annotations

import jax
import jax.numpy as jnp

import afes_venus_jax.config as cfg
import afes_venus_jax.state as state
import afes_venus_jax.tendencies as tend
import afes_venus_jax.implicit as implicit
import afes_venus_jax.diffusion as diffusion


@jax.jit
def step(mstate: state.ModelState):
    zeta_t, div_t, T_t, lnps_t = tend.compute_nonlinear_tendencies(mstate)
    new_state = implicit.semi_implicit_step(mstate, (zeta_t, div_t, T_t, lnps_t))
    new_state = diffusion.apply_diffusion(new_state)
    # Robert–Asselin filter (weak form)
    zeta = (1 - cfg.ra) * new_state.zeta + cfg.ra * mstate.zeta
    div = (1 - cfg.ra) * new_state.div + cfg.ra * mstate.div
    T = (1 - cfg.ra) * new_state.T + cfg.ra * mstate.T
    lnps = (1 - cfg.ra) * new_state.lnps + cfg.ra * mstate.lnps
    return state.ModelState(zeta, div, T, lnps)


def integrate(initial: state.ModelState, nsteps: int):
    def _step(carry, _):
        new_state = step(carry)
        return new_state, new_state
    return jax.lax.scan(_step, initial, None, length=nsteps)
