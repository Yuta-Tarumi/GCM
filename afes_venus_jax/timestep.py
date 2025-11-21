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
def step(mstate: state.ModelState, time_seconds: float = 0.0):
    zeta_t, div_t, T_t, lnps_t = tend.compute_nonlinear_tendencies(mstate, time_seconds=time_seconds)
    new_state = implicit.semi_implicit_step(mstate, (zeta_t, div_t, T_t, lnps_t))
    return diffusion.apply_diffusion(new_state)


def integrate(initial: state.ModelState, nsteps: int):
    def _step(carry, step_idx):
        prev_state, curr_state = carry
        time_seconds = step_idx * cfg.dt
        raw_new_state = step(curr_state, time_seconds=time_seconds)

        if cfg.use_raw_filter:
            filt_curr, filt_new = _robert_asselin_williams(prev_state, curr_state, raw_new_state)
            return (filt_curr, filt_new), filt_new

        filt_new = _robert_asselin(curr_state, raw_new_state)
        return (curr_state, filt_new), filt_new

    step_indices = jnp.arange(nsteps)
    carry_out, states = jax.lax.scan(_step, (initial, initial), step_indices)
    return carry_out[1], states


def _robert_asselin(previous: state.ModelState, new_state: state.ModelState):
    zeta = (1 - cfg.ra) * new_state.zeta + cfg.ra * previous.zeta
    div = (1 - cfg.ra) * new_state.div + cfg.ra * previous.div
    T = (1 - cfg.ra) * new_state.T + cfg.ra * previous.T
    lnps = (1 - cfg.ra) * new_state.lnps + cfg.ra * previous.lnps
    return state.ModelState(zeta, div, T, lnps)


def _robert_asselin_williams(
    prev_state: state.ModelState, curr_state: state.ModelState, new_state: state.ModelState
):
    eps = cfg.ra
    gamma = cfg.ra_williams_factor

    filt_curr = curr_state.__class__(
        curr_state.zeta + eps * (prev_state.zeta - 2 * curr_state.zeta + new_state.zeta),
        curr_state.div + eps * (prev_state.div - 2 * curr_state.div + new_state.div),
        curr_state.T + eps * (prev_state.T - 2 * curr_state.T + new_state.T),
        curr_state.lnps + eps * (prev_state.lnps - 2 * curr_state.lnps + new_state.lnps),
    )

    filt_new = new_state.__class__(
        new_state.zeta + 0.5 * gamma * (filt_curr.zeta - curr_state.zeta),
        new_state.div + 0.5 * gamma * (filt_curr.div - curr_state.div),
        new_state.T + 0.5 * gamma * (filt_curr.T - curr_state.T),
        new_state.lnps + 0.5 * gamma * (filt_curr.lnps - curr_state.lnps),
    )

    return filt_curr, filt_new
