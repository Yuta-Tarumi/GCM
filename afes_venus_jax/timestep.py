"""Single-step time integration combining explicit and implicit pieces."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .tendencies import nonlinear_tendencies
from .implicit import apply_semi_implicit, robert_asselin_filter
from .diffusion import apply_diffusion
from .config import Config
from .state import StateTree


def step(state: StateTree, cfg: Config) -> StateTree:
    tend = nonlinear_tendencies(state, cfg)
    zeta_new = state.zeta + cfg.dt * tend["zeta"]
    div_new = state.div + cfg.dt * tend["div"]
    T_new = state.T + cfg.dt * tend["T"]
    lnps_new = state.lnps + cfg.dt * tend["lnps"]

    div_si, T_si, lnps_si = apply_semi_implicit(div_new, T_new, lnps_new, cfg)
    new_state = state.__class__(zeta=zeta_new, div=div_si, T=T_si, lnps=lnps_si)
    new_state = apply_diffusion(new_state, cfg)
    # RA filter using previous state
    zeta_f = robert_asselin_filter(new_state.zeta, state.zeta, cfg.ra)
    div_f = robert_asselin_filter(new_state.div, state.div, cfg.ra)
    T_f = robert_asselin_filter(new_state.T, state.T, cfg.ra)
    lnps_f = robert_asselin_filter(new_state.lnps, state.lnps, cfg.ra)
    sanitized = state.__class__(
        zeta=jnp.nan_to_num(zeta_f),
        div=jnp.nan_to_num(div_f),
        T=jnp.nan_to_num(T_f),
        lnps=jnp.nan_to_num(lnps_f),
    )
    return sanitized


jit_step = jax.jit(step, static_argnames=("cfg",))
