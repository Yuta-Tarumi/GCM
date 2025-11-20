"""Single-step time integration combining explicit and implicit pieces."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .tendencies import nonlinear_tendencies
from .implicit import apply_semi_implicit, robert_asselin_filter
from .diffusion import apply_diffusion
from .config import Config
from .state import StateTree
from .spharm import psi_chi_from_vort_div, uv_from_psi_chi


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

    # Enforce a simple CFL cap by scaling vorticity/divergence if the implied
    # wind speed would step more than half a grid box in one time step. This
    # prevents the advective update from blowing up in the early spin-up stages
    # when the flow is initialised from random noise.
    psi, chi = psi_chi_from_vort_div(zeta_f, div_f, cfg)
    u, v = uv_from_psi_chi(psi, chi, cfg)
    speed_max = jnp.max(jnp.sqrt(u**2 + v**2))
    dx = cfg.a * (2 * jnp.pi / cfg.nlon)
    dy = cfg.a * (jnp.pi / cfg.nlat)
    max_speed = 0.5 * jnp.minimum(dx, dy) / cfg.dt
    scale = jnp.minimum(1.0, max_speed / jnp.maximum(speed_max, 1e-12))

    sanitized = state.__class__(
        zeta=jnp.nan_to_num(zeta_f * scale),
        div=jnp.nan_to_num(div_f * scale),
        T=jnp.nan_to_num(T_f),
        lnps=jnp.nan_to_num(lnps_f),
    )
    return sanitized


jit_step = jax.jit(step, static_argnames=("cfg",))
