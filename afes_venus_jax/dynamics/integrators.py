"""One-step time integrator with physics hooks."""
from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp
from .. import config, diffusion, friction
from ..physics import solar, newtonian
from . import nonlinear, semi_implicit
from ..state import ModelState


@partial(jax.jit, static_argnums=2)
def step(state: ModelState, t: float, cfg: config.ModelConfig | None = None) -> ModelState:
    """Advance the model state by one explicit step."""
    if cfg is None:
        cfg = config.DEFAULT
    dt = cfg.numerics.dt

    dzeta, ddiv, dT, dlnps = nonlinear.nonlinear_tendencies(state, cfg)
    total = ModelState(zeta=dzeta, div=ddiv, T=dT, lnps=dlnps)

    if cfg.physics.enable_solar:
        dT_solar = solar.solar_heating_tendency(t, cfg)
        total = ModelState(
            zeta=total.zeta,
            div=total.div,
            T=total.T + dT_solar,
            lnps=total.lnps,
        )

    if cfg.physics.enable_newtonian:
        dT_newt = newtonian.cooling_tendency(state, cfg)
        total = ModelState(
            zeta=total.zeta,
            div=total.div,
            T=total.T + dT_newt,
            lnps=total.lnps,
        )

    dT_vert = diffusion.vertical_diffusion_temperature(state, cfg)
    total = ModelState(
        zeta=total.zeta,
        div=total.div,
        T=total.T + dT_vert,
        lnps=total.lnps,
    )

    damp_zeta, damp_div = friction.apply_rayleigh_and_sponge(state, cfg)
    total = ModelState(
        zeta=total.zeta + damp_zeta,
        div=total.div + damp_div,
        T=total.T,
        lnps=total.lnps,
    )

    ddiv_si, dT_si, dlnps_si = semi_implicit.apply(total.div, total.T, total.lnps, cfg.numerics.semi_implicit_alpha)
    total = ModelState(zeta=total.zeta, div=ddiv_si, T=dT_si, lnps=dlnps_si)

    new_state = ModelState(
        zeta=state.zeta + dt * total.zeta,
        div=state.div + dt * total.div,
        T=state.T + dt * total.T,
        lnps=state.lnps + dt * total.lnps,
    )

    new_state = diffusion.apply_hyperdiffusion(new_state, cfg)
    return new_state


def integrate(state: ModelState, nsteps: int, cfg: config.ModelConfig | None = None):
    if cfg is None:
        cfg = config.DEFAULT

    def body_fn(carry, step_idx):
        t = step_idx * cfg.numerics.dt
        new_state = step(carry, t, cfg)
        return new_state, new_state

    final, traj = jax.lax.scan(body_fn, state, jnp.arange(nsteps))
    return final, traj
