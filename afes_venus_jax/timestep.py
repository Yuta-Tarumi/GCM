"""One-step time integrator."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from . import tendencies, implicit, diffusion, config


@jax.jit
def step(state):
    """Advance the model state by one leapfrog step."""
    dzeta, ddiv, dT, dlnps = tendencies.nonlinear_tendencies(state)
    ddiv, dT, dlnps = implicit.implicit_correction(ddiv, dT, dlnps, config.alpha)
    # Forward Euler for simplicity
    new_state = state.__class__(
        zeta=state.zeta + config.dt * dzeta,
        div=state.div + config.dt * ddiv,
        T=state.T + config.dt * dT,
        lnps=state.lnps + config.dt * dlnps,
    )
    new_state = diffusion.apply_all(new_state, config.tau_hdiff, config.order_hdiff)
    return new_state


def integrate(state, nsteps: int):
    def body_fn(carry, _):
        new_state = step(carry)
        return new_state, new_state
    final, traj = jax.lax.scan(body_fn, state, None, length=nsteps)
    return final, traj
