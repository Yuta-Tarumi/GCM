"""Single time step driver."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import ModelConfig
from .diffusion import hyperdiffuse
from .implicit import semi_implicit_update
from .state import ModelState
from .tendencies import compute_tendencies


def step(state: ModelState, cfg: ModelConfig) -> ModelState:
    tzeta, tdiv, tT, tlnps = compute_tendencies(state, cfg)
    zeta_new, div_new, T_new, lnps_new = semi_implicit_update(state, (tzeta, tdiv, tT, tlnps), cfg)

    zeta_new = hyperdiffuse(zeta_new, cfg)
    div_new = hyperdiffuse(div_new, cfg)
    T_new = hyperdiffuse(T_new, cfg)
    lnps_new = hyperdiffuse(lnps_new, cfg)

    # Robert-Asselin filter (leapfrog surrogate with a single previous state stored in lnps_new imaginary part)
    zeta_filt = zeta_new * (1 - cfg.ra)
    div_filt = div_new * (1 - cfg.ra)
    T_filt = T_new * (1 - cfg.ra)
    lnps_filt = lnps_new * (1 - cfg.ra)

    return ModelState(zeta_filt, div_filt, T_filt, lnps_filt)


step_jit = jax.jit(step, static_argnums=1)
