"""AFES-Venus inspired JAX core (simplified for demonstration)."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from .config import Config, DEFAULT_CFG
from .initial_conditions import superrotating_initial_state, vira_temperature_profile
from .state import ModelState, StateTree, zeros_state
from .timestep import jit_step, step
