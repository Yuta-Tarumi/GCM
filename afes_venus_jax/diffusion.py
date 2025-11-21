"""Spectral hyperdiffusion."""
from __future__ import annotations

import jax.numpy as jnp

from .config import ModelConfig
from .spharm import lap_spec


def hyperdiffuse(field_spec: jnp.ndarray, cfg: ModelConfig) -> jnp.ndarray:
    """Apply ∇⁴ hyperdiffusion to a spectral field."""
    l = jnp.arange(cfg.Lmax + 1, dtype=jnp.float64)
    eig = (l * (l + 1) / (cfg.a ** 2)) ** cfg.order_hdiff
    eig = eig[:, None]
    eig_max = (cfg.Lmax * (cfg.Lmax + 1) / (cfg.a ** 2)) ** cfg.order_hdiff
    nu = 1.0 / (cfg.tau_hdiff * eig_max)
    return field_spec - cfg.dt * nu * eig * field_spec
