"""Sigma-coordinate utilities."""
from __future__ import annotations
import jax.numpy as jnp
from . import config


def sigma_levels(nlev: int | None = None):
    cfg = config.DEFAULT
    if nlev is None:
        nlev = cfg.numerics.nlev
    sigma_half = jnp.linspace(1.0, 0.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    return sigma_full, sigma_half


def level_heights(nlev: int | None = None) -> jnp.ndarray:
    sigma_full, _ = sigma_levels(nlev)
    scale_height = 7500.0
    sigma_clip = jnp.clip(sigma_full, 1e-4, 1.0)
    return -scale_height * jnp.log(sigma_clip)


def sigma_mask_above_height(height_m: float, nlev: int | None = None) -> jnp.ndarray:
    heights = level_heights(nlev)
    return (heights >= height_m).astype(jnp.float64)
