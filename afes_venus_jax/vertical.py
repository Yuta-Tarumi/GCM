"""Vertical coordinate utilities for sigma-Lorenz staggering."""
from __future__ import annotations

import jax.numpy as jnp

from .config import ModelConfig


def sigma_levels(cfg: ModelConfig):
    """Return sigma half/full level arrays.

    Half levels are constructed from an exponential mapping intended to mimic a
    ~15 km scale height. Full levels are midpoints of the half levels.
    """
    z_half = jnp.linspace(0.0, 120_000.0, cfg.L + 1)
    sigma_half = jnp.exp(-z_half / 15_000.0)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    return sigma_half, sigma_full


def hydrostatic_geopotential(T: jnp.ndarray, ps: jnp.ndarray, cfg: ModelConfig):
    """Integrate hydrostatic balance to derive geopotential on full levels.

    Parameters
    ----------
    T : array
        Temperature [K], shape (..., L).
    ps : array
        Surface pressure [Pa], shape (...,).
    cfg : ModelConfig
        Configuration with R_gas and g.

    Returns
    -------
    phi : array
        Geopotential [m^2 s^-2] at full levels, shape (..., L).
    """
    sigma_half, sigma_full = sigma_levels(cfg)
    # simple hydrostatic integration using mid-layer pressures
    p_half = ps[..., None] * sigma_half
    p_full = ps[..., None] * sigma_full
    # thickness between half levels
    dp = p_half[..., :-1] - p_half[..., 1:]
    # hypsometric equation
    phi = jnp.cumsum(cfg.R_gas * T * dp / p_full / cfg.g, axis=-1)
    return phi
