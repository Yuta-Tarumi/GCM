"""Vertical grid and hydrostatic helper utilities."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import Config


def sigma_levels(cfg: Config):
    """Construct Lorenz sigma levels using an exponential scale height.

    Returns
    -------
    sigma_half: jnp.ndarray
        Half-level sigma values, shape (L+1,).
    sigma_full: jnp.ndarray
        Full-level sigma values, shape (L,).
    """
    L = cfg.L
    z_half = jnp.linspace(0.0, 120e3, L + 1)
    H_ref = 15_000.0
    sigma_half = jnp.exp(-z_half / H_ref)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    return sigma_half, sigma_full


def reference_temperature_profile(cfg: Config) -> jnp.ndarray:
    """Dry-adiabatic reference profile capped by a minimum temperature."""

    _, sigma_full = sigma_levels(cfg)
    kappa = cfg.R_gas / cfg.cp
    return jnp.maximum(cfg.T_surface_ref * sigma_full**kappa, cfg.T_cap)


def hydrostatic_geopotential(T: jnp.ndarray, lnps: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Integrate hydrostatic balance to obtain geopotential.

    Parameters
    ----------
    T: array_like
        Temperature [K], shape (L, ...).
    lnps: array_like
        Log surface pressure with same trailing dimensions as T.
    cfg: Config
        Configuration with gas constant and gravity.
    """
    sigma_half, sigma_full = sigma_levels(cfg)
    ps = jnp.exp(lnps)
    p_full = sigma_full[:, None, None] * ps
    p_half = sigma_half[:, None, None] * ps
    # Simple trapezoidal integration in pressure coordinates
    dp = p_half[:-1] - p_half[1:]
    Tv = T  # no moisture
    dPhi = cfg.R_gas * Tv * jnp.log(p_half[:-1] / p_half[1:])
    Phi = jnp.cumsum(dPhi[::-1], axis=0)[::-1]
    return Phi
