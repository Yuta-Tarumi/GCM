"""Vertical discretisation utilities for σ–Lorenz levels."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import afes_venus_jax.config as cfg


# Reference scale height and model top used for the sigma–altitude mapping.
H_REF = 15_000.0
Z_TOP = 120_000.0


def sigma_levels(L: int = cfg.L):
    """Construct Lorenz levels in sigma coordinate.

    Returns
    -------
    sigma_full : ndarray, shape (L,)
    sigma_half : ndarray, shape (L+1,)
    """
    z_half = np.linspace(0.0, Z_TOP, L + 1)
    sigma_half = np.exp(-z_half / H_REF)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    return jnp.array(sigma_full), jnp.array(sigma_half)


def level_altitudes(L: int = cfg.L):
    """Return altitude (full and half) consistent with :func:`sigma_levels`."""

    z_half = np.linspace(0.0, Z_TOP, L + 1)
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return jnp.array(z_full), jnp.array(z_half)


def hydrostatic_geopotential(T: jnp.ndarray, ps: jnp.ndarray, sigma_half: jnp.ndarray):
    """Integrate hydrostatic balance to obtain geopotential on full levels."""
    R = cfg.R_gas

    def integrate_column(Tcol, p_surf):
        p_half = sigma_half * p_surf
        dp = p_half[:-1] - p_half[1:]
        Tv = Tcol  # dry
        Phi = jnp.cumsum((R * Tv) * dp[::-1] / p_half[:-1][::-1])
        return Phi[::-1]

    # Vectorise over latitude and longitude while keeping the vertical
    # dimension intact.
    integrate_lon = jax.vmap(integrate_column, in_axes=(1, 0), out_axes=1)
    integrate_lat = jax.vmap(integrate_lon, in_axes=(1, 1), out_axes=1)
    return integrate_lat(T, ps)
