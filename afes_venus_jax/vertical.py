"""Sigma-Lorenz vertical utilities."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from . import config


def sigma_levels(L: int = config.L):
    """Construct Lorenz vertical levels using exponential mapping.

    Returns
    -------
    sigma_full : (L,) array
    sigma_half : (L+1,) array
    z_half : (L+1,) array [m]
    """
    z_half = jnp.linspace(0.0, 120e3, L + 1)
    sigma_half = jnp.exp(-z_half / 15_000.0)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    return sigma_full, sigma_half, z_half


@jax.jit
def pressure_from_sigma(ps: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Compute pressure at full levels given surface pressure."""
    return sigma[:, None, None] * ps


@jax.jit
def hydrostatic_geopotential(T: jnp.ndarray, ps: jnp.ndarray, sigma_full: jnp.ndarray):
    """Approximate hydrostatic geopotential using constant scale height.

    Parameters
    ----------
    T : (L, nlat, nlon) array
        Temperature profile [K].
    ps : (nlat, nlon) array
        Surface pressure [Pa].
    sigma_full : (L,) array
        Full-level sigma.
    """
    # Use simple g*z mapping based on sigma heights
    _, sigma_half, z_half = sigma_levels(len(sigma_full))
    phi_half = config.g * z_half
    phi_full = 0.5 * (phi_half[:-1] + phi_half[1:])
    return phi_full[:, None, None] + jnp.zeros_like(T)
