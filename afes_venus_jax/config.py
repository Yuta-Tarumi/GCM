"""Configuration for AFES-Venus-style spectral core."""
from __future__ import annotations
import jax
import jax.numpy as jnp

# Planetary constants (Venus)
a = 6_051_800.0
"""Planetary radius [m]."""

g = 8.87
"""Gravity [m s^-2]."""

Omega = -2 * jnp.pi / (243.0226 * 86400.0)
"""Rotation rate [s^-1] (retrograde negative)."""

R_gas = 8.314462618 / 0.04401
"""Specific gas constant for CO2 [J kg^-1 K^-1]."""

cp = 1000.0
"""Specific heat at constant pressure [J kg^-1 K^-1]."""

ps_ref = 9.2e6
"""Reference surface pressure [Pa] for sigma mapping."""

# Numerics defaults
Lmax = 42
nlat = 64
nlon = 128
L = 60
dt = 600.0
alpha = 0.5
ra = 0.05
tau_hdiff = 0.1 * 86400.0
order_hdiff = 4


@jax.jit
def spectral_wavenumbers(lmax: int = Lmax) -> jnp.ndarray:
    """Return squared total wavenumber array for triangular truncation.

    Returns
    -------
    k2 : (lmax+1,) array
        k2[l] = l(l+1)/a^2.
    """
    l = jnp.arange(lmax + 1)
    return l * (l + 1) / (a ** 2)
