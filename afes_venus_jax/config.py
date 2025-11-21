"""Configuration for AFES-Venus-style spectral core.

Defines planetary constants and numerical defaults used across the
package. Values follow the Venus setup and T42L60 reference grid.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp

# Planetary constants (SI units)
a = 6_051_800.0  # planetary radius [m]
g = 8.87  # gravity [m s^-2]
Omega = -2 * jnp.pi / (243.0226 * 86400.0)  # rotation rate [s^-1]
R_gas = 8.314462618 / 0.04401  # CO2 specific gas constant [J kg^-1 K^-1]
cp = 1000.0  # heat capacity [J kg^-1 K^-1]
ps_ref = 9.2e6  # reference surface pressure [Pa]

# Numerical defaults
_default_Lmax = 42  # spectral truncation (triangular T42)
_default_nlat = 64
_default_nlon = 128
_default_L = 60  # vertical full levels (Lorenz)
dt = 600.0  # time step [s]
alpha = 0.5  # SI off-centering
ra = 0.05  # Robert–Asselin coefficient
tau_hdiff = 0.1 * 86400.0  # e-folding time for hyperdiffusion [s]
order_hdiff = 4


def _int_env(name: str, default: int) -> int:
    """Read an integer from the environment if provided."""

    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


Lmax = _int_env("AFES_VENUS_JAX_LMAX", _default_Lmax)
nlat = _int_env("AFES_VENUS_JAX_NLAT", _default_nlat)
nlon = _int_env("AFES_VENUS_JAX_NLON", _default_nlon)
L = _int_env("AFES_VENUS_JAX_L", _default_L)

jax_enable_x64 = os.getenv("AFES_VENUS_JAX_ENABLE_X64", "True").lower() != "false"

if jax_enable_x64:
    jax.config.update("jax_enable_x64", True)


def spectral_shape(lmax: int = Lmax) -> tuple[int, int]:
    """Return the spectral array shape for (ell, m) indices."""
    return lmax + 1, lmax + 1
