"""Configuration for AFES-Venus-style spectral core.

Defines planetary constants and numerical defaults used across the
package. Values follow the Venus setup and T42L60 reference grid.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp

try:
    from s2fft.sampling import s2_samples

    _s2fft_available = True
except Exception:  # pragma: no cover - optional acceleration dependency
    s2_samples = None
    _s2fft_available = False

# Planetary constants (SI units)
a = 6_051_800.0  # planetary radius [m]
g = 8.87  # gravity [m s^-2]
Omega = -2 * jnp.pi / (243.0226 * 86400.0)  # rotation rate [s^-1]
R_gas = 8.314462618 / 0.04401  # CO2 specific gas constant [J kg^-1 K^-1]
cp = 1000.0  # heat capacity [J kg^-1 K^-1]
ps_ref = 9.2e6  # reference surface pressure [Pa]

def _int_env(name: str, default: int) -> int:
    """Read an integer from the environment if provided."""

    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean from the environment if provided."""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() not in {"false", "0", ""}


# Numerical defaults
_default_Lmax = 42  # spectral truncation (triangular T42)
_default_nlat = 64
_default_nlon = 128
_default_L = 60  # vertical full levels (Lorenz)
dt = 600.0  # time step [s]
alpha = 0.5  # SI off-centering
ra = 0.05  # Robert–Asselin/RAW filter strength
ra_williams_factor = 0.53  # Williams correction factor for RAW filter
use_raw_filter = _bool_env("AFES_VENUS_JAX_USE_RAW_FILTER", True)  # enable the higher-order RAW time filter
use_semi_lagrangian_advection = _bool_env("AFES_VENUS_JAX_USE_SEMI_LAGRANGIAN_ADVECTION", False)
tau_div_damp = None  # optional scale-selective divergence damping timescale [s]; disabled when None
order_div_damp = 1  # Laplacian power for divergence damping (1 = ∇², 2 = ∇⁴, ...)
tau_hdiff = 0.1 * 86400.0  # e-folding time for hyperdiffusion [s]
order_hdiff = 4


Lmax = _int_env("AFES_VENUS_JAX_LMAX", _default_Lmax)
s2fft_sampling = os.getenv("AFES_VENUS_JAX_S2FFT_SAMPLING", "mw").lower()
use_s2fft = os.getenv("AFES_VENUS_JAX_USE_S2FFT", "false").lower() not in {"false", "0", ""}

if use_s2fft and _s2fft_available:
    # S2FFT expects band-limit L such that ell < L.
    _bandlimit = Lmax + 1
    nlat = int(s2_samples.ntheta(L=_bandlimit, sampling=s2fft_sampling))
    nlon = int(s2_samples.nphi_equiang(L=_bandlimit, sampling=s2fft_sampling))
    spectral_backend = "s2fft"
else:
    use_s2fft = False
    spectral_backend = "quadrature"
    nlat = _int_env("AFES_VENUS_JAX_NLAT", _default_nlat)
    nlon = _int_env("AFES_VENUS_JAX_NLON", _default_nlon)
L = _int_env("AFES_VENUS_JAX_L", _default_L)

# Disable 64-bit by default to reduce GPU memory pressure. Users can opt-in
# via AFES_VENUS_JAX_ENABLE_X64=True when higher precision is required.
jax_enable_x64 = os.getenv("AFES_VENUS_JAX_ENABLE_X64", "False").lower() != "false"

if jax_enable_x64:
    jax.config.update("jax_enable_x64", True)


def spectral_shape(lmax: int = Lmax) -> tuple[int, int]:
    """Return the spectral array shape for (ell, m) indices."""
    return lmax + 1, lmax + 1
