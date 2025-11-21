"""Configuration for AFES-Venus-style spectral core.

The module exposes lightweight, module-level parameters that mirror the
AFES-Venus T42L60 setup. A dedicated :mod:`afes_venus_jax.t42l60_config`
module holds the full collection of AFES-like numerical tunings; the
values below simply forward to that preset so the rest of the codebase
can keep importing ``afes_venus_jax.config`` without refactoring.
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

from afes_venus_jax import t42l60_config

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


def _float_env(name: str, default: float, allow_none: bool = False) -> float | None:
    """Read a floating-point value from the environment if provided."""

    raw = os.getenv(name)
    if raw is None:
        return default
    if allow_none and raw == "":
        return None

    try:
        return float(raw)
    except ValueError:
        return default


# Numerical defaults (T42L60-focused)
_default_Lmax = t42l60_config.LMAX  # spectral truncation (triangular T42)
_default_nlat = t42l60_config.NLAT
_default_nlon = t42l60_config.NLON
_default_L = t42l60_config.LLEVELS  # vertical full levels (Lorenz)
dt = t42l60_config.DT  # time step [s]
alpha = t42l60_config.SI_ALPHA  # SI off-centering
ra = t42l60_config.RA_COEFF  # Robert–Asselin/RAW filter strength
ra_williams_factor = t42l60_config.RA_WILLIAMS_FACTOR  # Williams correction factor for RAW filter
time_filter = os.getenv("AFES_VENUS_JAX_TIME_FILTER", t42l60_config.TIME_FILTER).lower()
use_raw_filter = time_filter == "raw"
use_semi_lagrangian_advection = _bool_env("AFES_VENUS_JAX_USE_SEMI_LAGRANGIAN_ADVECTION", True)
# Apply weak divergence damping by default to mirror AFES-Venus production runs.
tau_div_damp = _float_env("AFES_VENUS_JAX_TAU_DIV_DAMP", t42l60_config.TAU_DIV_DAMP, allow_none=True)
order_div_damp = 1  # Laplacian power for divergence damping (1 = ∇², 2 = ∇⁴, ...)
tau_hdiff = t42l60_config.TAU_HYPERDIFF  # e-folding time for hyperdiffusion [s]
order_hdiff = 4
nu4_hdiff = t42l60_config.NU4_HYPERDIFF
bottom_rayleigh_tau = t42l60_config.TAU_BOTTOM
bottom_rayleigh_ramp = t42l60_config.BOTTOM_LAYERS_RAMP
vertical_diffusion_kz = t42l60_config.KZ
sponge_config = t42l60_config.SPONGE
tau_rad_profile = t42l60_config.TAU_RAD_PROFILE
T_eq_profile = t42l60_config.T_EQ_PROFILE


Lmax = _int_env("AFES_VENUS_JAX_LMAX", _default_Lmax)
s2fft_sampling = os.getenv("AFES_VENUS_JAX_S2FFT_SAMPLING", "mw").lower()
use_s2fft = os.getenv("AFES_VENUS_JAX_USE_S2FFT", "true").lower() not in {"false", "0", ""}

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
