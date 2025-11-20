"""Configuration defaults for AFES-Venus-style core."""
from __future__ import annotations

import dataclasses
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class Config:
    """Model configuration and planetary constants.

    Attributes
    ----------
    a: float
        Planetary radius [m].
    g: float
        Gravity [m s^-2].
    Omega: float
        Rotation rate [s^-1].
    R_gas: float
        Specific gas constant [J kg^-1 K^-1].
    cp: float
        Specific heat at constant pressure [J kg^-1 K^-1].
    ps_ref: float
        Reference surface pressure [Pa].
    Lmax: int
        Spectral truncation (triangular).
    nlat: int
        Number of Gaussian latitudes.
    nlon: int
        Number of longitudes.
    L: int
        Number of full sigma levels (Lorenz staggering).
    dt: float
        Time step [s].
    alpha: float
        Semi-implicit off-centering.
    ra: float
        Robert-Asselin filter coefficient.
    tau_hdiff: float
        Hyperdiffusion e-folding timescale at Lmax [s].
    order_hdiff: int
        Hyperdiffusion order.
    """

    a: float = 6_051_800.0
    g: float = 8.87
    Omega: float = -2 * jnp.pi / (243.0226 * 86400.0)
    R_gas: float = 8.314462618 / 0.04401
    cp: float = 1000.0
    ps_ref: float = 9.2e6

    Lmax: int = 42
    nlat: int = 64
    nlon: int = 128
    L: int = 60
    dt: float = 600.0
    alpha: float = 0.5
    ra: float = 0.05
    tau_hdiff: float = 0.1 * 86400.0
    order_hdiff: int = 4


DEFAULT_CFG = Config()
"""Default configuration for Venus T42L60."""
