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
    T_surface_ref: float
        Reference surface temperature used for Newtonian cooling [K].
    T_cap: float
        Minimum temperature cap for the reference profile [K].
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
    tau_newtonian: float
        Newtonian cooling timescale [s].
    nu_vert: float
        Vertical eddy diffusivity/viscosity [m^2 s^-1].
    tau_rayleigh_surface: float
        Rayleigh friction e-folding time applied at the lowest level [s].
    sponge_start_alt: float
        Altitude [m] where the upper-level sponge begins (eddies only).
    tau_sponge_top: float
        E-folding time [s] for eddy damping at the model top.
    sponge_exponent: float
        Controls how quickly the sponge ramps up with height (>=1).
    solar_heating_rate: float
        Peak shortwave heating rate [K s^-1].
    solar_heating_peak_sigma: float
        Sigma level of peak shortwave heating.
    solar_heating_width: float
        Gaussian width of the shortwave heating profile in sigma.
    solar_diurnal_contrast: float
        Fractional day-night modulation of shortwave heating (0=uniform,
        1=zero heating at midnight, scaled by cosine of solar zenith).
    subsolar_longitude: float
        Subsolar longitude [rad] where the diurnal heating maximum is centered.
    """

    a: float = 6_051_800.0
    g: float = 8.87
    Omega: float = -2 * jnp.pi / (243.0226 * 86400.0)
    R_gas: float = 8.314462618 / 0.04401
    cp: float = 1000.0
    ps_ref: float = 9.2e6
    T_surface_ref: float = 735.0
    T_cap: float = 170.0

    Lmax: int = 42
    nlat: int = 64
    nlon: int = 128
    L: int = 60
    dt: float = 600.0
    alpha: float = 0.5
    ra: float = 0.05
    tau_hdiff: float = 0.1 * 86400.0
    order_hdiff: int = 2
    tau_newtonian: float = 50.0 * 86400.0
    nu_vert: float = 0.0015
    tau_rayleigh_surface: float = 0.5 * 86400.0
    sponge_start_alt: float = 80e3
    tau_sponge_top: float = 0.1 * 86400.0
    sponge_exponent: float = 2.0
    solar_heating_rate: float = 2.0 / 86400.0
    solar_heating_peak_sigma: float = 0.1
    solar_heating_width: float = 0.08
    solar_diurnal_contrast: float = 1.0
    subsolar_longitude: float = 0.0


DEFAULT_CFG = Config()
"""Default configuration for Venus T42L60."""
