"""Gaussian grid construction for spectral transforms.

Provides a Gaussian (Gauss–Legendre) latitude grid with uniform longitude
spacing suitable for pseudo-spectral evaluation at T42 resolution.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

import afes_venus_jax.config as cfg


def gaussian_grid(nlat: int = cfg.nlat, nlon: int = cfg.nlon):
    """Construct Gaussian grid.

    Returns
    -------
    lats : ndarray, shape (nlat,)
        Latitudes in radians (South to North).
    lons : ndarray, shape (nlon,)
        Longitudes in radians [0, 2π).
    weights : ndarray, shape (nlat,)
        Gaussian quadrature weights for integration over cosφ.
    """
    mu, w = leggauss(nlat)
    lats = np.arcsin(mu)
    lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)
    return lats, lons, w


def grid_arrays(nlat: int = cfg.nlat, nlon: int = cfg.nlon):
    """Return broadcast latitude/longitude arrays for grid computations."""
    lats, lons, w = gaussian_grid(nlat, nlon)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return jnp.array(lat2d), jnp.array(lon2d), jnp.array(w)
