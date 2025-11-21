"""Gaussian grid utilities."""
from __future__ import annotations

import functools
import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from .config import ModelConfig


def gaussian_grid(cfg: ModelConfig):
    """Construct Gaussian grid.

    Parameters
    ----------
    cfg : ModelConfig
        Configuration with nlat, nlon.

    Returns
    -------
    lats : np.ndarray
        Latitudes [rad], shape (nlat,).
    lons : np.ndarray
        Longitudes [rad], shape (nlon,).
    weights : np.ndarray
        Gaussian quadrature weights for integration over mu=cos(theta), shape (nlat,).
    """
    mu, w = leggauss(cfg.nlat)
    lats = np.arcsin(mu)
    lons = np.linspace(0.0, 2 * np.pi, cfg.nlon, endpoint=False)
    return lats, lons, w


@functools.lru_cache(None)
def quadrature_weights(nlat: int, nlon: int):
    """Return tensor of quadrature weights for mu,phi integration."""
    mu, w = leggauss(nlat)
    dphi = 2 * np.pi / nlon
    return jnp.asarray(w) * dphi
