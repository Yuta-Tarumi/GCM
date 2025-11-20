"""Gaussian grid utilities for T42-like resolutions."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import legendre
from .config import Config


def gaussian_grid(cfg: Config):
    """Construct a Gaussian grid.

    Parameters
    ----------
    cfg: Config
        Model configuration providing ``nlat`` and ``nlon``.

    Returns
    -------
    lats: jnp.ndarray
        Latitudes in radians, shape (nlat,).
    lons: jnp.ndarray
        Longitudes in radians, shape (nlon,).
    weights: jnp.ndarray
        Gaussian quadrature weights, shape (nlat,).
    """
    nlat = cfg.nlat
    nlon = cfg.nlon
    x, w = legendre.leggauss(nlat)
    lats = jnp.arcsin(jnp.asarray(x))
    lons = jnp.linspace(0.0, 2 * jnp.pi, nlon, endpoint=False)
    return lats, lons, jnp.asarray(w)


def quadrature_mean(field: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Compute a weighted mean over the Gaussian grid.

    Parameters
    ----------
    field: array_like
        Field with leading dimensions (..., nlat, nlon).
    weights: array_like
        Quadrature weights of shape (nlat,).
    """
    w = weights[..., None]
    return jnp.sum(field * w, axis=(-2, -1)) / jnp.sum(w)
