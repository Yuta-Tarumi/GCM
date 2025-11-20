"""Gaussian grid utilities for T42-like resolutions."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import legendre
from .config import Config


def gaussian_grid(cfg: Config):
    """Construct an evenly spaced latitude-longitude grid.

    The spectral operators in :mod:`afes_venus_jax.spharm` assume uniform grid
    spacing and periodicity when using FFTs.  Using an equispaced latitude
    discretisation keeps the grid-consistent with the transforms and avoids the
    large truncation errors that arise when pairing FFTs with nonuniform
    Gaussian latitudes.

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
        Uniform quadrature weights, shape (nlat,).
    """
    nlat = cfg.nlat
    nlon = cfg.nlon
    lats = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, nlat, endpoint=False)
    lons = jnp.linspace(0.0, 2 * jnp.pi, nlon, endpoint=False)
    weights = jnp.ones_like(lats)
    return lats, lons, weights


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
