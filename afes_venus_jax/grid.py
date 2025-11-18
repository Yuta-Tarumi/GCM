"""Gaussian grid utilities for pseudo-spectral evaluation."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from . import config


def gaussian_grid(nlat: int | None = None, nlon: int | None = None):
    """Construct a Gaussian grid suitable for T42 pseudo-spectral transforms.

    Parameters
    ----------
    nlat, nlon : int
        Number of latitudes and longitudes.

    Returns
    -------
    lats : (nlat,) ndarray
        Latitudes in radians from south to north.
    lons : (nlon,) ndarray
        Longitudes in radians.
    weights : (nlat,) ndarray
        Gaussian quadrature weights.
    """
    cfg = config.DEFAULT
    if nlat is None:
        nlat = cfg.numerics.nlat
    if nlon is None:
        nlon = cfg.numerics.nlon

    mu, w = leggauss(nlat)
    lats = np.arcsin(mu)
    lons = np.linspace(0.0, 2 * np.pi, nlon, endpoint=False)
    return jnp.asarray(lats), jnp.asarray(lons), jnp.asarray(w)


@jax.jit
def cosine_latitudes(lats: jnp.ndarray) -> jnp.ndarray:
    """Return cos(phi) with clipping for numerical stability."""
    return jnp.clip(jnp.cos(lats), 1e-7, None)


@jax.jit
def area_weights(weights: jnp.ndarray, nlon: int | None = None) -> jnp.ndarray:
    """Compute 2D area weights normalized to integrate to 4π.

    Parameters
    ----------
    weights : (nlat,) array
        Gaussian weights.
    nlon : int
        Number of longitudes.
    """
    if nlon is None:
        nlon = config.DEFAULT.numerics.nlon
    return weights[:, None] * jnp.ones((weights.shape[0], nlon)) * (2 * jnp.pi / nlon)
