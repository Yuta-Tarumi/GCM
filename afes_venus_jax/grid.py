"""Gaussian grid utilities for T42-like resolutions."""
from __future__ import annotations

import jax.numpy as jnp
from numpy.polynomial import legendre
from .config import Config


def gaussian_grid(cfg: Config):
    """Construct a true Gaussian latitude grid with uniform longitudes.

    The nodes follow the Gauss–Legendre quadrature abscissas, providing the
    spherical quadrature needed for exact spectral transforms with associated
    Legendre polynomials through total wavenumber ``cfg.Lmax``.
    """

    nlat = cfg.nlat
    nlon = cfg.nlon

    # Gauss–Legendre abscissas/weights on [-1, 1] in terms of μ = sin φ.
    mu, w = legendre.leggauss(nlat)
    lats = jnp.arcsin(mu)

    # Shift longitudes to [0, 2π).
    lons = jnp.linspace(0.0, 2 * jnp.pi, nlon, endpoint=False)

    # Quadrature weights are already appropriate for integrating over μ;
    # they implicitly include the cos φ Jacobian via dμ = cos φ dφ.
    weights = jnp.asarray(w)
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
