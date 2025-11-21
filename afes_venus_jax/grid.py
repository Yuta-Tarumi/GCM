"""Gaussian grid construction for spectral transforms.

Provides a Gaussian (Gauss–Legendre) latitude grid with uniform longitude
spacing suitable for pseudo-spectral evaluation at T42 resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

import afes_venus_jax.config as cfg


_DATA_DIR = Path(__file__).parent / "data"


def _default_cache_path(nlat: int, nlon: int) -> Path:
    return _DATA_DIR / f"gaussian_grid_nlat{nlat}_nlon{nlon}.npz"


def gaussian_grid(
    nlat: int = cfg.nlat,
    nlon: int = cfg.nlon,
    *,
    cache: bool = False,
    cache_path: str | Path | None = None,
):
    """Construct Gaussian grid.

    Returns
    -------
    lats : ndarray, shape (nlat,)
        Latitudes in radians (South to North).
    lons : ndarray, shape (nlon,)
        Longitudes in radians [0, 2π).
    weights : ndarray, shape (nlat,)
        Gaussian quadrature weights for integration over cosφ.

    Notes
    -----
    If ``cache`` is True or ``cache_path`` is provided, the grid will be loaded
    from the specified ``.npz`` file when present; otherwise, the freshly
    constructed grid is written to that path for reuse.
    """
    if cache or cache_path is not None:
        path = _default_cache_path(nlat, nlon) if cache_path is None else Path(cache_path)
        if path.is_file():
            data = np.load(path)
            return data["lats"], data["lons"], data["weights"]

    mu, w = leggauss(nlat)
    lats = np.arcsin(mu)
    lons = np.linspace(0, 2 * np.pi, nlon, endpoint=False)

    if cache or cache_path is not None:
        path = _default_cache_path(nlat, nlon) if cache_path is None else Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, lats=lats, lons=lons, weights=w)

    return lats, lons, w


def grid_arrays(nlat: int = cfg.nlat, nlon: int = cfg.nlon):
    """Return broadcast latitude/longitude arrays for grid computations."""
    lats, lons, w = gaussian_grid(nlat, nlon)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return jnp.array(lat2d), jnp.array(lon2d), jnp.array(w)
