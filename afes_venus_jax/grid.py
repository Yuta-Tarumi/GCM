"""Spectral grids for spherical-harmonic transforms.

Provides the legacy Gaussian (Gauss–Legendre) grid and a backend-aware
spectral grid that switches to the S2FFT equiangular sampling when
``AFES_VENUS_JAX_USE_S2FFT`` is enabled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

import afes_venus_jax.config as cfg

try:
    from s2fft.sampling import s2_samples

    _HAS_S2FFT = True
except Exception:  # pragma: no cover - optional acceleration dependency
    _HAS_S2FFT = False


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


def spectral_grid(
    nlat: int | None = None,
    nlon: int | None = None,
    *,
    lmax: int = cfg.Lmax,
    cache: bool = False,
    cache_path: str | Path | None = None,
):
    """Return the active spectral grid depending on the configured backend."""

    if cfg.use_s2fft and _HAS_S2FFT:
        bandlimit = lmax + 1
        thetas = s2_samples.thetas(L=bandlimit, sampling=cfg.s2fft_sampling)
        lats = np.pi / 2 - thetas  # convert colatitude to geodetic latitude
        lons = s2_samples.phis_equiang(L=bandlimit, sampling=cfg.s2fft_sampling)
        weights = np.ones_like(lats) / lats.size
        return lats, lons, weights

    # Fall back to the Gaussian quadrature grid used by the reference setup.
    return gaussian_grid(
        cfg.nlat if nlat is None else nlat,
        cfg.nlon if nlon is None else nlon,
        cache=cache,
        cache_path=cache_path,
    )


def grid_arrays(nlat: int = cfg.nlat, nlon: int = cfg.nlon):
    """Return broadcast latitude/longitude arrays for grid computations."""
    lats, lons, w = spectral_grid(nlat, nlon)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return jnp.array(lat2d), jnp.array(lon2d), jnp.array(w)
