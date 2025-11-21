"""Spherical-harmonic utilities using explicit quadrature.

These routines provide simple spectral analysis/synthesis compatible with
JAX. They are not optimised like S2FFT but are fully differentiable and
compatible with JIT for modest truncations such as T42.
"""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import lpmv, gammaln

import afes_venus_jax.config as cfg
import afes_venus_jax.grid as grid


@functools.lru_cache(None)
def _precompute_basis(nlat: int, nlon: int, lmax: int):
    lats, lons, w = grid.gaussian_grid(nlat, nlon)
    lon2d, lat2d = np.meshgrid(lons, lats)
    mu = np.sin(lat2d)
    basis = np.zeros((lmax + 1, lmax + 1, nlat, nlon), dtype=np.complex128)
    for ell in range(lmax + 1):
        for m in range(ell + 1):
            norm = jnp.exp(
                0.5
                * (jnp.log(2 * ell + 1) - jnp.log(4 * jnp.pi) + gammaln(ell - m + 1) - gammaln(ell + m + 1))
            )
            P = lpmv(m, ell, mu)
            basis[ell, m] = norm * P * jnp.exp(1j * m * lon2d)
    return jnp.array(basis), jnp.array(lats), jnp.array(lons), jnp.array(w)


def analysis_grid_to_spec(field_grid: jnp.ndarray, lmax: int = cfg.Lmax):
    """Forward spherical-harmonic transform by quadrature.

    Parameters
    ----------
    field_grid : array, shape (..., nlat, nlon)
        Real or complex grid field.
    lmax : int
        Spectral truncation.
    """
    nlat, nlon = field_grid.shape[-2:]
    basis, lats, lons, w = _precompute_basis(nlat, nlon, lmax)
    lon_weight = 2 * jnp.pi / nlon
    weights = w[:, None] * lon_weight
    def _transform(f):
        integrand = f[None, None, :, :] * jnp.conjugate(basis)
        coeffs = jnp.sum(integrand * weights[None, None, :, :], axis=(-1, -2))
        return coeffs
    return jax.vmap(_transform, in_axes=0)(field_grid) if field_grid.ndim > 2 else _transform(field_grid)


def synthesis_spec_to_grid(coeffs: jnp.ndarray, nlat: int = cfg.nlat, nlon: int = cfg.nlon):
    """Inverse spherical-harmonic synthesis.

    Parameters
    ----------
    coeffs : array, shape (..., lmax+1, lmax+1)
        Spectral coefficients (m>=0 storage).
    """
    lmax = coeffs.shape[-2] - 1
    basis, lats, lons, w = _precompute_basis(nlat, nlon, lmax)
    def _synth(c):
        return jnp.real(jnp.sum(c[:, :, None, None] * basis, axis=(0, 1)))
    return jax.vmap(_synth, in_axes=0)(coeffs) if coeffs.ndim > 2 else _synth(coeffs)


def lap_spec(flm: jnp.ndarray):
    ell = jnp.arange(flm.shape[-2])[:, None]
    factor = -ell * (ell + 1) / (cfg.a ** 2)
    return factor[..., None] * flm


def invert_laplacian(flm: jnp.ndarray):
    ell = jnp.arange(flm.shape[-2])[:, None]
    factor = ell * (ell + 1)
    inv = jnp.where(factor > 0, -cfg.a ** 2 / factor, 0.0)
    return inv[..., None] * flm


def psi_chi_from_zeta_div(zeta_lm: jnp.ndarray, div_lm: jnp.ndarray):
    psi = invert_laplacian(zeta_lm)
    chi = invert_laplacian(div_lm)
    return psi, chi


def uv_from_psi_chi(psi_lm: jnp.ndarray, chi_lm: jnp.ndarray, nlat: int = cfg.nlat, nlon: int = cfg.nlon):
    """Recover winds on the grid from streamfunction and velocity potential."""
    psi = synthesis_spec_to_grid(psi_lm, nlat, nlon)
    chi = synthesis_spec_to_grid(chi_lm, nlat, nlon)
    lats, lons, w = grid.gaussian_grid(nlat, nlon)
    lat2d, lon2d = jnp.meshgrid(jnp.array(lons), jnp.array(lats))
    dlon = 2 * jnp.pi / nlon
    dpsi_dlon = (jnp.roll(psi, -1, axis=-1) - jnp.roll(psi, 1, axis=-1)) / (2 * dlon)
    dchi_dlon = (jnp.roll(chi, -1, axis=-1) - jnp.roll(chi, 1, axis=-1)) / (2 * dlon)
    dlat = lat2d[1, 0] - lat2d[0, 0]
    dpsi_dlat = (jnp.roll(psi, -1, axis=-2) - jnp.roll(psi, 1, axis=-2)) / (2 * dlat)
    dchi_dlat = (jnp.roll(chi, -1, axis=-2) - jnp.roll(chi, 1, axis=-2)) / (2 * dlat)
    cosphi = jnp.cos(jnp.array(lats))[:, None]
    u = (-dpsi_dlat + dchi_dlon) / (cfg.a * cosphi)
    v = (dpsi_dlon + dchi_dlat * 0.0 + dchi_dlat) / cfg.a
    return u, v
