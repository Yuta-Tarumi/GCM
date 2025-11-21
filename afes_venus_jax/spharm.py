"""Spherical-harmonic utilities.

Provides both the reference quadrature-based transforms and an optimised
S2FFT backend that mirrors the high-performance AFES-Venus spectral core
when ``AFES_VENUS_JAX_USE_S2FFT`` is enabled.
"""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import lpmv, gammaln

import afes_venus_jax.config as cfg
import afes_venus_jax.grid as grid

try:
    import s2fft
    from s2fft.sampling import s2_samples

    _HAS_S2FFT = True
except Exception:  # pragma: no cover - optional acceleration dependency
    _HAS_S2FFT = False


@functools.lru_cache(None)
def _precompute_basis(nlat: int, nlon: int, lmax: int):
    lats, lons, w = grid.spectral_grid(nlat, nlon, lmax=lmax)
    lon2d, lat2d = np.meshgrid(lons, lats)
    mu = np.sin(lat2d)
    basis = np.zeros((lmax + 1, lmax + 1, nlat, nlon), dtype=np.complex128)
    for ell in range(lmax + 1):
        for m in range(ell + 1):
            norm = np.exp(
                0.5
                * (np.log(2 * ell + 1) - np.log(4 * np.pi) + gammaln(ell - m + 1) - gammaln(ell + m + 1))
            )
            P = lpmv(m, ell, mu)
            basis[ell, m] = norm * P * np.exp(1j * m * lon2d)
    return jnp.array(basis), jnp.array(lats), jnp.array(lons), jnp.array(w)


def _analysis_s2fft(field_grid: jnp.ndarray, lmax: int) -> jnp.ndarray:
    bandlimit = lmax + 1
    spin = jnp.array(0, dtype=jnp.int32)

    def _forward(f):
        return s2fft.forward_jax(
            f,
            bandlimit,
            sampling=cfg.s2fft_sampling,
            reality=jnp.isrealobj(f),
            spin=spin,
        )

    if field_grid.ndim <= 2:
        coeffs_full = _forward(field_grid)
    else:
        leading = field_grid.shape[:-2]
        flat_grid = field_grid.reshape((-1,) + field_grid.shape[-2:])
        flat_coeffs = jax.vmap(_forward)(flat_grid)
        coeffs_full = flat_coeffs.reshape(leading + (bandlimit, 2 * bandlimit - 1))
    # Discard negative m, retaining the triangular (m>=0) storage used by the
    # rest of the code base.
    return coeffs_full[..., bandlimit - 1 : bandlimit - 1 + bandlimit]


def _analysis_quadrature(field_grid: jnp.ndarray, lmax: int):
    nlat, nlon = field_grid.shape[-2:]
    basis, lats, lons, w = _precompute_basis(nlat, nlon, lmax)
    lon_weight = 2 * jnp.pi / nlon
    weights = w[:, None] * lon_weight

    def _transform(f):
        integrand = f[None, None, :, :] * jnp.conjugate(basis)
        coeffs = jnp.sum(integrand * weights[None, None, :, :], axis=(-1, -2))
        return coeffs

    if field_grid.ndim <= 2:
        return _transform(field_grid)

    leading = field_grid.shape[:-2]
    flat_grid = field_grid.reshape((-1,) + field_grid.shape[-2:])
    flat_spec = jax.vmap(_transform)(flat_grid)
    return flat_spec.reshape(leading + (lmax + 1, lmax + 1))


def analysis_grid_to_spec(field_grid: jnp.ndarray, lmax: int = cfg.Lmax):
    """Forward spherical-harmonic transform using the configured backend."""

    if cfg.use_s2fft and _HAS_S2FFT:
        return _analysis_s2fft(field_grid, lmax)

    return _analysis_quadrature(field_grid, lmax)


def _to_full_m(coeffs: jnp.ndarray) -> jnp.ndarray:
    bandlimit = coeffs.shape[-1]
    if coeffs.ndim == 2:
        coeffs = coeffs[None, ...]

    sign = (-1.0) ** jnp.arange(1, bandlimit)
    reshape = (1,) * (coeffs.ndim - 2) + (1, bandlimit - 1)
    sign = sign.reshape(reshape)
    neg_m = jnp.conjugate(coeffs[..., :, 1:]) * sign
    full = jnp.concatenate([jnp.flip(neg_m, axis=-1), coeffs], axis=-1)

    if full.shape[0] == 1:
        full = full[0]
    return full


def _synthesis_s2fft(coeffs: jnp.ndarray, nlat: int, nlon: int):
    bandlimit = coeffs.shape[-2]
    full_coeffs = _to_full_m(coeffs)
    reality = jnp.isrealobj(coeffs)

    spin = jnp.array(0, dtype=jnp.int32)

    def _inverse(flm):
        return s2fft.inverse_jax(
            flm,
            bandlimit,
            sampling=cfg.s2fft_sampling,
            reality=reality,
            spin=spin,
        )

    if full_coeffs.ndim == 2:
        grid = _inverse(full_coeffs)
    else:
        leading = full_coeffs.shape[:-2]
        flat_coeffs = full_coeffs.reshape((-1,) + full_coeffs.shape[-2:])
        flat_grid = jax.vmap(_inverse)(flat_coeffs)
        grid = flat_grid.reshape(leading + (nlat, nlon))

    return jnp.real(grid)


def _synthesis_quadrature(coeffs: jnp.ndarray, nlat: int, nlon: int):
    lmax = coeffs.shape[-2] - 1
    basis, lats, lons, w = _precompute_basis(nlat, nlon, lmax)
    if coeffs.ndim == 2:
        return jnp.real(jnp.einsum("lm,lmij->ij", coeffs, basis))

    return jnp.real(jnp.einsum("...lm,lmij->...ij", coeffs, basis))


def synthesis_spec_to_grid(coeffs: jnp.ndarray, nlat: int = cfg.nlat, nlon: int = cfg.nlon):
    """Inverse spherical-harmonic synthesis using the configured backend."""

    if cfg.use_s2fft and _HAS_S2FFT:
        return _synthesis_s2fft(coeffs, nlat, nlon)

    return _synthesis_quadrature(coeffs, nlat, nlon)


def lap_spec(flm: jnp.ndarray):
    ell = jnp.arange(flm.shape[-2])[:, None]
    factor = -ell * (ell + 1) / (cfg.a ** 2)
    return factor * flm


def invert_laplacian(flm: jnp.ndarray):
    ell = jnp.arange(flm.shape[-2])[:, None]
    factor = ell * (ell + 1)
    inv = jnp.where(factor > 0, -cfg.a ** 2 / factor, 0.0)
    return inv * flm


def psi_chi_from_zeta_div(zeta_lm: jnp.ndarray, div_lm: jnp.ndarray):
    psi = invert_laplacian(zeta_lm)
    chi = invert_laplacian(div_lm)
    return psi, chi


def uv_from_psi_chi(psi_lm: jnp.ndarray, chi_lm: jnp.ndarray, nlat: int = cfg.nlat, nlon: int = cfg.nlon):
    """Recover winds on the grid from streamfunction and velocity potential."""
    psi = synthesis_spec_to_grid(psi_lm, nlat, nlon)
    chi = synthesis_spec_to_grid(chi_lm, nlat, nlon)
    lats, lons, w = grid.spectral_grid(nlat, nlon)
    lat_axis = jnp.array(lats)
    lat2d, lon2d = jnp.meshgrid(lat_axis, jnp.array(lons), indexing="ij")
    dlon = 2 * jnp.pi / nlon
    dpsi_dlon = (jnp.roll(psi, -1, axis=-1) - jnp.roll(psi, 1, axis=-1)) / (2 * dlon)
    dchi_dlon = (jnp.roll(chi, -1, axis=-1) - jnp.roll(chi, 1, axis=-1)) / (2 * dlon)

    dpsi_dlat = jnp.gradient(psi, lat_axis, axis=-2)
    dchi_dlat = jnp.gradient(chi, lat_axis, axis=-2)

    dpsi_dlat = dpsi_dlat.at[0].set(0.0)
    dpsi_dlat = dpsi_dlat.at[-1].set(0.0)
    dchi_dlat = dchi_dlat.at[0].set(0.0)
    dchi_dlat = dchi_dlat.at[-1].set(0.0)

    cosphi = jnp.cos(lat_axis)[:, None]
    cosphi_safe = jnp.clip(cosphi, 1e-8, None)

    u = (-dpsi_dlat + dchi_dlon) / (cfg.a * cosphi_safe)
    v = (dpsi_dlon + dchi_dlat * 0.0 + dchi_dlat) / cfg.a
    return u, v
