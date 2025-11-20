"""Lightweight spherical spectral operators using FFTs.

This module mimics a spectral transform interface while using separable FFTs
as a stand-in for true spherical harmonics. The implementation is sufficient
for testing and pedagogy while keeping the code JIT friendly.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import Config


def _fftfreq(n, d=1.0):
    return jnp.fft.fftfreq(n, d) * 2 * jnp.pi


def analysis_grid_to_spec(field: jnp.ndarray) -> jnp.ndarray:
    """Forward transform grid → spectral using 2D FFT.

    Parameters
    ----------
    field: array_like
        Grid field with shape (..., nlat, nlon).
    """
    return jnp.fft.fft2(field, axes=(-2, -1)) / field.shape[-2] / field.shape[-1]


def synthesis_spec_to_grid(flm: jnp.ndarray, nlat: int | None = None, nlon: int | None = None) -> jnp.ndarray:
    """Inverse transform spectral → grid using 2D FFT."""
    if nlat is None:
        nlat = flm.shape[-2]
    if nlon is None:
        nlon = flm.shape[-1]
    return jnp.fft.ifft2(flm * nlat * nlon, axes=(-2, -1)).real


def lap_spec(flm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Apply a Laplacian in spectral space."""
    nlat, nlon = flm.shape[-2], flm.shape[-1]
    k_lat = _fftfreq(nlat, d=jnp.pi / nlat)
    k_lon = _fftfreq(nlon, d=2 * jnp.pi / nlon)
    k2 = k_lat[:, None] ** 2 + k_lon[None, :] ** 2
    return -(k2 / cfg.a**2) * flm


def invert_laplacian(flm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Invert the Laplacian with zero-mean regularisation."""
    nlat, nlon = flm.shape[-2], flm.shape[-1]
    k_lat = _fftfreq(nlat, d=jnp.pi / nlat)
    k_lon = _fftfreq(nlon, d=2 * jnp.pi / nlon)
    k2 = k_lat[:, None] ** 2 + k_lon[None, :] ** 2
    return jnp.where(k2 == 0, 0.0, -flm / (k2 / cfg.a**2))


def psi_chi_from_vort_div(zeta_lm: jnp.ndarray, div_lm: jnp.ndarray, cfg: Config):
    psi = invert_laplacian(zeta_lm, cfg)
    chi = invert_laplacian(div_lm, cfg)
    return psi, chi


def uv_from_psi_chi(psi_lm: jnp.ndarray, chi_lm: jnp.ndarray, cfg: Config):
    """Recover winds from streamfunction and velocity potential.

    This uses spectral derivatives under a doubly-periodic approximation; it is
    not a full spin-harmonic gradient but is sufficient for verifying algebraic
    consistency between vorticity/divergence and winds.
    """
    nlat, nlon = psi_lm.shape[-2], psi_lm.shape[-1]
    k_lat = _fftfreq(nlat, d=jnp.pi / nlat)
    k_lon = _fftfreq(nlon, d=2 * jnp.pi / nlon)

    def grad(flm):
        return synthesis_spec_to_grid(1j * k_lon[None, :] * flm, nlat, nlon), synthesis_spec_to_grid(
            1j * k_lat[:, None] * flm, nlat, nlon
        )

    dchi_dlon, dchi_dlat = grad(chi_lm)
    dpsi_dlon, dpsi_dlat = grad(psi_lm)
    u = (1.0 / cfg.a) * (dchi_dlon - dpsi_dlat)
    v = (1.0 / cfg.a) * (dchi_dlat + dpsi_dlon)
    return u, v


def vort_div_from_uv(u: jnp.ndarray, v: jnp.ndarray, cfg: Config):
    u_lm = analysis_grid_to_spec(u)
    v_lm = analysis_grid_to_spec(v)
    nlat, nlon = u.shape[-2], u.shape[-1]
    k_lat = _fftfreq(nlat, d=jnp.pi / nlat)
    k_lon = _fftfreq(nlon, d=2 * jnp.pi / nlon)
    vort_lm = 1j * k_lon[None, :] * v_lm - 1j * k_lat[:, None] * u_lm
    div_lm = 1j * k_lon[None, :] * u_lm + 1j * k_lat[:, None] * v_lm
    return vort_lm, div_lm
