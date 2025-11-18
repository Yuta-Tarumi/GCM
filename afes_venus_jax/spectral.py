"""Lightweight pseudo-spectral transform helpers."""
from __future__ import annotations
import jax
import jax.numpy as jnp
from . import config, grid

__all__ = [
    "analysis_grid_to_spec",
    "synthesis_spec_to_grid",
    "apply_two_thirds_filter",
    "lap_spec",
    "invert_laplacian",
    "psi_chi_from_vort_div",
    "uv_from_psi_chi",
]


def _two_thirds_mask(nlon: int) -> jnp.ndarray:
    cutoff = int(((nlon // 2) * 2) / 3)
    rfft_len = nlon // 2 + 1
    mask = jnp.ones((rfft_len,), dtype=jnp.float64)
    return mask.at[cutoff + 1 :].set(0.0)


@jax.jit
def apply_two_thirds_filter(field: jnp.ndarray) -> jnp.ndarray:
    """Apply the 2/3 de-aliasing filter along longitude."""
    nlon = field.shape[-1]
    mask = _two_thirds_mask(nlon)
    fhat = jnp.fft.rfft(field, axis=-1)
    filtered = jnp.fft.irfft(fhat * mask, n=nlon, axis=-1)
    return filtered


@jax.jit
def analysis_grid_to_spec(field_grid: jnp.ndarray) -> jnp.ndarray:
    """Pseudo spectral analysis (identity placeholder)."""
    return field_grid.astype(jnp.complex64)


@jax.jit
def synthesis_spec_to_grid(flm: jnp.ndarray) -> jnp.ndarray:
    """Return grid representation by taking the real part."""
    return jnp.real(flm)


@jax.jit
def lap_spec(flm: jnp.ndarray) -> jnp.ndarray:
    """Apply a finite-difference Laplacian in grid space."""
    field = synthesis_spec_to_grid(flm)
    lats, lons, _ = grid.gaussian_grid()
    dlon = lons[1] - lons[0]
    dphi = lats[1] - lats[0]
    cosphi = grid.cosine_latitudes(lats)
    f_lon = jnp.roll(field, -1, axis=-1) - 2 * field + jnp.roll(field, 1, axis=-1)
    f_lon = f_lon / (config.planet.radius ** 2 * (dlon ** 2))
    f_phi = jnp.roll(field, -1, axis=-2) - 2 * field + jnp.roll(field, 1, axis=-2)
    f_phi = f_phi / (config.planet.radius ** 2 * (dphi ** 2))
    tanphi = jnp.tan(lats)[:, None]
    dfdphi = (jnp.roll(field, -1, axis=-2) - jnp.roll(field, 1, axis=-2)) / (2 * dphi)
    metric = (tanphi / (config.planet.radius ** 2)) * dfdphi
    lap = f_lon / (cosphi[:, None] ** 2) + f_phi + metric
    return analysis_grid_to_spec(jnp.real(lap))


@jax.jit
def invert_laplacian(flm: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Pseudo-inverse Laplacian using Fourier meridional bands."""
    field = synthesis_spec_to_grid(flm)
    fhat = jnp.fft.rfft(field, axis=-1)
    nlon = field.shape[-1]
    m = jnp.arange(fhat.shape[-1])
    k2 = (m / config.planet.radius) ** 2
    inv = jnp.where(k2[None, :] > eps, -1.0 / k2[None, :], 0.0)
    psi_hat = fhat * inv.astype(jnp.complex128)
    psi = jnp.fft.irfft(psi_hat, n=nlon, axis=-1)
    return analysis_grid_to_spec(psi)


@jax.jit
def psi_chi_from_vort_div(zeta_lm: jnp.ndarray, div_lm: jnp.ndarray):
    """Recover streamfunction and velocity potential from vorticity/divergence."""
    psi = invert_laplacian(zeta_lm)
    chi = invert_laplacian(div_lm)
    return psi, chi


def _gradients(field: jnp.ndarray, lats: jnp.ndarray, lons: jnp.ndarray):
    dlon = lons[1] - lons[0]
    dphi = lats[1] - lats[0]
    df_dlon = (jnp.roll(field, -1, axis=-1) - jnp.roll(field, 1, axis=-1)) / (2 * dlon)
    df_dphi = (jnp.roll(field, -1, axis=-2) - jnp.roll(field, 1, axis=-2)) / (2 * dphi)
    return df_dphi, df_dlon


@jax.jit
def uv_from_psi_chi(psi_lm: jnp.ndarray, chi_lm: jnp.ndarray):
    """Compute horizontal wind components from streamfunction and velocity potential."""
    lats, lons, _ = grid.gaussian_grid()
    psi = synthesis_spec_to_grid(psi_lm)
    chi = synthesis_spec_to_grid(chi_lm)
    dpsi_dphi, dpsi_dlon = _gradients(psi, lats, lons)
    dchi_dphi, dchi_dlon = _gradients(chi, lats, lons)
    cosphi = grid.cosine_latitudes(lats)[:, None]
    a = config.planet.radius
    u = -(1.0 / (a * cosphi)) * dpsi_dphi + (1.0 / a) * dchi_dlon
    v = (1.0 / a) * dpsi_dlon + (1.0 / (a * cosphi)) * dchi_dphi
    return u, v
