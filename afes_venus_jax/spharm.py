"""True spherical-harmonic operators with Gaussian quadrature."""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp
from .config import Config
from .grid import gaussian_grid


def _norm_factor(ell: jnp.ndarray, m: jnp.ndarray):
    return jnp.sqrt(
        ((2 * ell + 1) / (4 * jnp.pi))
        * jax.scipy.special.factorial(ell - m)
        / jax.scipy.special.factorial(ell + m)
    )


def _associated_legendre(cfg: Config):
    return _associated_legendre_cached(cfg.Lmax, cfg.nlat)


@functools.lru_cache(maxsize=None)
def _associated_legendre_cached(Lmax: int, nlat: int):
    cfg = Config(Lmax=Lmax, nlat=nlat, nlon=2 * nlat)
    lats, _, _ = gaussian_grid(cfg)
    mu = jnp.sin(lats)
    cos_lat = jnp.cos(lats)
    L = cfg.Lmax
    P = jnp.zeros((L + 1, L + 1, cfg.nlat))

    # Initial diagonal P_m^m
    P = P.at[0, 0].set(1.0)
    for m in range(1, L + 1):
        sign = -1.0 if m % 2 else 1.0
        P = P.at[m, m].set(sign * jnp.prod(2 * jnp.arange(1, m + 1) - 1) * (1 - mu**2) ** (m / 2))
    # First off-diagonal
    for m in range(0, L):
        P = P.at[m + 1, m].set((2 * m + 1) * mu * P[m, m])
    # Upward recursion for remaining rows
    for m in range(0, L + 1):
        for ell in range(m + 2, L + 1):
            P = P.at[ell, m].set(((2 * ell - 1) * mu * P[ell - 1, m] - (ell + m - 1) * P[ell - 2, m]) / (ell - m))

    # Derivative w.r.t. latitude φ using dP/dφ = cos φ * dP/dμ
    dP_dmu = jnp.zeros_like(P)
    denom = jnp.clip(1 - mu**2, 1e-12)
    for m in range(0, L + 1):
        for ell in range(m, L + 1):
            if ell == 0:
                dP_dmu = dP_dmu.at[ell, m].set(0.0)
            else:
                dP_dmu = dP_dmu.at[ell, m].set((ell * mu * P[ell, m] - (ell + m) * P[ell - 1, m]) / denom)
    dP_dphi = cos_lat * dP_dmu
    norms = _norm_factor(jnp.arange(L + 1)[:, None], jnp.arange(L + 1)[None, :])
    return P, dP_dphi, norms


def analysis_grid_to_spec(field: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Grid → spherical-harmonic coefficients (ell, m>=0)."""

    lats, _, weights = gaussian_grid(cfg)
    P, _, norms = _associated_legendre(cfg)
    fft_field = jnp.fft.fft(field, axis=-1)
    delta_lon = 2 * jnp.pi / cfg.nlon
    coeffs = []
    for m in range(cfg.Lmax + 1):
        # Fourier component for this zonal wavenumber
        Fm = delta_lon * fft_field[..., m]
        y_lat = norms[:, m][:, None] * P[:, m]
        coeffs_m = jnp.einsum('...i,i,li->...l', Fm, weights, y_lat)
        coeffs.append(coeffs_m)
    return jnp.stack(coeffs, axis=-1)


def synthesis_spec_to_grid(flm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    """Spherical-harmonic coefficients → grid field."""

    lats, _, _ = gaussian_grid(cfg)
    P, _, norms = _associated_legendre(cfg)
    lat_contribs = []
    for m in range(cfg.Lmax + 1):
        y_lat = norms[:, m][:, None] * P[:, m]
        lat_contribs.append(jnp.einsum('...l,li->...i', flm[..., m], y_lat))
    lat_contribs = jnp.stack(lat_contribs, axis=-1)

    # Build complex Fourier spectrum for inverse FFT
    lon_spec = jnp.zeros(flm.shape[:-2] + (cfg.nlat, cfg.nlon), dtype=jnp.complex128)
    for m in range(cfg.Lmax + 1):
        lon_spec = lon_spec.at[..., :, m].set(cfg.nlon * lat_contribs[..., m])
        if m > 0:
            lon_spec = lon_spec.at[..., :, -m].set(cfg.nlon * jnp.conj(lat_contribs[..., m]) * ((-1) ** m))
    grid = jnp.fft.ifft(lon_spec, axis=-1).real
    return grid


def lap_spec(flm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    ell = jnp.arange(cfg.Lmax + 1)
    eig = -(ell * (ell + 1) / cfg.a**2)
    return flm * eig[:, None]


def invert_laplacian(flm: jnp.ndarray, cfg: Config) -> jnp.ndarray:
    ell = jnp.arange(cfg.Lmax + 1)
    eig = ell * (ell + 1) / cfg.a**2
    eig = jnp.where(eig == 0, jnp.inf, eig)
    return -flm / eig[:, None]


def psi_chi_from_vort_div(zeta_lm: jnp.ndarray, div_lm: jnp.ndarray, cfg: Config):
    psi = invert_laplacian(zeta_lm, cfg)
    chi = invert_laplacian(div_lm, cfg)
    return psi, chi


def _spectral_derivatives(flm: jnp.ndarray, cfg: Config):
    """Return (d/dlon, d/dphi) of a scalar spectral field on the grid."""

    lats, _, _ = gaussian_grid(cfg)
    P, dP_dphi, norms = _associated_legendre(cfg)
    lat_dlon = []
    lat_dphi = []
    for m in range(cfg.Lmax + 1):
        y_lat = norms[:, m][:, None] * P[:, m]
        dy_dphi = norms[:, m][:, None] * dP_dphi[:, m]
        lat_part = jnp.einsum('...l,li->...i', flm[..., m], y_lat)
        lat_part_phi = jnp.einsum('...l,li->...i', flm[..., m], dy_dphi)
        lat_dlon.append(1j * m * lat_part)
        lat_dphi.append(lat_part_phi)
    lat_dlon = jnp.stack(lat_dlon, axis=-1)
    lat_dphi = jnp.stack(lat_dphi, axis=-1)

    lon_spec_dlon = jnp.zeros(flm.shape[:-2] + (cfg.nlat, cfg.nlon), dtype=jnp.complex128)
    lon_spec_dphi = jnp.zeros_like(lon_spec_dlon)
    for m in range(cfg.Lmax + 1):
        lon_spec_dlon = lon_spec_dlon.at[..., :, m].set(cfg.nlon * lat_dlon[..., m])
        lon_spec_dphi = lon_spec_dphi.at[..., :, m].set(cfg.nlon * lat_dphi[..., m])
        if m > 0:
            factor = ((-1) ** m)
            lon_spec_dlon = lon_spec_dlon.at[..., :, -m].set(cfg.nlon * jnp.conj(lat_dlon[..., m]) * factor)
            lon_spec_dphi = lon_spec_dphi.at[..., :, -m].set(cfg.nlon * jnp.conj(lat_dphi[..., m]) * factor)

    dlon_grid = jnp.fft.ifft(lon_spec_dlon, axis=-1).real
    dphi_grid = jnp.fft.ifft(lon_spec_dphi, axis=-1).real
    return dlon_grid, dphi_grid


def uv_from_psi_chi(psi_lm: jnp.ndarray, chi_lm: jnp.ndarray, cfg: Config):
    dchi_dlon, dchi_dphi = _spectral_derivatives(chi_lm, cfg)
    dpsi_dlon, dpsi_dphi = _spectral_derivatives(psi_lm, cfg)

    lats, _, _ = gaussian_grid(cfg)
    cos_lat = jnp.cos(lats)[:, None]
    metric_lon = 1.0 / (cfg.a * cos_lat)
    metric_lat = 1.0 / cfg.a
    u = metric_lon * dchi_dlon - metric_lat * dpsi_dphi
    v = metric_lat * dchi_dphi + metric_lon * dpsi_dlon
    return u, v


def vort_div_from_uv(u: jnp.ndarray, v: jnp.ndarray, cfg: Config):
    dudlon, _ = scalar_gradients(u, cfg)
    dvdlon, _ = scalar_gradients(v, cfg)

    lats, _, _ = gaussian_grid(cfg)
    cos_lat = jnp.cos(lats)[:, None]

    _, d_u_cos_dphi = scalar_gradients(u * cos_lat, cfg)
    _, d_v_cos_dphi = scalar_gradients(v * cos_lat, cfg)

    vort = (dvdlon - d_u_cos_dphi) / (cfg.a * cos_lat)
    div = (dudlon + d_v_cos_dphi) / (cfg.a * cos_lat)
    return analysis_grid_to_spec(vort, cfg), analysis_grid_to_spec(div, cfg)


def scalar_gradients(field: jnp.ndarray, cfg: Config):
    """Compute longitude/latitude derivatives of a grid scalar via SH transforms."""

    return _spectral_derivatives(analysis_grid_to_spec(field, cfg), cfg)
