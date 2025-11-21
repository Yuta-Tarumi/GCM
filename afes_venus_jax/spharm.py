"""Spherical harmonic transforms via explicit matrices."""
from __future__ import annotations

import functools
import numpy as np
import jax.numpy as jnp
from scipy.special import sph_harm

from .config import ModelConfig
from .grid import gaussian_grid

Array = jnp.ndarray


def spectral_shape(cfg: ModelConfig):
    return (cfg.Lmax + 1, 2 * cfg.Lmax + 1)


@functools.lru_cache(None)
def precompute_basis(cfg: ModelConfig):
    lats, lons, w = gaussian_grid(cfg)
    theta = np.pi / 2 - np.asarray(lats)
    phi = np.asarray(lons)
    nlat, nlon = len(lats), len(lons)
    w_dphi = np.repeat(np.asarray(w) * (2 * np.pi / nlon), nlon)
    sqrt_w = np.sqrt(w_dphi)

    lm_list = []
    for l in range(cfg.Lmax + 1):
        for m in range(-l, l + 1):
            lm_list.append((l, m))
    ncoeff = len(lm_list)

    Y = np.zeros((nlat * nlon, ncoeff), dtype=np.complex128)
    for idx, (l, m) in enumerate(lm_list):
        Y[:, idx] = sph_harm(m, l, phi[None, :], theta[:, None]).reshape(-1)

    # Compute pseudoinverse with normal equations to reduce SVD workspace.
    # Y_w has shape (nlat * nlon, ncoeff); the Gram matrix is only
    # (ncoeff, ncoeff), so this path trims peak memory considerably while
    # keeping the same least-squares solution.
    Y_w = sqrt_w[:, None] * Y
    gram = Y_w.conj().T @ Y_w
    pinv = np.linalg.solve(gram + 1e-12 * np.eye(gram.shape[0]), Y_w.conj().T)
    l_idx = np.array([l for l, _ in lm_list], dtype=np.int32)
    m_idx = np.array([m for _, m in lm_list], dtype=np.int32) + cfg.Lmax
    return (
        jnp.asarray(Y.T),
        jnp.asarray(pinv.T),
        jnp.asarray(sqrt_w),
        jnp.asarray(lats),
        jnp.asarray(lons),
        jnp.asarray(l_idx),
        jnp.asarray(m_idx),
    )


def analysis_grid_to_spec(field_grid: Array, cfg: ModelConfig) -> Array:
    YT, pinvT, sqrt_w, _, _, l_idx, m_idx = precompute_basis(cfg)
    squeezed = False
    if field_grid.ndim == 2:
        field_grid = field_grid[None, ...]
        squeezed = True
    flat = field_grid.reshape(-1, cfg.nlat * cfg.nlon)
    coeff_mat = (flat * sqrt_w) @ pinvT
    spec = jnp.zeros(flat.shape[:-1] + spectral_shape(cfg), dtype=jnp.complex128)
    spec = spec.at[..., l_idx, m_idx].set(coeff_mat)
    if squeezed:
        spec = spec[0]
    return spec


def synthesis_spec_to_grid(flm: Array, cfg: ModelConfig) -> Array:
    YT, _, sqrt_w, _, _, l_idx, m_idx = precompute_basis(cfg)
    leading = flm.shape[:-2]
    coeff = flm[..., l_idx, m_idx].reshape((-1, l_idx.shape[0]))
    flat = coeff @ YT
    grid = flat.reshape(leading + (cfg.nlat, cfg.nlon))
    return grid


def lap_spec(flm: Array, cfg: ModelConfig) -> Array:
    l = jnp.arange(cfg.Lmax + 1, dtype=jnp.float64)
    factor = -l * (l + 1) / (cfg.a ** 2)
    return flm * factor[:, None]


def invert_laplacian(flm: Array, cfg: ModelConfig) -> Array:
    l = jnp.arange(cfg.Lmax + 1, dtype=jnp.float64)
    factor = jnp.where(l > 0, -1.0 / (l * (l + 1)), 0.0)
    return flm * factor[:, None] * (cfg.a ** 2)


def psi_chi_from_zeta_div(zeta: Array, div: Array, cfg: ModelConfig):
    psi = invert_laplacian(zeta, cfg)
    chi = invert_laplacian(div, cfg)
    return psi, chi


def _lat_deltas(lats: Array) -> Array:
    dl = jnp.empty_like(lats)
    dl = dl.at[0].set(lats[1] - lats[0])
    dl = dl.at[1:-1].set(0.5 * (lats[2:] - lats[:-2]))
    dl = dl.at[-1].set(lats[-1] - lats[-2])
    return dl


def uv_from_psi_chi(psi: Array, chi: Array, cfg: ModelConfig, lats: Array | None = None, lons: Array | None = None):
    if lats is None or lons is None:
        lats, lons, _ = gaussian_grid(cfg)
    lats = jnp.asarray(lats)
    lons = jnp.asarray(lons)
    psi_g = synthesis_spec_to_grid(psi, cfg)
    chi_g = synthesis_spec_to_grid(chi, cfg)

    dphi = 2 * jnp.pi / cfg.nlon
    dlat_arr = _lat_deltas(lats)
    dlat_b = dlat_arr[:, None]

    dpsi_dlon = (jnp.roll(psi_g, -1, axis=-1) - jnp.roll(psi_g, 1, axis=-1)) / (2 * dphi)
    dchi_dlon = (jnp.roll(chi_g, -1, axis=-1) - jnp.roll(chi_g, 1, axis=-1)) / (2 * dphi)

    dpsi_dlat = (jnp.roll(psi_g, -1, axis=-2) - jnp.roll(psi_g, 1, axis=-2)) / (2 * dlat_b)
    dchi_dlat = (jnp.roll(chi_g, -1, axis=-2) - jnp.roll(chi_g, 1, axis=-2)) / (2 * dlat_b)

    coslat = jnp.cos(lats)[:, None]
    u = (1.0 / (cfg.a * coslat)) * dchi_dlon - (1.0 / cfg.a) * dpsi_dlat
    v = (1.0 / cfg.a) * dchi_dlat + (1.0 / (cfg.a * coslat)) * dpsi_dlon
    return u, v
