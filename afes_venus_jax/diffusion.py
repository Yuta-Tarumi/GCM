"""Horizontal hyperdiffusion and vertical mixing."""
from __future__ import annotations
import jax.numpy as jnp
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from . import config, spectral, vertical
from .state import ModelState


def _iterated_laplacian(field: jnp.ndarray, order: int) -> jnp.ndarray:
    result = field
    for _ in range(order):
        result = spectral.lap_spec(result)
    return result


def _laplacian_numpy(field: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    a = config.planet.radius
    dlon = float(lons[1] - lons[0])
    dphi = float(lats[1] - lats[0])
    cosphi = np.clip(np.cos(lats), 1e-7, None)[:, None]
    tanphi = np.tan(lats)[:, None]
    f_lon = (np.roll(field, -1, axis=-1) - 2 * field + np.roll(field, 1, axis=-1)) / (a**2 * dlon**2)
    f_phi = (np.roll(field, -1, axis=-2) - 2 * field + np.roll(field, 1, axis=-2)) / (a**2 * dphi**2)
    dfdphi = (np.roll(field, -1, axis=-2) - np.roll(field, 1, axis=-2)) / (2 * dphi)
    metric = (tanphi / (a**2)) * dfdphi
    return f_lon / (cosphi**2) + f_phi + metric


def _gaussian_grid_numpy(nlat: int, nlon: int) -> tuple[np.ndarray, np.ndarray]:
    mu, _ = leggauss(nlat)
    lats = np.arcsin(mu)
    lons = np.linspace(0.0, 2 * np.pi, nlon, endpoint=False)
    return lats, lons


@lru_cache(maxsize=None)
def _max_hyperdiff_eigenvalue(nlev: int, nlat: int, nlon: int, order: int) -> float:
    """Estimate the largest eigenvalue of the iterated Laplacian on this grid."""

    lats_np, lons_np = _gaussian_grid_numpy(nlat, nlon)
    m = max(1, (nlon // 2) - 1)
    lon_wave = np.sin(m * lons_np)[None, None, :]
    mode = np.broadcast_to(lon_wave, (nlev, nlat, nlon)).astype(np.float64)
    lap = mode.copy()
    for _ in range(order // 2):
        lap = _laplacian_numpy(lap, lats_np, lons_np)
    eig = np.max(np.abs(lap) / np.maximum(np.abs(mode), 1e-12))
    return float(eig)


def _hyperdiff_coefficient(cfg: config.ModelConfig) -> float:
    eig = _max_hyperdiff_eigenvalue(
        cfg.numerics.nlev,
        cfg.numerics.nlat,
        cfg.numerics.nlon,
        cfg.numerics.hyperdiff_order,
    )
    return 1.0 / (cfg.numerics.hyperdiff_tau_smallest * eig)


def hyperdiffusion_tendency(field: jnp.ndarray, cfg: config.ModelConfig | None = None) -> jnp.ndarray:
    if cfg is None:
        cfg = config.DEFAULT
    m = cfg.numerics.hyperdiff_order // 2
    lap = _iterated_laplacian(field, m)
    nu = _hyperdiff_coefficient(cfg)
    return -nu * lap


def apply_hyperdiffusion(state: ModelState, cfg: config.ModelConfig | None = None) -> ModelState:
    if cfg is None:
        cfg = config.DEFAULT
    dt = cfg.numerics.dt
    return ModelState(
        zeta=state.zeta + dt * hyperdiffusion_tendency(state.zeta, cfg),
        div=state.div + dt * hyperdiffusion_tendency(state.div, cfg),
        T=state.T + dt * hyperdiffusion_tendency(state.T, cfg),
        lnps=state.lnps + dt * hyperdiffusion_tendency(state.lnps[None, ...], cfg)[0],
    )


def vertical_diffusion_temperature(state: ModelState, cfg: config.ModelConfig | None = None) -> jnp.ndarray:
    if cfg is None:
        cfg = config.DEFAULT
    Kz = cfg.physics.kz
    heights = vertical.level_heights(cfg.numerics.nlev)
    dz = jnp.gradient(heights)
    T = spectral.synthesis_spec_to_grid(state.T)
    lap = (jnp.roll(T, -1, axis=0) - 2 * T + jnp.roll(T, 1, axis=0)) / (dz[:, None, None] ** 2)
    lap = lap.at[0].set(0.0).at[-1].set(0.0)
    return spectral.analysis_grid_to_spec(Kz * lap)
