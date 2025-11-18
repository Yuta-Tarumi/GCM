"""Horizontal hyperdiffusion and vertical mixing."""
from __future__ import annotations
import jax.numpy as jnp
from . import config, spectral, vertical
from .state import ModelState


def _iterated_laplacian(field: jnp.ndarray, order: int) -> jnp.ndarray:
    result = field
    for _ in range(order):
        result = spectral.lap_spec(result)
    return result


def _hyperdiff_coefficient(cfg: config.ModelConfig) -> float:
    kmax = cfg.numerics.nlon / 2
    return 1.0 / (cfg.numerics.hyperdiff_tau_smallest * (kmax ** cfg.numerics.hyperdiff_order))


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
