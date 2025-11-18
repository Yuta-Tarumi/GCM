"""Rayleigh damping utilities."""
from __future__ import annotations
import jax.numpy as jnp
from . import config, spectral, vertical
from .state import ModelState


def _rayleigh_mask(cfg: config.ModelConfig) -> jnp.ndarray:
    sigma_full, _ = vertical.sigma_levels(cfg.numerics.nlev)
    return (sigma_full > 0.9).astype(jnp.float64)


def _sponge_mask(cfg: config.ModelConfig) -> jnp.ndarray:
    return vertical.sigma_mask_above_height(cfg.physics.sponge_top_height, cfg.numerics.nlev)


def apply_rayleigh_and_sponge(state: ModelState, cfg: config.ModelConfig | None = None):
    if cfg is None:
        cfg = config.DEFAULT
    low_mask = _rayleigh_mask(cfg)[:, None, None]
    top_mask = _sponge_mask(cfg)[:, None, None]
    tau_low = cfg.physics.rayleigh_low_tau
    tau_top = cfg.physics.sponge_tau
    zeta = spectral.analysis_grid_to_spec(
        -low_mask * spectral.synthesis_spec_to_grid(state.zeta) / tau_low
        - top_mask * spectral.synthesis_spec_to_grid(state.zeta) / tau_top
    )
    div = spectral.analysis_grid_to_spec(
        -low_mask * spectral.synthesis_spec_to_grid(state.div) / tau_low
        - top_mask * spectral.synthesis_spec_to_grid(state.div) / tau_top
    )
    return zeta, div
