"""Newtonian relaxation towards a prescribed VIRA-like profile."""
from __future__ import annotations
import jax.numpy as jnp
from .. import config, spectral, vertical


def reference_temperature(cfg: config.ModelConfig | None = None) -> jnp.ndarray:
    if cfg is None:
        cfg = config.DEFAULT
    heights = vertical.level_heights(cfg.numerics.nlev)
    t_surface = 350.0
    t_top = 170.0
    return t_top + (t_surface - t_top) * jnp.exp(-heights / 17_000.0)


def relaxation_timescale(cfg: config.ModelConfig | None = None) -> jnp.ndarray:
    if cfg is None:
        cfg = config.DEFAULT
    heights = vertical.level_heights(cfg.numerics.nlev)
    anchors = jnp.array([0.0, 10e3, 40e3, 80e3, 120e3])
    tau_days = jnp.array([1e4, 5e3, 500.0, 10.0, 0.1])
    tau = jnp.interp(heights, anchors, tau_days)
    return tau * 86400.0


def cooling_tendency(state, cfg: config.ModelConfig | None = None) -> jnp.ndarray:
    if cfg is None:
        cfg = config.DEFAULT
    T = spectral.synthesis_spec_to_grid(state.T)
    T_eq = reference_temperature(cfg)[:, None, None]
    tau = relaxation_timescale(cfg)[:, None, None]
    tendency = (T_eq - T) / tau
    return spectral.analysis_grid_to_spec(tendency)
