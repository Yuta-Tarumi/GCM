"""Simplified solar heating for Venus."""
from __future__ import annotations
import jax.numpy as jnp
from .. import config, grid, spectral, vertical


def _solar_cosine(lats: jnp.ndarray, lons: jnp.ndarray, time_s: float, cfg: config.ModelConfig) -> jnp.ndarray:
    omega_solar = cfg.planet.solar_day_rate
    subsolar_lon = (omega_solar * time_s) % (2 * jnp.pi)
    decl = jnp.deg2rad(cfg.planet.obliquity_deg)
    cos_mu = (
        jnp.sin(lats)[:, None] * jnp.sin(decl)
        + jnp.cos(lats)[:, None] * jnp.cos(decl) * jnp.cos(lons[None, :] - subsolar_lon)
    )
    return jnp.clip(cos_mu, 0.0, 1.0)


def _vertical_weights(profile: str, cfg: config.ModelConfig) -> jnp.ndarray:
    heights = vertical.level_heights(cfg.numerics.nlev)
    if profile == "uniform_50_80km":
        mask = jnp.logical_and(heights >= 50_000.0, heights <= 80_000.0)
        weights = mask.astype(jnp.float64)
    elif profile == "tomasko1980":
        anchors = jnp.array([50e3, 55e3, 60e3, 65e3, 70e3, 80e3, 90e3])
        shape = jnp.array([0.1, 0.2, 0.25, 0.2, 0.15, 0.08, 0.02])
        weights = jnp.interp(heights, anchors, shape, left=0.0, right=0.0)
    else:
        weights = jnp.ones_like(heights)
    total = jnp.sum(weights)
    weights = jnp.where(total > 0, weights / total, jnp.ones_like(weights) / weights.size)
    return weights


def solar_heating_tendency(time_s: float, cfg: config.ModelConfig | None = None) -> jnp.ndarray:
    """Return spectral solar heating tendency for temperature."""
    if cfg is None:
        cfg = config.DEFAULT
    lats, lons, _ = grid.gaussian_grid(cfg.numerics.nlat, cfg.numerics.nlon)
    cos_mu = _solar_cosine(lats, lons, time_s, cfg)
    amp = cfg.physics.solar_peak_heating_K_per_day / 86400.0
    heating_surface = amp * cos_mu
    weights = _vertical_weights(cfg.physics.solar_profile, cfg)[:, None, None]
    heating = weights * heating_surface[None, ...]
    heating = jnp.maximum(heating, 0.0)
    return spectral.analysis_grid_to_spec(heating)
