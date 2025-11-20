"""Spectral hyperdiffusion utilities."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from .config import Config


def diffusion_operator(cfg: Config, nlat: int, nlon: int):
    """Precompute spectral hyperdiffusion coefficient field."""
    k_lat = jnp.fft.fftfreq(nlat, d=jnp.pi / nlat) * 2 * jnp.pi
    k_lon = jnp.fft.fftfreq(nlon, d=2 * jnp.pi / nlon) * 2 * jnp.pi
    ell = jnp.sqrt(k_lat[:, None] ** 2 + k_lon[None, :] ** 2)
    ellmax = jnp.maximum(jnp.max(ell), 1e-12)
    nu = 1.0 / (cfg.tau_hdiff * (ellmax / cfg.a) ** (2 * cfg.order_hdiff))
    coeff = -nu * (ell / cfg.a) ** (2 * cfg.order_hdiff)
    coeff = jnp.where(ell == 0, 0.0, coeff)
    return coeff


def apply_diffusion(state, cfg: Config):
    coeff = diffusion_operator(cfg, state.zeta.shape[-2], state.zeta.shape[-1])
    factor = 1.0 + coeff * cfg.dt
    zeta = state.zeta * factor
    div = state.div * factor
    T = state.T * factor
    return state.__class__(zeta=zeta, div=div, T=T, lnps=state.lnps)
