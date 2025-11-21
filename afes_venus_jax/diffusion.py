"""Spectral hyperdiffusion operator."""
from __future__ import annotations

import jax.numpy as jnp

import afes_venus_jax.config as cfg


def hyperdiffusion(flm: jnp.ndarray, lmax: int = cfg.Lmax, tau: float = cfg.tau_hdiff, order: int = cfg.order_hdiff):
    ell = jnp.arange(flm.shape[-2])[:, None]
    eig = (ell * (ell + 1) / (cfg.a ** 2)) ** order
    eig_max = (lmax * (lmax + 1) / (cfg.a ** 2)) ** order
    nu = 1.0 / (tau * eig_max)
    return flm - cfg.dt * nu * eig * flm


def apply_diffusion(mstate):
    zeta = hyperdiffusion(mstate.zeta)
    div = hyperdiffusion(mstate.div)
    T = hyperdiffusion(mstate.T)
    return mstate.__class__(zeta, div, T, mstate.lnps)
