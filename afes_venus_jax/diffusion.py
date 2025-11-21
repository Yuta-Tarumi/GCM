"""Spectral hyperdiffusion operator."""
from __future__ import annotations

import jax.numpy as jnp

import afes_venus_jax.config as cfg


def _nu4(order: int, tau: float, lmax: int, dtype) -> float:
    eig_max = (jnp.array(lmax, dtype=dtype) * (lmax + 1) / (cfg.a ** 2)) ** order
    tiny = jnp.finfo(dtype).tiny
    eig_max = jnp.maximum(eig_max, tiny)
    return 1.0 / (tau * eig_max)


def hyperdiffusion(flm: jnp.ndarray, lmax: int = cfg.Lmax, tau: float = cfg.tau_hdiff, order: int = cfg.order_hdiff):
    ell = jnp.arange(flm.shape[-2], dtype=flm.real.dtype)[:, None]
    eig = (ell * (ell + 1) / (cfg.a ** 2)) ** order
    nu = cfg.nu4_hdiff if cfg.nu4_hdiff is not None else _nu4(order, tau, lmax, flm.real.dtype)
    return flm - cfg.dt * nu * eig * flm


def apply_diffusion(mstate):
    zeta = hyperdiffusion(mstate.zeta)
    div = hyperdiffusion(mstate.div)
    T = hyperdiffusion(mstate.T)
    lnps = hyperdiffusion(mstate.lnps, order=cfg.order_hdiff, tau=cfg.tau_hdiff)

    if cfg.tau_div_damp is not None:
        div = _divergence_damping(div)

    return mstate.__class__(zeta, div, T, lnps)


def _divergence_damping(div_spec: jnp.ndarray):
    """Apply scale-selective divergence damping in spectral space."""

    ell = jnp.arange(div_spec.shape[-2], dtype=div_spec.real.dtype)[:, None]
    eig = (ell * (ell + 1) / (cfg.a ** 2)) ** cfg.order_div_damp
    eig_max = (jnp.array(cfg.Lmax, dtype=div_spec.real.dtype) * (cfg.Lmax + 1) / (cfg.a ** 2)) ** cfg.order_div_damp
    tiny = jnp.finfo(div_spec.real.dtype).tiny
    eig_max = jnp.maximum(eig_max, tiny)
    nu = 1.0 / (cfg.tau_div_damp * eig_max)
    return div_spec - cfg.dt * nu * eig * div_spec
