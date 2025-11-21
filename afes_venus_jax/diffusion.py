"""Spectral hyperdiffusion operator."""
from __future__ import annotations

import jax.numpy as jnp

import afes_venus_jax.config as cfg


def hyperdiffusion(flm: jnp.ndarray, lmax: int = cfg.Lmax, tau: float = cfg.tau_hdiff, order: int = cfg.order_hdiff):
    ell = jnp.arange(flm.shape[-2], dtype=flm.real.dtype)[:, None]
    eig = (ell * (ell + 1) / (cfg.a ** 2)) ** order
    eig_max = (jnp.array(lmax, dtype=flm.real.dtype) * (lmax + 1) / (cfg.a ** 2)) ** order
    # Guard against underflow of eig_max in single precision, which would
    # otherwise lead to ``nu=inf`` and NaNs in the filtered fields.
    tiny = jnp.finfo(flm.real.dtype).tiny
    eig_max = jnp.maximum(eig_max, tiny)
    nu = 1.0 / (tau * eig_max)
    return flm - cfg.dt * nu * eig * flm


def apply_diffusion(mstate):
    zeta = hyperdiffusion(mstate.zeta)
    div = hyperdiffusion(mstate.div)
    T = hyperdiffusion(mstate.T)

    if cfg.tau_div_damp is not None:
        div = _divergence_damping(div)

    return mstate.__class__(zeta, div, T, mstate.lnps)


def _divergence_damping(div_spec: jnp.ndarray):
    """Apply scale-selective divergence damping in spectral space."""

    ell = jnp.arange(div_spec.shape[-2], dtype=div_spec.real.dtype)[:, None]
    eig = (ell * (ell + 1) / (cfg.a ** 2)) ** cfg.order_div_damp
    eig_max = (jnp.array(cfg.Lmax, dtype=div_spec.real.dtype) * (cfg.Lmax + 1) / (cfg.a ** 2)) ** cfg.order_div_damp
    tiny = jnp.finfo(div_spec.real.dtype).tiny
    eig_max = jnp.maximum(eig_max, tiny)
    nu = 1.0 / (cfg.tau_div_damp * eig_max)
    return div_spec - cfg.dt * nu * eig * div_spec
