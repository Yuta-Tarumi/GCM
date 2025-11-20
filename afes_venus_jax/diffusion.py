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


def hyperdiffusion_timescale(cfg: Config, ell_fraction: float) -> float:
    """Return the e-folding time [s] at a given fraction of the max wavenumber.

    The model sets the hyperdiffusion coefficient so that the smallest resolved
    scale (``ell = ellmax``) decays with ``cfg.tau_hdiff``.  At lower wavenumber
    ``ell = ell_fraction * ellmax``, the decay time scales as

    ``tau(ell) = cfg.tau_hdiff * (1 / ell_fraction) ** (2 * cfg.order_hdiff)``.

    Parameters
    ----------
    cfg
        Model configuration.
    ell_fraction
        Fraction of the spectral radius (``0 < ell_fraction <= 1``).

    Returns
    -------
    float
        E-folding time [s] implied by the current hyperdiffusion settings.
    """

    ell_fraction = jnp.clip(jnp.asarray(ell_fraction), 1e-12, 1.0)
    return float(cfg.tau_hdiff * (1.0 / ell_fraction) ** (2 * cfg.order_hdiff))


def apply_diffusion(state, cfg: Config):
    coeff = diffusion_operator(cfg, state.zeta.shape[-2], state.zeta.shape[-1])
    factor = 1.0 + coeff * cfg.dt
    zeta = state.zeta * factor
    div = state.div * factor
    T = state.T * factor
    return state.__class__(zeta=zeta, div=div, T=T, lnps=state.lnps)
